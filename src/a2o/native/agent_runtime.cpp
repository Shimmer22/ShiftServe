#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
static int batch_size = 512;
static int context_size = 2048;
static std::string actual_vision_backend;
static void vision_log(ggml_log_level, const char * message, void *) {
    std::fputs(message,stderr);
    std::string text(message), marker="CLIP using ";
    auto start=text.find(marker);
    if(start!=std::string::npos) {
        start+=marker.size(); auto end=text.find(" backend",start);
        if(end!=std::string::npos) actual_vision_backend=text.substr(start,end-start);
    }
}
// Diagnostic only: buffer placement is not an operator-time attribution.
static bool trace_vision_node(ggml_tensor * tensor, bool ask, void *) {
    if (ask) {
        auto base=tensor;
        while(!base->buffer && base->view_src) base=base->view_src;
        std::cerr << "VISION_NODE " << ggml_op_name(tensor->op) << " " << tensor->name
                  << " buffer=" << (base->buffer?ggml_backend_buffer_name(base->buffer):"unknown")
                  << " shape=" << tensor->ne[0] << "," << tensor->ne[1]
                  << "," << tensor->ne[2] << "," << tensor->ne[3] << '\n';
    }
    return false;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
static void require(bool ok, const std::string & msg) { if (!ok) throw std::runtime_error(msg); }
static std::string read(const std::string & path) {
    std::ifstream f(path); require(bool(f), "cannot read " + path);
    return {std::istreambuf_iterator<char>(f), {}};
}
static std::string json(const std::string & s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c < 32) { char b[7]; snprintf(b, sizeof b, "\\u%04x", c); o << b; }
        else o << c;
    }
    return o.str() + '"';
}

static std::map<std::string, long long> resource_sample() {
    struct rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage)==0, "getrusage failed");
    std::map<std::string, long long> sample={
        {"minor_faults", usage.ru_minflt}, {"major_faults", usage.ru_majflt}};
    std::ifstream status("/proc/self/status");
    require(bool(status), "cannot read process memory status");
    std::string line;
    while(std::getline(status,line)) {
        std::istringstream in(line); std::string key; long long value;
        in >> key;
        if(key=="VmRSS:" || key=="VmSwap:" || key=="RssAnon:" || key=="RssFile:") {
            in >> value; sample[key.substr(0,key.size()-1)+"_kib"]=value;
        }
    }
    return sample;
}

#include "backend/engine.h"
#include "backend/inference.h"
#include "state/sequence.h"
#include "state/snapshot.h"
#include "vision/prepared_image.h"

#include "session/replay_session.h"
#include "profiling/prefill.h"
#include "profiling/paths.h"
#include "session/tool_session.h"

int main(int argc, char ** argv) { try {
    std::map<std::string, std::string> a;
    for (int i=1; i<argc; i+=2) { require(i+1<argc, "arguments require values"); a[argv[i]]=argv[i+1]; }
    auto get = [&](std::string k, std::string d="") { auto it=a.find(k); return it==a.end()?d:it->second; };
    std::string mode=get("--mode", "cpu"), policy=get("--policy", "none"), fmt=get("--format", "chat");
    bool hybrid=mode=="hybrid", npu=mode=="npu", pre=get("--pre-move", "0")=="1";
    batch_size=std::stoi(get("--batch-size", "512"));
    context_size=std::stoi(get("--ctx-size", "2048"));
    require(batch_size>0 && batch_size<=512, "batch-size must be within [1,512]");
    bool cpu_repack=get("--cpu-repack", "1")=="1";
    require(mode=="cpu" || npu || hybrid, "mode must be cpu, npu, hybrid");
    require(policy=="none" || policy=="recent", "policy must be none or recent");
    require(fmt=="chat" || fmt=="raw", "format must be chat or raw");
    require(!pre || hybrid, "pre-move requires hybrid mode");
    int max_tokens=std::stoi(get("--max-tokens", "64"));
    require(max_tokens>0, "max-tokens must be positive");
    std::string system=read(get("--system")), history=read(get("--history")), previous=read(get("--previous"));
    std::string prompt=read(get("--prompt")), image=get("--image");
    std::string vision_embeddings=get("--vision-embeddings");
    bool has_history=!history.empty();
    std::string vision=get("--vision-backend", "auto");
    if(vision=="auto") vision=(npu||hybrid)?"opencl":"cpu";
    require(vision=="cpu" || vision=="opencl" || vision=="htp", "vision-backend must be auto, cpu, opencl, htp");
    std::string vision_prepare=get("--vision-prepare", "arrival");
    require(vision_prepare=="arrival" || vision_prepare=="a2o", "invalid vision-prepare");
    bool trace_vision=get("--trace-vision", "0")=="1";
    std::string vision_flash=get("--vision-flash", "auto");
    require(vision_flash=="auto" || vision_flash=="off" || vision_flash=="on", "invalid vision-flash");
    require(fmt=="chat" || image.empty(), "raw diagnostic uses text only");
    ggml_backend_load_all();
    if(vision!="cpu") {
        const char * device=vision=="htp"?"HTP0":"GPUOpenCL";
        require(ggml_backend_dev_by_name(device),std::string("vision device unavailable: ")+device);
        setenv("MTMD_BACKEND_DEVICE",device,1);
        if(vision=="htp") {
            // Ask the installed binary, not the potentially different shared source tree.
            auto ctx=ggml_init({65536,nullptr,true}); require(ctx,"capability probe allocation failed");
            auto weights=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,1280,1280);
            for(int rows:{560,1024,1025,2240}) {
                auto input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1280,rows);
                auto output=ggml_mul_mat(ctx,weights,input);
                std::cerr << "VISION_CAP F16_LINEAR rows=" << rows << " supported="
                          << ggml_backend_dev_supports_op(ggml_backend_dev_by_name(device),output) << '\n';
            }
            ggml_free(ctx);
        }
    } else { unsetenv("MTMD_BACKEND_DEVICE"); }
    if(get("--tool-session")=="1") return run_tool_session(a);
    if(get("--path-profile")=="1") return run_paths(a);
    if(!get("--profile-history").empty()) return run_profile(a);
    if(!get("--session").empty()) return run_session(a);
    std::map<std::string,double> times;
    std::map<std::string,std::map<std::string,long long>> resources;
    auto sample=[&](const std::string & stage) {
        auto value=resource_sample();
        std::cerr << "PHASE " << stage << " rss_kib=" << value["VmRSS_kib"]
                  << " swap_kib=" << value["VmSwap_kib"] << " major_faults=" << value["major_faults"] << '\n';
        resources[stage]=std::move(value);
    };
    double t=now_ms();
    Engine home(get("--model"), npu, cpu_repack);
    std::unique_ptr<Engine> accelerator;
    if(hybrid) accelerator=std::make_unique<Engine>(get("--model"), true, cpu_repack);
    times["model_context_init_ms"]=now_ms()-t;
    sample("contexts_ready");
    Engine & pf=hybrid?*accelerator:home;
    mtmd_context * mm=nullptr;
    auto init_vision=[&]() {
        if(image.empty()) return;
        double start=now_ms(); auto mp=mtmd_context_params_default(); mp.use_gpu=vision!="cpu";
        mp.flash_attn_type=vision_flash=="off"?LLAMA_FLASH_ATTN_TYPE_DISABLED:
            (vision_flash=="on"?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_AUTO);
        mp.n_threads=4; mp.warmup=false; mp.print_timings=true;
        if(trace_vision) mp.cb_eval=trace_vision_node;
        actual_vision_backend.clear(); mtmd_helper_log_set(vision_log,nullptr);
        mm=mtmd_init_from_file(get("--mmproj").c_str(),pf.model,mp); require(mm,"mmproj init failed");
        std::string expected=vision=="htp"?"HTP0":(vision=="opencl"?"OpenCL":"CPU");
        require(actual_vision_backend==expected,"vision backend mismatch: expected "+expected+", got "+actual_vision_backend);
        times["vision_init_ms"]=now_ms()-start;
        sample("vision_ready");
    };
    const llama_vocab * vocab=llama_model_get_vocab(home.model);
    int nv=llama_vocab_n_tokens(vocab);
    if(!image.empty()) prompt=std::string(mtmd_default_marker())+"\n"+prompt;
    std::vector<llama_chat_message> messages={{"system",system.c_str()}};
    std::string prefix, full;
    int system_tokens=0, action_start=0;
    if(fmt=="chat") {
        system_tokens=tokens(vocab,format(home.model,messages,false),true).size();
        if(has_history) {
            messages.push_back({"user",history.c_str()});
            action_start=tokens(vocab,format(home.model,messages,false),true).size();
            messages.push_back({"assistant",previous.c_str()});
            prefix=format(home.model,messages,false);
        }
        messages.push_back({"user",prompt.c_str()}); full=format(home.model,messages,true);
    } else { prefix=history; full=prefix+prompt; }
    require(full.compare(0,prefix.size(),prefix)==0, "chat prefix is not stable");
    std::ofstream(get("--out")+".prompt.txt") << full;
    llama_pos pos=0;
    t=now_ms(); auto hist_tokens=tokens(vocab,prefix,true);
    if(has_history) evaluate(home.ctx,hist_tokens,pos);
    times["history_prefill_ms"]=now_ms()-t;
    sample("history_ready");
    int original_history=has_history?hist_tokens.size():0, removed=0;
    if(get("--roundtrip", "0")=="1" && has_history) {
        t=now_ms(); transfer(home.ctx,home.ctx); times["same_backend_roundtrip_checked_ms"]=now_ms()-t;
    }
    t=now_ms();
    if(policy=="recent" && has_history) {
        require(fmt=="chat", "recent policy requires chat format");
        int keep=std::stoi(get("--keep-history", "96"));
        require(keep>=0, "keep-history must be nonnegative");
        // Preserve system and entire previous action, including its message framing.
        int end=std::max(system_tokens,std::min(action_start,original_history-keep));
        if(end>system_tokens) {
            require(llama_memory_seq_rm(llama_get_memory(home.ctx),0,system_tokens,end),"KV removal failed");
            removed=end-system_tokens;
        }
    }
    times["kv_select_ms"]=now_ms()-t;
    size_t upload_bytes=0, download_bytes=0;
    auto upload=[&]() { double u=now_ms(); upload_bytes=transfer(home.ctx,pf.ctx); times["upload_checked_ms"]=now_ms()-u; };
    double gap_start=now_ms();
    if(pre && has_history) upload();
    if(vision_prepare=="a2o") init_vision();
    double budget=std::stod(get("--a2o-ms", "0")); require(budget>=0,"a2o-ms must be nonnegative");
    double prep_elapsed=times["kv_select_ms"]+(now_ms()-gap_start);
    double remain=budget-prep_elapsed;
    if(remain>0) std::this_thread::sleep_for(std::chrono::duration<double,std::milli>(remain));
    times["a2o_wall_ms"]=now_ms()-gap_start+times["kv_select_ms"];
    times["prep_work_ms"]=times["kv_select_ms"]+(pre?times["upload_checked_ms"]:0)
        +(vision_prepare=="a2o"?times["vision_init_ms"]:0);
    times["prep_overrun_ms"]=std::max(0.0,prep_elapsed-budget);
    double arrival=now_ms();
    sample("observation_arrival");
    if(hybrid && !pre && has_history) upload();
    if(vision_prepare=="arrival") init_vision();
    std::string suffix=full.substr(prefix.size());
    t=now_ms(); int new_tokens=0;
    if(mm) {
        auto bitmap=mtmd_helper_bitmap_init_from_file(mm,image.c_str()); require(bitmap,"image load failed");
        const mtmd_bitmap * bs[]={bitmap};
        auto chunks=mtmd_input_chunks_init(); mtmd_input_text txt={suffix.c_str(),!has_history,true};
        require(mtmd_tokenize(mm,chunks,&txt,bs,1)==0,"multimodal tokenize failed");
        new_tokens=mtmd_helper_get_n_tokens(chunks);
        times["image_preprocess_ms"]=now_ms()-t;
        for(size_t i=0;i<mtmd_input_chunks_size(chunks);++i) {
            const auto chunk=mtmd_input_chunks_get(chunks,i);
            double start=now_ms();
            if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                std::vector<float> external;
                float * embeddings=nullptr;
                if(!vision_embeddings.empty()) {
                    size_t count=mtmd_input_chunk_get_n_tokens(chunk)*llama_model_n_embd_inp(pf.model);
                    std::ifstream input(vision_embeddings,std::ios::binary|std::ios::ate);
                    require(bool(input) && input.tellg()==std::streamoff(count*sizeof(float)),"vision embedding shape mismatch");
                    external.resize(count); input.seekg(0);
                    input.read(reinterpret_cast<char*>(external.data()),count*sizeof(float));
                    embeddings=external.data();
                    times["vision_embedding_load_ms"]+=now_ms()-start;
                } else {
                    require(mtmd_encode_chunk(mm,chunk)==0,"vision encode failed");
                    embeddings=mtmd_get_output_embd(mm);
                    times["vision_encode_ms"]+=now_ms()-start;
                }
                start=now_ms();
                require(mtmd_helper_decode_image_chunk(mm,pf.ctx,chunk,embeddings,pos,0,batch_size,&pos)==0,"image prefill failed");
            } else {
                require(mtmd_helper_eval_chunk_single(mm,pf.ctx,chunk,pos,0,batch_size,i+1==mtmd_input_chunks_size(chunks),&pos)==0,"text prefill failed");
            }
            llama_synchronize(pf.ctx);
            times["llm_prefill_ms"]+=now_ms()-start;
        }
        llama_synchronize(pf.ctx); mtmd_input_chunks_free(chunks); mtmd_bitmap_free(bitmap);
    } else { auto ts=tokens(vocab,suffix,!has_history); new_tokens=ts.size(); evaluate(pf.ctx,ts,pos); }
    times["observation_eval_ms"]=now_ms()-t;
    sample("observation_evaluated");
    llama_token first=greedy(pf.ctx,nv);
    // This executable handles one observation. Once prefill is synchronized,
    // encoder weights and scratch buffers are no longer used by decode.
    if(mm) {
        t=now_ms(); mtmd_free(mm); mm=nullptr;
        times["vision_release_ms"]=now_ms()-t; sample("vision_released");
    }
    if(hybrid) {
        t=now_ms(); download_bytes=transfer(pf.ctx,home.ctx); times["download_checked_ms"]=now_ms()-t;
    }
    std::string output; std::vector<llama_token> generated;
    sample("decode_start");
    bool eog=false; t=now_ms();
    for(int i=0;i<max_tokens;++i) {
        auto tok=i==0?first:greedy(home.ctx,nv);
        if(llama_vocab_is_eog(vocab,tok)) { eog=true; break; }
        generated.push_back(tok); output+=piece(vocab,tok);
        evaluate(home.ctx,{tok},pos);
    }
    times["decode_ms"]=now_ms()-t;
    times["observation_to_response_ms"]=now_ms()-arrival;
    sample("decode_end");
    std::ofstream out(get("--out")); require(bool(out),"cannot write result");
    out << "{\"mode\":" << json(mode) << ",\"kv_policy\":" << json(policy)
        << ",\"batch_size\":" << batch_size << ",\"cpu_repack\":" << (cpu_repack?"true":"false")
        << ",\"vision_flash\":" << json(vision_flash)
        << ",\"vision_provider\":" << json(vision_embeddings.empty()?"mtmd":"embedding_replay")
        << ",\"vision_prepare\":" << json(vision_prepare) << ",\"trace_vision\":" << (trace_vision?"true":"false")
        << ",\"pre_move\":" << (pre?"true":"false") << ",\"format\":" << json(fmt)
        << ",\"vision_backend\":" << json(image.empty()?"none":(vision=="htp"?"HTP0":(vision=="opencl"?"OpenCL":"CPU")))
        << ",\"history_tokens\":" << original_history << ",\"removed_tokens\":" << removed
        << ",\"observation_tokens\":" << new_tokens << ",\"next_position\":" << pos
        << ",\"upload_bytes\":" << upload_bytes << ",\"download_bytes\":" << download_bytes
        << ",\"transfer_checked\":" << (hybrid?"true":"false") << ",\"first_token\":" << first
        << ",\"first_piece\":" << json(piece(vocab,first)) << ",\"eog\":" << (eog?"true":"false")
        << ",\"output\":" << json(output) << ",\"generated_ids\":[";
    for(size_t i=0;i<generated.size();++i) out << (i?",":"") << generated[i];
    out << "],\"timing_ms\":{"; bool comma=false;
    for(const auto & kv:times) { out << (comma?",":"") << json(kv.first) << ':' << kv.second; comma=true; }
    out << "},\"resources\":{"; comma=false;
    for(const auto & stage:resources) {
        out << (comma?",":"") << json(stage.first) << ":{"; comma=true; bool inner=false;
        for(const auto & metric:stage.second) {
            out << (inner?",":"") << json(metric.first) << ':' << metric.second; inner=true;
        }
        out << '}';
    }
    out << "}}\n";
    std::cout << output << '\n'; return 0;
} catch(const std::exception & e) { std::cerr << "agent_runtime: " << e.what() << '\n'; return 1; } }

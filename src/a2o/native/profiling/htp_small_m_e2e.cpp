// 固定完整H快照，比较两个DSP版本的P31+后续decode；不构造dummy token。
#define main a2o_legacy_main
#define json a2o_json_string
#include "../agent_runtime.cpp"
#undef json
#undef main
#include <filesystem>
#include "nlohmann/json.hpp"
using J=nlohmann::ordered_json;

static std::string checksum(const std::vector<uint8_t> &bytes) {
    uint64_t hash=14695981039346656037ULL;
    for(auto byte:bytes){hash^=byte;hash*=1099511628211ULL;}
    std::ostringstream out;out<<std::hex<<hash;return out.str();
}
static std::vector<float> logits(llama_context *ctx,int nv) {
    auto *p=llama_get_logits_ith(ctx,-1);require(p,"missing logits");
    std::vector<float> result(p,p+nv);
    for(auto v:result)require(std::isfinite(v),"nonfinite logits");
    return result;
}
static void dump(const char *base,const std::string &suffix,const std::vector<float> &values) {
    std::ofstream out(std::string(base)+suffix,std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    require(bool(out),"logits dump failed");
}
int main(int argc,char **argv) {try {
    require(argc==5,"usage: probe MODEL FIXTURE SEED_JSON OUTPUT_JSON");
    context_size=8192;batch_size=128;
    J fixture=J::parse(read(argv[2])),metadata=J::parse(read(argv[3]));
    auto prefix=fixture.at("prefix_ids").get<std::vector<llama_token>>();
    auto input=fixture.at("input_ids").get<std::vector<llama_token>>();
    auto forced=fixture.at("generated_ids").get<std::vector<llama_token>>();
    require(prefix.size()==4096 && input.size()==31 && forced.size()==16,"unexpected H/P/D");
    auto &m=metadata.at("metadata");
    require(m.at("format")==2 && m.at("context")==context_size && m.at("batch")==batch_size && m.at("prefix_ids")==prefix,"seed config mismatch");
    std::string identity=std::string(argv[1])+":"+std::to_string(std::filesystem::file_size(argv[1]))+":"+
        std::to_string(static_cast<long long>(std::filesystem::last_write_time(argv[1]).time_since_epoch().count()))+":";
    auto stored=m.at("identity").get<std::string>();
    require(stored.rfind(identity,0)==0 && stored.substr(stored.size()-14)==":cpu_threads=6","seed model/thread identity mismatch");
    SessionSnapshot frozen;{auto bytes=read(std::string(argv[3])+".state");frozen.state.assign(bytes.begin(),bytes.end());}
    frozen.next_pos=prefix.size();frozen.evaluated_tokens=prefix.size();
    require(frozen.state.size()==metadata.at("state_bytes").get<size_t>() && checksum(frozen.state)==metadata.at("state_checksum").get<std::string>(),"seed state checksum mismatch");
    ggml_backend_load_all();
    double start=now_ms();Engine engine(argv[1],true,false,false,4);double init=now_ms()-start;
    auto *ctx=engine.ctx;auto *vocab=llama_model_get_vocab(engine.model);int nv=llama_vocab_n_tokens(vocab);
    J result={{"H",prefix.size()},{"P",input.size()},{"D",forced.size()},{"backend","HTP0"},{"htp_host_threads",4},
        {"init_ms",init},{"seed_checksum",checksum(frozen.state)},{"seed_bytes",frozen.state.size()},
        {"scope","fixed-history full-model P31 and HTP decode; init/validation excluded; no task-quality claim"}};
    auto save=[&](){std::ofstream out(argv[4]);out<<result.dump(2);require(bool(out),"result write failed");};
    restore_session(ctx,frozen);llama_pos pos=prefix.size();evaluate(ctx,input,pos);
    auto prefill_logits=logits(ctx,nv);dump(argv[4],".prefill.f32",prefill_logits);
    SessionSnapshot after_prefill;after_prefill.state=export_seq(ctx);after_prefill.next_pos=pos;
    std::string prefill_hash=checksum(after_prefill.state);
    for(auto token:forced)evaluate(ctx,{token},pos);
    auto forced_logits=logits(ctx,nv);dump(argv[4],".forced.f32",forced_logits);
    auto final_state=export_seq(ctx);std::string final_hash=checksum(final_state);size_t final_bytes=final_state.size();
    final_state.clear();final_state.shrink_to_fit();
    restore_session(ctx,after_prefill);pos=after_prefill.next_pos;
    for(auto token:forced)evaluate(ctx,{token},pos);
    auto replay_logits=logits(ctx,nv);
    require(replay_logits==forced_logits,"continuation changed after state roundtrip");
    require(checksum(export_seq(ctx))==final_hash,"final state changed after roundtrip continuation");
    restore_session(ctx,after_prefill);pos=after_prefill.next_pos;
    // 序列快照不包含采样logits，恢复prefill logits后再做自由greedy延续。
    auto *last=llama_get_logits_ith(ctx,-1);require(last,"missing restored logits buffer");
    std::copy(prefill_logits.begin(),prefill_logits.end(),last);
    std::vector<llama_token> generated;std::string text;bool eog=false;
    for(size_t i=0;i<forced.size();++i){auto token=greedy(ctx,nv);generated.push_back(token);
        if(llama_vocab_is_eog(vocab,token)){eog=true;break;}text+=piece(vocab,token);evaluate(ctx,{token},pos);}
    result["validation"]={{"finite_logits",true},{"state_roundtrip_verified",true},{"continuation_bitwise_verified",true},
        {"prefill_state_checksum",prefill_hash},{"prefill_state_bytes",after_prefill.state.size()},
        {"forced_state_checksum",final_hash},{"forced_state_bytes",final_bytes},
        {"greedy_ids",generated},{"greedy_text",text},{"greedy_eog",eog},{"prefill_top1",std::max_element(prefill_logits.begin(),prefill_logits.end())-prefill_logits.begin()}};
    after_prefill.state.clear();after_prefill.state.shrink_to_fit();save();
    auto trial=[&](){double work=0,start=now_ms();restore_session(ctx,frozen,&work);double restore=now_ms()-start;
        llama_pos p=prefix.size();start=now_ms();evaluate(ctx,input,p);double prefill=now_ms()-start;
        start=now_ms();for(auto token:forced)evaluate(ctx,{token},p);double decode=now_ms()-start;
        // 校验在计时外，每次均比较完整最终logits，不仅检查有限值。
        require(logits(ctx,nv)==forced_logits,"timing trial continuation not deterministic");
        return J{{"restore_work_ms",work},{"restore_with_validation_ms",restore},{"restore_validation_ms",restore-work},
            {"prefill_ms",prefill},{"decode_ms",decode},{"prefill_decode_ms",prefill+decode},
            {"total_ms",work+prefill+decode},{"final_logits_bitwise_verified",true}};};
    result["warmup"]=trial();result["trials"]=J::array();save();
    for(int i=0;i<10;++i){auto row=trial();row["repetition"]=i;result["trials"].push_back(row);save();}
    result["complete"]=true;save();return 0;
}catch(const std::exception &error){std::cerr<<"E2E_FAILED "<<error.what()<<'\n';return 1;}}

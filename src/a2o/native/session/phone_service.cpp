// 私有研究服务：复用原工程的推理/状态机制和 llama.cpp 官方模板、解析及采样器。
#define main a2o_legacy_main
#define json a2o_json_string
#include "../agent_runtime.cpp"
#undef json
#undef main
#include "chat.h"
#include "sampling.h"
#include "httplib.h"
#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#ifndef _WIN32
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>
#include <sys/socket.h>
#endif
#include "nlohmann/json.hpp"
using J=nlohmann::ordered_json;
#include "replay_cache.h"
#include "../scheduling/cold_route.h"
#include "../scheduling/greedy_return.h"
#include "../scheduling/kv_reduce_cost.h"
#include "../scheduling/m1_explorer.h"
#include "../profiling/prefill_prior.h"
#include "../profiling/phone_eval_trace.h"
#include "kv_window.h"
#include "../profiling/kv_reduce_probe.h"
#include "../profiling/memory_ledger.h"

#ifndef A2O_GIT_COMMIT
#define A2O_GIT_COMMIT "unknown"
#endif
#ifndef A2O_BUILD_ID
#define A2O_BUILD_ID __DATE__ " " __TIME__
#endif

namespace {
struct BigCoreMask {
    bool valid = false;
    cpu_set_t set{};
};

static const BigCoreMask & get_big_core_mask() {
    static const BigCoreMask mask = []() -> BigCoreMask {
        BigCoreMask out;
        CPU_ZERO(&out.set);
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path base("/sys/devices/system/cpu/cpufreq");
        if (!fs::is_directory(base, ec)) return out;
        struct PolicyInfo { long long max_khz; std::vector<int> cpus; };
        std::vector<PolicyInfo> policies;
        long long min_khz = std::numeric_limits<long long>::max();
        for (const auto & entry : fs::directory_iterator(base, ec)) {
            if (!entry.is_directory(ec)) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("policy", 0) != 0) continue;
            std::ifstream f_max(entry.path() / "cpuinfo_max_freq");
            std::ifstream f_cpus(entry.path() / "affected_cpus");
            long long khz = 0;
            if (!(f_max >> khz) || khz <= 0 || !f_cpus) continue;
            PolicyInfo pi{khz, {}};
            int cpu = -1;
            while (f_cpus >> cpu) {
                if (cpu >= 0 && cpu < CPU_SETSIZE) pi.cpus.push_back(cpu);
            }
            if (!pi.cpus.empty()) {
                min_khz = std::min(min_khz, khz);
                policies.push_back(std::move(pi));
            }
        }
        int count = 0;
        for (const auto & pi : policies) {
            if (pi.max_khz > min_khz) {
                for (int cpu : pi.cpus) {
                    CPU_SET(cpu, &out.set);
                    ++count;
                }
            }
        }
        out.valid = count > 0;
        return out;
    }();
    return mask;
}

class ScopedBigCoreAffinity {
public:
    explicit ScopedBigCoreAffinity(bool enable) {
#ifndef _WIN32
        if (!enable) return;
        const auto & target = get_big_core_mask();
        if (!target.valid) return;
        CPU_ZERO(&prev_);
        if (sched_getaffinity(0, sizeof(prev_), &prev_) == 0 &&
            sched_setaffinity(0, sizeof(target.set), &target.set) == 0) {
            active_ = true;
        }
#else
        (void)enable;
#endif
    }
    ~ScopedBigCoreAffinity() {
#ifndef _WIN32
        if (active_) {
            sched_setaffinity(0, sizeof(prev_), &prev_);
        }
#endif
    }
    ScopedBigCoreAffinity(const ScopedBigCoreAffinity &) = delete;
    ScopedBigCoreAffinity & operator=(const ScopedBigCoreAffinity &) = delete;
private:
    bool active_ = false;
#ifndef _WIN32
    cpu_set_t prev_{};
#endif
};
} // namespace

struct PhoneService {
    std::string condition;
    std::unique_ptr<Engine> cpu,htp;
    common_chat_templates_ptr templates;
    const llama_vocab *vocab=nullptr;
    std::shared_ptr<const SessionSnapshot> anchor;
    std::vector<llama_token> anchor_ids;
    std::vector<int> anchor_live_positions;
    size_t anchor_pinned_tokens=0;
    PreparationLane lane;
    std::mutex mutex;
    double init_ms=0,prepare_ms=0;
    std::vector<J> costs;
    J cost_model=nullptr,kv_cost_model=nullptr;
    KvReduceCostTable kv_cost_table;
    ColdRouteCosts cold_costs;
    uint64_t version=0;
    int cpu_threads=4;
    bool htp_kv_offload=false, flash_attention=false, cpu_repack=false, kv_reduce=false, compact_anchor=false;
    bool single_resident=false, shared_weights=false, default_enable_thinking=false, htp_big_cores=false;
    int kv_keep_tokens=512, kv_sink_tokens=4, kv_recent_tokens=64, kv_pool_window=5, kv_tail_tokens=16, kv_layer_stride=4, kv_min_evict_tokens=256, expected_decode_tokens=96;
    double kv_keep_ratio=-1.0;
    bool kv_hwm_trim=true;
    int cpu_kv_hwm=0;
    bool memory_ledger_enabled=false;
    KvVerification kv_verification=KvVerification::import_size;
    J startup_memory=nullptr;
    bool m1_explorer_enabled=false;
    a2o_m1::M1JointExplorer m1_explorer;
    std::string device_name="xiaomi14pro";
    int startup_gap_ms=-1;
    std::string kv_policy="prefill_prior";
    PrefillPrior prior;
    std::string model_path;
    std::chrono::steady_clock::time_point prepare_started_at{};
    std::string model_id="qwen35-9b-q4_0";
    std::unique_ptr<FrozenReplayCache> replay_cache;
    PhoneService(const std::string &model,const std::string &which, bool memory_ledger):
        condition(which), memory_ledger_enabled(memory_ledger) {
        model_path=model;
        if(const char *value=std::getenv("A2O_CPU_THREADS"))cpu_threads=std::stoi(value);
        require(cpu_threads>0 && cpu_threads<=16,"invalid CPU thread count");
        if(const char *value=std::getenv("A2O_CPU_REPACK")){require(std::string(value)=="0"||std::string(value)=="1","invalid CPU repack flag");cpu_repack=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_HTP_KV_OFFLOAD")){require(std::string(value)=="0"||std::string(value)=="1","invalid KV offload flag");htp_kv_offload=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_HTP_BIG_CORES")){require(std::string(value)=="0"||std::string(value)=="1","invalid HTP big cores flag");htp_big_cores=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_FLASH_ATTN")){require(std::string(value)=="0"||std::string(value)=="1","invalid flash attention flag");flash_attention=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_KV_REDUCE")){require(std::string(value)=="0"||std::string(value)=="1","invalid KV reduce flag");kv_reduce=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_KV_KEEP_TOKENS"))kv_keep_tokens=std::stoi(value);
        if(const char *value=std::getenv("A2O_KV_KEEP_RATIO"))kv_keep_ratio=std::stod(value);
        if(const char *value=std::getenv("A2O_KV_SINK_TOKENS"))kv_sink_tokens=std::stoi(value);
        if(const char *value=std::getenv("A2O_KV_RECENT_TOKENS"))kv_recent_tokens=std::stoi(value);
        if(const char *value=std::getenv("A2O_KV_POOL_WINDOW"))kv_pool_window=std::stoi(value);
        if(const char *value=std::getenv("A2O_KV_TAIL_TOKENS"))kv_tail_tokens=std::stoi(value);
        if(const char *value=std::getenv("A2O_KV_LAYER_STRIDE"))kv_layer_stride=std::stoi(value);
        if(const char *value=std::getenv("A2O_KV_MIN_EVICT_TOKENS"))kv_min_evict_tokens=std::stoi(value);
        require(kv_layer_stride>=1,"invalid A2O_KV_LAYER_STRIDE");
        require(kv_min_evict_tokens>=0,"invalid A2O_KV_MIN_EVICT_TOKENS");
        if(const char *value=std::getenv("A2O_KV_POLICY"))kv_policy=value;
        if(const char *value=std::getenv("A2O_COMPACT_ANCHOR")){require(std::string(value)=="0"||std::string(value)=="1","invalid compact anchor flag");compact_anchor=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_KV_HWM_TRIM")){require(std::string(value)=="0"||std::string(value)=="1","invalid kv_hwm_trim flag");kv_hwm_trim=std::string(value)=="1";}
        require(!memory_ledger_enabled || std::getenv("A2O_KV_HWM_TRIM"),
                "memory ledger requires explicit A2O_KV_HWM_TRIM=0 or 1");
        if(const char *value=std::getenv("A2O_M1_EXPLORER")){require(std::string(value)=="0"||std::string(value)=="1","invalid m1_explorer flag");m1_explorer_enabled=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_VERIFY_KV_STATE")){
            require(std::string(value)=="0"||std::string(value)=="1","invalid verify KV state flag");
            if(std::string(value)=="1")kv_verification=KvVerification::full_bytes;
        }
        if(const char *value=std::getenv("A2O_ENABLE_THINKING")){require(std::string(value)=="0"||std::string(value)=="1","invalid enable_thinking flag");default_enable_thinking=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_DEVICE"))device_name=value;
        if(const char *value=std::getenv("A2O_GAP_MS"))startup_gap_ms=std::stoi(value);
        if(const char *value=std::getenv("A2O_EXPECTED_DECODE"))expected_decode_tokens=std::stoi(value);
        require(!kv_reduce || which=="cpu" || which=="route","kv_reduce requires cpu or route condition");
        require(!kv_reduce || !flash_attention,"kv_reduce requires flash_attention=0");
        require(kv_policy=="prefill_prior" || kv_policy=="recent" || kv_policy=="adaptive_cost","unsupported A2O_KV_POLICY");
        bool need_prior_cb=kv_reduce && (kv_policy=="prefill_prior" || kv_policy=="adaptive_cost");
        prior.collect_ffn=false;prior.collect_kv=need_prior_cb;prior.kv_tail_tokens=kv_tail_tokens;prior.kv_layer_stride=kv_layer_stride;
        if(const char *value=std::getenv("A2O_MODEL_ID"))model_id=value;
        if(const char *value=std::getenv("A2O_SINGLE_RESIDENT")){require(std::string(value)=="0"||std::string(value)=="1","invalid residency flag");single_resident=std::string(value)=="1";}
        if (const char *value = std::getenv("A2O_SHARED_WEIGHTS")) {
            require(std::string(value) == "0" || std::string(value) == "1", "invalid shared weights flag");
            shared_weights = std::string(value) == "1";
        }
        require(!shared_weights || (which == "route" && !single_resident && !cpu_repack),
                "shared weights require route with stock CPU packing and both backend contexts");
        replay_cache=std::make_unique<FrozenReplayCache>(model+":"+std::to_string(std::filesystem::file_size(model))+":"+std::to_string(static_cast<long long>(std::filesystem::last_write_time(model).time_since_epoch().count()))+":"+__DATE__+":"+__TIME__+":cpu_threads="+std::to_string(cpu_threads)+(shared_weights?":shared_weights=1":""));
        a2o::memory::Ledger memory(memory_ledger_enabled, kv_hwm_trim);
        memory.sample("before_model_load");
        const double before_load_collection=memory.overhead_ms();
        double start=now_ms();
        if(which!="htp" && !shared_weights)cpu=std::make_unique<Engine>(model,false,cpu_repack,true,cpu_threads,flash_attention);
        if(cpu) memory.sample("after_cpu_model_context");
        if(which!="cpu" && !single_resident)htp=std::make_unique<Engine>(model,true,false,htp_kv_offload,4,flash_attention);
        if(htp) memory.sample("after_htp_model_context");
        if (shared_weights) {
            cpu = std::make_unique<Engine>(*htp, cpu_threads, flash_attention,nullptr,nullptr);
        }
        require(cpu||htp,"single residency requires route or cpu startup");
        auto *m=cpu?cpu->model:htp->model;vocab=llama_model_get_vocab(m);
        templates=common_chat_templates_init(m,"");init_ms=now_ms()-start-(memory.overhead_ms()-before_load_collection);
        memory.sample("after_model_load");
        if(memory_ledger_enabled) startup_memory=memory.result();
    }
    ~PhoneService() { try { lane.close(); } catch (...) {} }

    Engine &home(){return cpu?*cpu:*htp;}
    J select_stage_backend(const std::string &backend,const std::vector<llama_token> &warm_ids) {
        require(backend=="cpu"||backend=="htp","invalid stage backend");
        if(!single_resident || (backend=="cpu"?bool(cpu):bool(htp)))return {{"switched",false},{"model_context_init_ms",0},{"warmup_ms",0}};
        lane.close();anchor={};anchor_ids.clear();templates.reset();vocab=nullptr;
        double begin=now_ms();cpu.reset();htp.reset();double release=now_ms()-begin;
        auto released=resource_sample();begin=now_ms();
        if(backend=="cpu")cpu=std::make_unique<Engine>(model_path,false,false,true,cpu_threads,flash_attention);
        else htp=std::make_unique<Engine>(model_path,true,false,htp_kv_offload,4,flash_attention);
        double initialize=now_ms()-begin;vocab=llama_model_get_vocab(home().model);
        templates=common_chat_templates_init(home().model,"");
        begin=now_ms();llama_pos pos=0;
        evaluate(home().ctx,std::vector<llama_token>(warm_ids.begin(),warm_ids.begin()+std::min(size_t(32),warm_ids.size())),pos);
        double warmup=now_ms()-begin;
        return {{"switched",true},{"release_ms",release},{"resources_after_release",released},
            {"model_context_init_ms",initialize},{"warmup_ms",warmup},{"warmup_tokens",pos}};
    }
    J enable_dual_state() {
        lane.close();anchor={};anchor_ids.clear();double begin=now_ms();
        if(!cpu)cpu=std::make_unique<Engine>(model_path,false,false,true,cpu_threads,flash_attention);
        if(!htp)htp=std::make_unique<Engine>(model_path,true,false,htp_kv_offload,4,flash_attention);
        single_resident=false;vocab=llama_model_get_vocab(cpu->model);templates=common_chat_templates_init(cpu->model,"");
        return {{"mode","dual_state_cost_only"},{"setup_ms",now_ms()-begin},{"resources",resource_sample()}};
    }
    common_chat_params render(const J &body) {
        common_chat_templates_inputs in;
        in.messages=common_chat_msgs_parse_oaicompat(body.at("messages"));
        in.tools=common_chat_tools_parse_oaicompat(body.value("tools",J::array()));
        bool think=default_enable_thinking;
        if(body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()){
            const auto &kw=body["chat_template_kwargs"];
            if(kw.contains("enable_thinking")){
                if(kw["enable_thinking"].is_boolean())think=kw["enable_thinking"].get<bool>();
                else if(kw["enable_thinking"].is_string()){
                    const auto s=kw["enable_thinking"].get<std::string>();
                    think=(s=="true" || s=="1");
                }
            }
        }
        in.enable_thinking=think;in.chat_template_kwargs["enable_thinking"]=think?"true":"false";
        in.parallel_tool_calls=false;in.use_jinja=true;in.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
        return common_chat_templates_apply(templates.get(),in);
    }
    J start_prepare(const J &body){
        lane.poll();
        if(condition!="route" || !anchor)return {{"started",false},{"reason","disabled_or_no_anchor"}};
        if(lane.occupied())return {{"started",false},{"reason","previous_preparation_busy"}};
        const bool serial=body.value("frozen_serial_context",false);
        double cpu_release_ms=0;
        if(serial){
            require(cpu && cpu->ctx && htp && !htp->ctx && !kv_reduce,"serial context probe needs only the CPU context and full KV");
            const double begin=now_ms();
            cpu->release_ctx();
            cpu_kv_hwm=0;
            cpu_release_ms=now_ms()-begin;
        }
        a2o::memory::Ledger memory(memory_ledger_enabled, kv_hwm_trim);
        memory.sample("before_tool_prepare");
        prepare_started_at=std::chrono::steady_clock::now();
        bool started=lane.start(anchor->version,[this,state=anchor](const auto &valid){
            double start=now_ms();if(!valid())return false;
            restore_session(htp->ensure_ctx(),*state,nullptr,kv_verification);prepare_ms=now_ms()-start;
            a2o::memory::Ledger prepared(memory_ledger_enabled, kv_hwm_trim);
            prepared.sample("after_tool_prepare");
            return valid();
        },[this]{htp->release_ctx();});
        J response={{"started",started},{"version",anchor->version},{"H",anchor_ids.size()},
                    {"serial_context",serial},{"cpu_context_release_ms",cpu_release_ms}};
        if(memory_ledger_enabled) response["memory_ledger"]=memory.result();
        return response;
    }
    J calibrate(const J &body){
        require(condition=="route" && cpu && htp,"calibration requires both explicit backends");
        lane.close();
        auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
        auto input=body.at("input_ids").get<std::vector<llama_token>>();
        const bool paired=body.contains("truth_decode_ids");
        const auto truth_ids=body.value("truth_decode_ids",std::vector<llama_token>{});
        const int gap_ms=body.value("truth_gap_ms",0);
        require(!paired || (!truth_ids.empty() && !prefix.empty() && gap_ms>=0 && gap_ms<=10000),
                "paired truth requires a warm prefix, decode IDs and gap in 0..10000 ms");
        for(auto tok:truth_ids)require(tok>=0 && tok<llama_vocab_n_tokens(vocab),"invalid paired decode token");
        int d=paired?int(truth_ids.size()):body.value("decode_evaluations",4);
        const bool cold=prefix.empty();
        const bool include_hc_compact=body.value("include_hc_compact",false);
        const int cal_keep_tokens=body.value("keep_tokens",kv_keep_tokens);
        if(include_hc_compact){
            require(kv_reduce && (kv_policy=="prefill_prior" || kv_policy=="adaptive_cost"),
                    "calibrate include_hc_compact requires kv_reduce=1 with prefill_prior");
            require(!cold && cal_keep_tokens>0,"calibrate include_hc_compact requires warm prefix and keep_tokens>0");
        }
        require(!input.empty() && prefix.size()+input.size()+d<size_t(context_size),"invalid calibration shape");
        llama_pos pos=0;
        SessionSnapshot state;
        if(!cold){
            llama_memory_clear(llama_get_memory(cpu->ctx),true);
            evaluate(cpu->ctx,prefix,pos);
            state.state=export_seq(cpu->ctx);
            state.next_pos=pos;
        }
        J rows=J::array();
        using Clock=std::chrono::steady_clock;
        // Tool return is a device-clock deadline. HTP actually restores during the gap.
        auto return_after_prepare=[](Clock::time_point tool_return){
            const auto prepared=Clock::now();
            const double wait=std::max(0.0,std::chrono::duration<double,std::milli>(prepared-tool_return).count());
            std::this_thread::sleep_until(tool_return);
            return wait;
        };
        for(int repetition=0;repetition<body.value("repetitions",3);++repetition){
            J row={{"repetition",repetition}};
            if(paired) row["paired_truth"]={{"version",2},{"D",d},{"gap_ms",gap_ms}};
            std::vector<float> cpu_logits,htp_logits;
            // 每次标定前释放 HTP context，使 hc_prepare_ms 真实计入按需重建 htp->ctx 的开销。
            htp->release_ctx();
            for(bool accelerated:{false,true}){
                const auto tool_return=Clock::now()+std::chrono::milliseconds(gap_ms);
                if(paired && !accelerated)std::this_thread::sleep_until(tool_return);
                const double setup_start=now_ms();
                auto *ctx=accelerated?htp->ensure_ctx():cpu->ctx;
                const double setup_ms=now_ms()-setup_start;
                double ps=now_ms();restore_session(ctx,state,nullptr,kv_verification);double preparation=now_ms()-ps;
                const double wait=paired && accelerated?return_after_prepare(tool_return):0.0;
                pos=prefix.size();double begin=now_ms();evaluate(ctx,input,pos);
                row[accelerated?"htp_prefill_ms":"cpu_prefill_ms"]=now_ms()-begin;
                const double capture_start=now_ms();
                auto logits=llama_get_logits_ith(ctx,-1);
                require(logits,"missing calibration logits");
                auto &values=accelerated?htp_logits:cpu_logits;
                values.assign(logits,logits+llama_vocab_n_tokens(vocab));
                const double capture_ms=paired?now_ms()-capture_start:0.0;
                const auto first=paired?truth_ids.front():greedy(ctx,llama_vocab_n_tokens(vocab));
                double handoff_start=now_ms();
                size_t handoff_bytes=0;
                if(accelerated){
                    handoff_bytes=transfer(ctx,cpu->ctx,kv_verification);
                    if(!paired){
                        auto *destination=llama_get_logits_ith(cpu->ctx,-1);
                        require(destination,"missing migration logits");
                        std::copy(values.begin(),values.end(),destination);
                    }
                }
                if(paired && accelerated)htp->release_ctx();
                const double handoff=accelerated?now_ms()-handoff_start:0.0;
                row[accelerated?"htp_to_cpu_ms":"cpu_handoff_ms"]=now_ms()-handoff_start;
                if(cold){
                    const double cost=setup_ms+preparation+now_ms()-begin;
                    row[accelerated?"cold_hc_ms":"cold_cc_ms"]=cost;
                }
                double decode_start=now_ms();
                auto *dec=cpu->ctx;std::vector<llama_token> ids;
                for(int i=0;i<d;++i){auto tok=paired?truth_ids[i]:i==0?first:greedy(dec,llama_vocab_n_tokens(vocab));ids.push_back(tok);evaluate(dec,{tok},pos);}
                row[accelerated?"hc_decode_ms":"cc_decode_ms"]=now_ms()-decode_start;
                row[accelerated?"hc_ms":"cc_ms"]=now_ms()-begin;
                row[accelerated?"hc_prepare_ms":"cc_restore_ms"]=accelerated?(setup_ms+preparation):preparation;
                row[accelerated?"hc_ids":"cc_ids"]=ids;
                if(paired){
                    const double restore=accelerated?0.0:preparation;
                    const double elapsed=row.at(accelerated?"hc_ms":"cc_ms").get<double>();
                    row["paired_truth"][accelerated?"HC-Full":"CC"]={
                        {"prepare_ms",accelerated?setup_ms+preparation:0.0},{"wait_ms",wait},{"restore_ms",restore},
                        {"prefill_ms",row.at(accelerated?"htp_prefill_ms":"cpu_prefill_ms")},
                        {"compact_ms",0.0},{"handoff_ms",handoff},{"decode_ms",row.at(accelerated?"hc_decode_ms":"cc_decode_ms")},
                        {"post_return_ms",wait+restore+elapsed-capture_ms},{"decode_ids",ids},
                        {"validation_capture_ms",capture_ms},{"handoff_bytes",handoff_bytes}};
                }
            }
            double dot=0,aa=0,bb=0,err=0;
            for(size_t i=0;i<cpu_logits.size();++i){require(std::isfinite(cpu_logits[i])&&std::isfinite(htp_logits[i]),"nonfinite calibration logits");dot+=cpu_logits[i]*htp_logits[i];aa+=cpu_logits[i]*cpu_logits[i];bb+=htp_logits[i]*htp_logits[i];err+=std::pow(cpu_logits[i]-htp_logits[i],2);}
            row["cosine"]=dot/std::sqrt(aa*bb);row["nrmse"]=std::sqrt(err/aa);
            row["passed"]=row["cosine"].get<double>()>=0.99 && row["cc_ids"]==row["hc_ids"];
            if(include_hc_compact){
                htp->release_ctx();
                const auto tool_return=Clock::now()+std::chrono::milliseconds(gap_ms);
                const double setup_start=now_ms();
                auto *ctx=htp->ensure_ctx();
                const double setup_ms=now_ms()-setup_start;
                double ps=now_ms();restore_session(ctx,state,nullptr,kv_verification);double preparation=now_ms()-ps;
                const double wait=paired?return_after_prepare(tool_return):0.0;
                pos=prefix.size();
                double pf0=now_ms();
                const bool collect_prior=!paired ||
                    (int(prefix.size()+input.size())-cal_keep_tokens>=kv_min_evict_tokens &&
                     cal_keep_tokens>kv_sink_tokens+kv_recent_tokens);
                evaluate_with_prior_tail(ctx,input,pos,prior,collect_prior,kv_tail_tokens);
                const double cmp_prefill_ms=now_ms()-pf0;
                const auto first=paired?truth_ids.front():greedy(ctx,llama_vocab_n_tokens(vocab));
                std::vector<float> cmp_logits;
                if(!paired){
                    auto logits=llama_get_logits_ith(ctx,-1);
                    cmp_logits.assign(logits,logits+llama_vocab_n_tokens(vocab));
                }
                double ev0=now_ms();
                J kv_cfg={{"kv_reduce_policy",collect_prior?"prefill_prior":"recent"},{"kv_keep_tokens",cal_keep_tokens},
                          {"kv_keep_ratio",-1.0},{"kv_sink_tokens",kv_sink_tokens},
                          {"kv_pinned_tokens",0},{"kv_recent_tokens",kv_recent_tokens},
                          {"kv_pool_window",kv_pool_window}};
                const int total_cal=int(prefix.size()+input.size());
                J cmp_kv=evict_kv_by_prior(ctx,total_cal,kv_cfg,prior,int(input.size()),-1,-1,nullptr);
                const double cmp_evict_ms=now_ms()-ev0;
                require(cmp_kv.value("evicted_count",0)>0,"calibrate hc_compact skipped compaction");
                llama_memory_clear(llama_get_memory(cpu->ctx),true);
                double hnd0=now_ms();
                const size_t cmp_handoff_bytes=transfer(ctx,cpu->ctx,kv_verification);
                require(cmp_handoff_bytes>0,"calibrate hc_compact handoff_bytes==0");
                if(!paired){
                    auto *destination=llama_get_logits_ith(cpu->ctx,-1);
                    require(destination,"missing hc_compact migration logits");
                    std::copy(cmp_logits.begin(),cmp_logits.end(),destination);
                }
                if(paired)htp->release_ctx();
                const double cmp_handoff_ms=now_ms()-hnd0;
                double d0=now_ms();
                auto *dec=cpu->ctx;std::vector<llama_token> cmp_ids;
                for(int i=0;i<d;++i){auto tok=paired?truth_ids[i]:i==0?first:greedy(dec,llama_vocab_n_tokens(vocab));cmp_ids.push_back(tok);evaluate(dec,{tok},pos);}
                const double cmp_decode_ms=now_ms()-d0;
                row["hc_compact_prepare_ms"]=setup_ms+preparation;
                row["hc_compact_prefill_ms"]=cmp_prefill_ms;
                row["hc_compact_evict_ms"]=cmp_evict_ms;
                row["hc_compact_handoff_ms"]=cmp_handoff_ms;
                row["hc_compact_handoff_bytes"]=cmp_handoff_bytes;
                row["hc_compact_decode_ms"]=cmp_decode_ms;
                row["hc_compact_ms"]=cmp_prefill_ms+cmp_evict_ms+cmp_handoff_ms;
                row["hc_compact_kept_count"]=cmp_kv.at("kept_count");
                row["hc_compact_evicted_count"]=cmp_kv.at("evicted_count");
                row["hc_compact_ids"]=cmp_ids;
                if(paired)row["paired_truth"]["HC-Compact"]={
                    {"prepare_ms",setup_ms+preparation},{"wait_ms",wait},{"restore_ms",0.0},
                    {"prefill_ms",cmp_prefill_ms},{"compact_ms",cmp_evict_ms},{"handoff_ms",cmp_handoff_ms},
                    {"decode_ms",cmp_decode_ms},{"post_return_ms",wait+(now_ms()-pf0)},{"decode_ids",cmp_ids},
                    {"collect_prior",collect_prior},{"handoff_bytes",cmp_handoff_bytes}};
            }
            rows.push_back(row);
        }
        anchor={};anchor_ids.clear();
        return {{"H",prefix.size()},{"P",input.size()},{"D",d},{"rows",rows}};
    }
    J infer(const J &body){
        a2o::memory::Ledger memory(memory_ledger_enabled, kv_hwm_trim);
        memory.sample("turn_start");
        J gap_probe=nullptr;
        // 常规 route 回退 CPU 时默认异步退役后台准备线程，避免在短 Gap 下同步阻塞。
        bool detach_preparation=(condition=="route");
        bool force_probe_cc=false;
        if (body.contains("frozen_tool_gap_ms")) {
            require(condition=="route" && body.contains("frozen_decode_ids") && anchor,
                    "device gap probe requires a frozen tool continuation on route");
            const double gap=body.at("frozen_tool_gap_ms").get<double>();
            require(std::isfinite(gap) && gap>=0.0 && gap<=10000.0, "invalid frozen tool gap");
            const auto mode=body.value("frozen_prepare_retirement",std::string("join"));
            require(mode=="join" || mode=="retire","invalid preparation retirement mode");
            detach_preparation=mode=="retire";
            force_probe_cc=body.value("frozen_force_cc_after_prepare",false);
            const auto begin=Clock::now();
            auto preparation=start_prepare(J::object());
            require(preparation.at("started").get<bool>(),"device gap probe failed to start preparation");
            std::this_thread::sleep_until(begin+std::chrono::duration<double,std::milli>(gap));
            const double actual=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            gap_probe={{"requested_gap_ms",gap},{"actual_gap_ms",actual},{"retirement",mode},
                       {"forced_cc",force_probe_cc},{"preparation",std::move(preparation)}};
        }
        double start=now_ms();auto chat=render(body);auto ids=tokens(vocab,chat.prompt,true);
        int limit=body.value("max_tokens",768);
        const bool frozen=body.contains("frozen_decode_ids");
        const bool force_hc=body.value("frozen_force_hc",false);
        std::string force_path=body.value("frozen_force_path",std::string(""));
        if(force_path.empty()){
            if(force_hc)force_path="hc_full";
            else if(force_probe_cc)force_path="cc";
        }
        const bool force_trim=force_path=="hc_trim";
        require(force_path.empty() || force_path=="cc" || force_path=="hc_full" ||
                force_path=="hc_compact" || force_trim,"invalid frozen_force_path");
        const bool force_any_hc=(force_path=="hc_full" || force_path=="hc_compact" || force_trim);
        const bool serial_context=cpu && !cpu->ctx;
        const auto lifecycle_probe=body.value("frozen_hc_lifecycle_probe",std::string());
        const bool sync_prepare=lifecycle_probe=="rebuild_sync";
        const bool probe_htp_decode=lifecycle_probe=="retain_htp_decode";
        const bool sync_hc=body.value("frozen_sync_hc",false);
        const bool retain_htp=lifecycle_probe=="retain" || probe_htp_decode || sync_hc;
        if(!lifecycle_probe.empty()){
            require(frozen && force_path=="hc_full" && shared_weights && !kv_reduce && !serial_context,
                    "lifecycle probe requires frozen shared HC-Full without compaction");
            require(sync_prepare || retain_htp || lifecycle_probe=="rebuild_async","invalid lifecycle probe mode");
            require(!sync_prepare || (!htp->ctx && !lane.occupied()),"synchronous probe requires an empty HTP lane");
        }
        if(sync_hc){
            require(frozen && force_path=="hc_full" && !kv_reduce && !serial_context && lifecycle_probe.empty(),
                    "sync HC requires frozen HC-Full without compaction or lifecycle probe");
            require(!lane.occupied(),"sync HC requires an empty preparation lane");
        }
        if(!force_path.empty())require(frozen,"forced path requires frozen decode");
        if(force_path=="hc_full")require(condition=="route" && htp,"forced HC-Full requires route and HTP context");
        if(force_path=="hc_compact")require(condition=="route" && htp && kv_reduce,"forced HC-Compact requires route, HTP context, and kv_reduce=1");
        if(force_trim)require(condition=="route" && htp && kv_reduce && kv_cost_table.valid &&
                              !m1_explorer_enabled,"forced HC trim requires route, a KV cost table and routing disabled");
        if(force_hc && force_path=="hc_full")require(!kv_reduce,"legacy frozen_force_hc requires full KV; use frozen_force_path=hc_compact for compact");
        if(serial_context)require(force_any_hc,"released CPU context requires a frozen HC probe");
        std::vector<llama_token> frozen_ids;
        double frozen_validation_ms=0.0;
        if(frozen){
            const double validation_start=now_ms();
            frozen_ids=body.at("frozen_decode_ids").get<std::vector<llama_token>>();
            require(!frozen_ids.empty() && frozen_ids.size()<=size_t(context_size),"invalid frozen decode length");
            const auto &expected=body.at("frozen_prompt_ids");
            require(expected.is_array() && expected.size()==ids.size() &&
                    std::equal(ids.begin(),ids.end(),expected.begin(),[](llama_token id,const J &value){
                        return id==value.get<llama_token>();
                    }),"frozen prompt IDs differ from rendered input");
            const int n_vocab=llama_vocab_n_tokens(vocab);
            for(auto tok:frozen_ids)require(tok>=0 && tok<n_vocab,"frozen decode token out of vocabulary");
            limit=int(frozen_ids.size());
            frozen_validation_ms=now_ms()-validation_start;
        }
        size_t common=0;while(common<std::min(ids.size(),anchor_ids.size()) && ids[common]==anchor_ids[common])++common;
        const bool reuse=anchor && !anchor_ids.empty() && common==anchor_ids.size();
        size_t pinned_tokens=reuse?anchor_pinned_tokens:0;
        if(!reuse){
            const auto &msgs=body.at("messages");
            if(msgs.is_array() && !msgs.empty() && msgs[0].value("role",std::string(""))=="system"){
                J sys_body={{"messages",J::array({msgs[0]})}};
                if(body.contains("tools") && !body["tools"].is_null())sys_body["tools"]=body["tools"];
                auto sys_chat=render(sys_body);
                auto sys_ids=tokens(vocab,sys_chat.prompt,true);
                size_t sys_common=0;
                while(sys_common<std::min(sys_ids.size(),ids.size()) && sys_ids[sys_common]==ids[sys_common])++sys_common;
                pinned_tokens=sys_common;
            }
        }
        std::vector<int> live_pos=reuse?anchor_live_positions:std::vector<int>{};
        const size_t h_eff=live_pos.size();
        size_t h=reuse?anchor_ids.size():0,p=ids.size()-h;
        const size_t physical_h=(reuse && compact_anchor)?h_eff:h;
        require(limit>=0,"max_tokens must be nonnegative");
        const size_t context_used=physical_h+p;
        require(context_used<=size_t(context_size),"context overflow; prompt exceeds capacity");
        const int available=int(size_t(context_size)-context_used);
        require(!frozen || limit<=available,"context overflow; refusing to truncate frozen decode");
        const bool context_limited=limit>available;
        limit=std::min(limit,available);
        const SessionKvWindow kv_window(kv_reduce,reuse,int(h_eff),int(p),int(pinned_tokens),kv_sink_tokens,kv_recent_tokens);
        const bool use_kv_cost=kv_window.can_reduce && !m1_explorer_enabled && (body.value("use_kv_cost_model",false) || kv_policy=="adaptive_cost" || (kv_cost_table.valid && kv_policy!="recent"));
        const int requested_d=body.value("expected_decode_tokens",kv_cost_table.valid?kv_cost_table.expected_decode_tokens:expected_decode_tokens);
        const int exp_d=context_limited?std::min(requested_d,limit):requested_d;
        if(use_kv_cost)require(kv_cost_table.valid,"adaptive_cost requires kv_model installed via /a2o/costs");
        const uint64_t anchor_version=anchor?anchor->version:0;
        std::string ready=lane.availability(anchor_version),route=condition=="htp"?"htp":"cpu",reason="fixed_backend";
        double wait_ms=0.0;
        J prediction=nullptr;
        J pending_wait=nullptr;
        std::optional<a2o_m1::M1JointDecision> ready_after_wait;
        auto select_m1=[&](bool allow_compact, bool allow_exploration){
            return m1_explorer.select_joint(int(h_eff),int(p),exp_d,kv_cost_table,true,
                allow_compact && kv_window.can_reduce,true,int(h),int(pinned_tokens),
                kv_sink_tokens,kv_window.recent,compact_anchor,allow_exploration);
        };
        auto m1_prediction=[&](const a2o_m1::M1JointDecision &m1)->J{
            return {{"kind","m1_explorer_joint_v1"},{"selected_path",m1.path},{"explored",m1.explored},
                    {"reason",m1.reason},{"h_eff",h_eff},{"h_active",m1.h_active},{"k_floor",m1.k_floor},
                    {"cc_mean_ms",m1.cc_mean_ms},{"cc_std_ms",m1.cc_std_ms},{"cc_lcb_ms",m1.cc_lcb_ms},
                    {"hc_full_mean_ms",m1.hc_full_mean_ms},{"hc_full_std_ms",m1.hc_full_std_ms},{"hc_full_lcb_ms",m1.hc_full_lcb_ms},
                    {"hc_compact_valid",m1.hc_compact_valid},{"hc_compact_eligible",m1.hc_compact_eligible},
                    {"hc_compact_mean_ms",m1.hc_compact_valid?J(m1.hc_compact_mean_ms):J(nullptr)},
                    {"hc_compact_std_ms",m1.hc_compact_valid?J(m1.hc_compact_std_ms):J(nullptr)},
                    {"hc_compact_lcb_ms",m1.hc_compact_valid?J(m1.hc_compact_lcb_ms):J(nullptr)},
                    {"single_turn_gain_ms",m1.single_turn_gain_ms},
                    {"delta_next_prefill_ms",m1.delta_next_prefill_ms},{"effective_gain_ms",m1.effective_gain_ms},
                    {"routing_p_min",m1_explorer.routing_p_min},{"beta",m1_explorer.beta},
                    {"max_explore_penalty_ms",m1_explorer.max_explore_penalty_ms},{"joint_kv_route",kv_reduce}};
        };
        J kv_cost_decision=nullptr;
        bool kv_admit=kv_window.can_reduce;
        int eff_keep=kv_keep_tokens;
        if (body.contains("frozen_kv_keep_tokens")) {
            require(frozen && kv_reduce && (!use_kv_cost || force_path=="hc_compact"), "frozen KV budget requires fixed-policy or forced HC-Compact frozen replay");
            eff_keep=body.at("frozen_kv_keep_tokens").get<int>();
            require(eff_keep>0 && eff_keep<=context_size, "frozen KV budget out of range");
        }
        std::string eff_policy=kv_policy;
        bool joint_decided=false;
        bool cold_hc=false;
        if(force_any_hc){
            if(sync_hc){
                require(ready=="absent","sync HC must not use background preparation");
                route="htp";
                kv_admit=false;
                reason="frozen_sync_hc_full";
            }else{
                require(reuse,"forced HC requires matching prepared history");
                if(ready=="pending"){
                    const double begin=now_ms();
                    lane.wait_ready(anchor_version,30000.0);
                    wait_ms=now_ms()-begin;
                    ready=lane.availability(anchor_version);
                }
                require(sync_prepare || ready=="ready","forced HC preparation did not complete");
                route="htp";
                if(force_path=="hc_compact"){
                    kv_admit=true;
                    reason="frozen_forced_hc_compact";
                    if(eff_policy=="adaptive_cost")eff_policy="prefill_prior";
                }else if(force_trim){
                    const auto trim=plan_kv_reduce(kv_cost_table,int(h_eff),int(p),exp_d,true,int(h),kv_window.keep_floor);
                    kv_cost_decision=kv_reduce_plan_to_json<J>(trim);
                    kv_admit=trim.admit;
                    eff_keep=trim.keep_tokens;
                    eff_policy="prefill_prior";
                    joint_decided=true;
                    reason="frozen_static_hc_trim";
                }else{
                    kv_admit=false;
                    reason="frozen_forced_hc_full";
                }
            }
        }else if(force_path=="cc"){
            if(force_probe_cc)require(reuse,"forced CPU probe requires the prepared anchor");
            route="cpu";
            kv_admit=false;
            joint_decided=true;
            reason=force_probe_cc?"frozen_probe_cc":"frozen_forced_cc";
        }else if(condition=="route" && anchor_ids.empty() && cold_costs.configured()){
            const auto point=cold_costs.estimate(static_cast<int>(p));
            reason="cold_outside_calibration";
            if(point){
                cold_hc=point->hc_ms<point->cc_ms;
                route=cold_hc?"htp":"cpu";
                reason="cold_route_cost";
                prediction={{"kind","cold_route_v1"},{"model_id",cold_costs.model_id},
                            {"cc_ms",point->cc_ms},{"hc_ms",point->hc_ms}};
            }
        }else if(condition=="route"){
            if(reuse && ready=="pending" && m1_explorer_enabled){
                // A pending lane uses expected costs and keeps the decision once it is ready.
                const auto m1_prep=select_m1(true,false);
                auto wait_path=m1_prep;
                const double elapsed_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-prepare_started_at).count();
                const double estimated_prep_ms=m1_explorer.cpu_to_htp.predict({1.0,double(h_eff)}).first;
                auto plan=[&](const a2o_m1::M1JointDecision &m1){
                    const double hc_ms=m1.kv_admit?m1.hc_compact_mean_ms:m1.hc_full_mean_ms;
                    return plan_greedy_return(false,elapsed_ms,estimated_prep_ms,m1.cc_mean_ms,hc_ms);
                };
                auto gp=plan(wait_path);
                // A future saving cannot pay for waiting on the current critical path.
                if(wait_path.kv_admit && !gp.continue_prepare){
                    wait_path=select_m1(false,false);
                    gp=plan(wait_path);
                }
                const bool attempted=wait_path.route=="htp" && gp.continue_prepare;
                if(attempted){
                    const double w0=now_ms();
                    lane.wait_ready(anchor_version,gp.wait_budget_ms);
                    wait_ms=now_ms()-w0;
                    ready=lane.availability(anchor_version);
                    if(ready=="ready")ready_after_wait=wait_path;
                }
                pending_wait={{"m1_prep",m1_prediction(m1_prep)},{"wait_path",wait_path.path},
                              {"elapsed_prep_ms",elapsed_ms},{"estimated_total_prep_ms",estimated_prep_ms},
                              {"estimated_remaining_prep_ms",gp.remaining_ms},{"continue_cost_ms",gp.continue_cost_ms},
                              {"wait_budget_ms",attempted?gp.wait_budget_ms:0.0},{"wait_ms",wait_ms},
                              {"attempted",attempted},{"availability_after",ready},
                              {"result",attempted?(ready=="ready"?"ready":"not_ready_after_wait"):"skipped"},
                              {"reason",wait_path.reason=="below_routing_p_min"?wait_path.reason:gp.reason}};
            }else if(reuse && ready=="pending" && !cost_model.is_null()){
                RouteLinearCostTable prep_tbl=parse_route_linear_cost_table(cost_model);
                JointRouteKvPlan jp_prep=plan_joint_route_kv(
                    prep_tbl,kv_cost_table,int(h_eff),int(p),exp_d,true,use_kv_cost,"cpu",int(h),kv_window.keep_floor);
                if(jp_prep.route=="htp" && jp_prep.cpu_best_total_ms>0.0 && jp_prep.htp_best_total_ms>0.0){
                    double elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prepare_started_at).count();
                    double pred_prep=prep_tbl.cpu_to_htp.eval(double(h_eff));
                    GreedyReturnPlan gp=plan_greedy_return(false,elapsed_ms,pred_prep,jp_prep.cpu_best_total_ms,jp_prep.htp_best_total_ms);
                    if(gp.continue_prepare && gp.wait_budget_ms>0.0){
                        double w0=now_ms();
                        lane.wait_ready(anchor_version,gp.wait_budget_ms);
                        wait_ms=now_ms()-w0;
                        ready=lane.availability(anchor_version);
                    }
                }
            }
            reason=ready!="ready"?"candidate_"+ready:!reuse?"anchor_prefix_mismatch":"calibration_missing";
            if(reuse && ready=="ready" && m1_explorer_enabled){
                const auto m1=ready_after_wait?*ready_after_wait:select_m1(true,true);
                route=m1.route;reason=m1.reason;
                prediction=m1_prediction(m1);
                prediction["prepare_ms_hidden"]=prepare_ms;
                if(kv_reduce){
                    kv_admit=m1.kv_admit;
                    eff_keep=m1.keep_tokens;
                    eff_policy="prefill_prior";
                    joint_decided=true;
                    kv_cost_decision={
                        {"admit",m1.kv_admit},{"keep_tokens",m1.keep_tokens},{"h_active",m1.h_active},
                        {"k_floor",m1.k_floor},{"H_total",int(h_eff+p)},
                        {"handoff_saving_ms",m1.handoff_saving_ms},{"decode_saving_ms",m1.decode_saving_ms},
                        {"prior_tax_ms",m1.prior_tax_ms},{"evict_ms",m1.evict_ms},{"reason",m1.reason},
                        {"single_turn_gain_ms",m1.single_turn_gain_ms},
                        {"delta_next_prefill_ms",m1.delta_next_prefill_ms},{"effective_gain_ms",m1.effective_gain_ms}
                    };
                }
            }else if(m1_explorer_enabled){
                // HTP not ready or no reusable anchor: fall back strictly to CC (no CPU-side KV compaction).
                route="cpu";
                kv_admit=false;
                joint_decided=true;
            }else if(reuse && ready=="ready" && !cost_model.is_null()) {
                RouteLinearCostTable route_tbl=parse_route_linear_cost_table(cost_model);
                JointRouteKvPlan jp=plan_joint_route_kv(
                    route_tbl,kv_cost_table,int(h_eff),int(p),exp_d,true,use_kv_cost,"cpu",int(h),kv_window.keep_floor);
                double restore=route_tbl.cpu_restore.eval(double(h_eff));
                double cp=route_tbl.cpu_prefill.eval(double(p),double(h_eff));
                double np=route_tbl.htp_prefill.eval(double(p),double(h_eff));
                double back=route_tbl.htp_to_cpu.eval(double(h_eff+p));
                route=jp.route;reason=jp.reason;
                prediction={{"kind","linear_hp_v1"},{"cpu_restore_ms",restore},{"cpu_prefill_ms",cp},
                    {"htp_prefill_ms",np},{"handoff_ms",back},{"cc_ms",jp.cc_ms},{"hc_ms",jp.hc_ms},
                    {"prepare_ms_hidden",prepare_ms},{"model_id",cost_model.at("model_id")},
                    {"extrapolated",jp.extrapolated},
                    {"h_eff",h_eff},{"cpu_no_kv_total_ms",jp.cpu_no_kv_total_ms},{"htp_no_kv_total_ms",jp.htp_no_kv_total_ms},
                    {"cpu_best_total_ms",jp.cpu_best_total_ms},{"htp_best_total_ms",jp.htp_best_total_ms},
                    {"selected_total_ms",jp.selected_total_ms},{"joint_kv_route",use_kv_cost && kv_cost_table.valid}};
                prediction["routing_p_min"]=cost_model.value("routing_p_min",size_t(16));
                if(use_kv_cost){
                    kv_cost_decision=kv_reduce_plan_to_json<J>(jp.kv_plan);
                    kv_admit=jp.kv_plan.admit;
                    eff_keep=jp.kv_plan.keep_tokens;
                    eff_policy="prefill_prior";
                    joint_decided=true;
                }
            } else if(reuse && ready=="ready") {
                // 兼容历史结果；新实验只安装线性模型。
                for(const auto &row:costs) if(h==row.at("h_min").get<size_t>() && p==row.at("p_min").get<size_t>()) {
                    prediction=row;route=row.at("hc_ms").get<double>()<row.at("cc_ms").get<double>()?"htp":"cpu";
                    reason="legacy_exact_calibration";break;
                }
            }
        }
        double observation_ms=now_ms()-start,restore_start=now_ms();
        llama_context *pre=route=="htp"?htp->ensure_ctx():home().ctx;
        bool cpu_with_retiring_preparation=false;
        double preparation_retirement_ms=0;
        if(cold_hc){
            lane.close();
            llama_memory_clear(llama_get_memory(pre),true);
        }else if(sync_prepare || sync_hc){
            if(reuse)restore_session(pre,*anchor,nullptr,kv_verification);
            else llama_memory_clear(llama_get_memory(pre),true);
        }else if(condition=="route" && route=="htp")lane.claim(anchor_version);
        else {
            const double retirement_started=now_ms();
            if(detach_preparation){
                // The retired worker keeps ownership of HTP until it exits.
                lane.retire();
                lane.poll();
                cpu_with_retiring_preparation=lane.occupied();
            }else{
                lane.close();
            }
            if(htp && route!="htp" && !cpu_with_retiring_preparation)htp->release_ctx();
            preparation_retirement_ms=now_ms()-retirement_started;
            if(reuse)restore_session(pre,*anchor,nullptr,kv_verification);else llama_memory_clear(llama_get_memory(pre),true);
        }
        double restore_ms=now_ms()-restore_start;
        std::vector<llama_token> suffix(ids.begin()+h,ids.end());llama_pos pos=h;
        // 剥离末尾空 <think> 块作为多轮锚点分界，避免下一轮渲染清除空 <think> 后 H_anchor 回退到 0。
        std::vector<llama_token> anchor_toks=ids;
        const std::string empty_think="<think>\n\n</think>\n\n";
        if(chat.prompt.size()>empty_think.size() && chat.prompt.compare(chat.prompt.size()-empty_think.size(),empty_think.size(),empty_think)==0){
            auto cand=tokens(vocab,std::string_view(chat.prompt).substr(0,chat.prompt.size()-empty_think.size()),true);
            if(cand.size()>h && cand.size()<ids.size() && std::equal(cand.begin(),cand.end(),ids.begin()))anchor_toks=std::move(cand);
        }
        if(use_kv_cost && !joint_decided){
            KvReducePlan kv_plan=plan_kv_reduce(kv_cost_table,int(h_eff),int(p),exp_d,condition=="route" && route=="htp",int(h),kv_window.keep_floor);
            kv_cost_decision=kv_reduce_plan_to_json<J>(kv_plan);
            kv_admit=kv_plan.admit;
            eff_keep=kv_plan.keep_tokens;
            eff_policy="prefill_prior";
        }
        const int active_after_prefill=int(h_eff+p);
        const bool use_ratio=(kv_keep_ratio>0.0 && kv_keep_ratio<1.0 && !use_kv_cost && !body.contains("frozen_kv_keep_tokens"));
        const int est_sink=std::min(active_after_prefill,int(std::max<size_t>(size_t(kv_sink_tokens),pinned_tokens)));
        const int est_recent=std::min(std::max(0,active_after_prefill-est_sink),kv_window.recent);
        const int est_mid=std::max(0,active_after_prefill-est_sink-est_recent);
        const int est_keep_target=use_ratio?(est_sink+est_recent+int(std::lround(kv_keep_ratio*double(est_mid)))):eff_keep;
        const int est_effective_keep=std::max(est_keep_target,est_sink+est_recent);
        const int est_evictable=std::max(0,active_after_prefill-est_effective_keep);
        const int est_prior_budget=std::max(0,est_keep_target-(est_sink+est_recent));
        const int ratio_floor=est_sink+kv_window.recent+64;
        const bool kv_will_reduce=kv_window.can_reduce && kv_admit && est_evictable>0 && (use_ratio ? (active_after_prefill>ratio_floor) : (active_after_prefill>eff_keep));
        // 若可淘汰 token 过少或无先验排序预算，直接退化为 recent，避免为极小裁剪量支付先验采集税。
        if(kv_will_reduce && eff_policy=="prefill_prior" && (est_prior_budget<=0 || est_evictable<kv_min_evict_tokens)){
            eff_policy="recent";
        }
        const bool collect_prior=kv_will_reduce && eff_policy=="prefill_prior";
        const bool legacy_anchor=body.value("frozen_legacy_snapshot",false);
        if(legacy_anchor)require(frozen,"legacy snapshot override requires frozen replay");
        const bool defer_anchor=!legacy_anchor && (!kv_will_reduce || compact_anchor);
        const bool defer_hc_anchor=shared_weights && route=="htp" && defer_anchor;
        const bool save_compact_anchor=compact_anchor && kv_will_reduce;
        const bool profile_graph=body.value("frozen_graph_profile",false);
        if(profile_graph)require(frozen,"graph profile requires frozen replay");
        PhoneGraphTrace prefill_graph(profile_graph), decode_graph(profile_graph);
        J clocks=J::object();
        const size_t anchor_split_in_suffix=anchor_toks.size()-h;
        J prefill_chunks=J::array();
        if(!lifecycle_probe.empty())require(profile_graph,"lifecycle probe requires graph profiling");
        auto measure_prefill=[&](size_t count,auto &&fn){
            double start=now_ms();
            if(lifecycle_probe.empty()){
                prefill_graph.measure(fn);
                return now_ms()-start;
            }
            PhoneGraphTrace part(true);
            part.measure(fn);
            const double elapsed=now_ms()-start;
            for(size_t i=0;i<part.totals.size();++i)prefill_graph.totals[i]+=part.totals[i];
            prefill_chunks.push_back({{"tokens",count},{"wall_ms",elapsed},{"graph",part.json()}});
            return elapsed;
        };
        std::vector<int> next_live_pos=live_pos;
        next_live_pos.reserve(h_eff+anchor_split_in_suffix);
        for(size_t i=h;i<anchor_toks.size();++i)next_live_pos.push_back(int(i));
        SessionSnapshot next;double prefill_ms=0,snapshot_ms=0;
        memory.sample("before_prefill");
        if(profile_graph)clocks["before_prefill"]=phone_clock_sample();
        prior.kv_callback_ms=0.0;
        {
            ScopedBigCoreAffinity big_core_scope(route=="htp" && htp_big_cores);
            const int exclude_tail_tokens=int(suffix.size()-anchor_split_in_suffix);
            if(anchor_split_in_suffix<suffix.size() && !defer_anchor && !save_compact_anchor){
                std::vector<llama_token> head_suf(suffix.begin(),suffix.begin()+anchor_split_in_suffix);
                std::vector<llama_token> tail_suf(suffix.begin()+anchor_split_in_suffix,suffix.end());
                // Acquire prior over the actual model input, before the template's empty thinking suffix.
                prefill_ms+=measure_prefill(head_suf.size(),[&]{evaluate_with_prior_tail(pre,head_suf,pos,prior,collect_prior,kv_tail_tokens);});
                double sn0=now_ms();next.state=export_seq(pre);next.next_pos=pos;next.evaluated_tokens=anchor_toks.size();next.version=++version;snapshot_ms=now_ms()-sn0;
                prefill_ms+=measure_prefill(tail_suf.size(),[&]{evaluate(pre,tail_suf,pos);});
            }else{
                prefill_ms=measure_prefill(suffix.size(),[&]{evaluate_with_prior_tail(pre,suffix,pos,prior,collect_prior,kv_tail_tokens,exclude_tail_tokens);});
                if(!defer_anchor && !save_compact_anchor){
                    double sn0=now_ms();next.state=export_seq(pre);next.next_pos=pos;next.evaluated_tokens=anchor_toks.size();next.version=++version;snapshot_ms=now_ms()-sn0;
                }else{next.version=++version;}
            }
        }
        live_pos.reserve(h_eff+p);
        for(size_t i=h;i<ids.size();++i)live_pos.push_back(int(i));
        if(cpu && pre==cpu->ctx)cpu_kv_hwm=std::max(cpu_kv_hwm,active_after_prefill);
        const bool hc=condition=="route" && route=="htp";
        // 首 token 使用产生本轮 prefill logits 的上下文。
        const bool sample_first_on_htp=hc && !frozen && limit>0;
        J kv_info=nullptr;double kv_reduce_ms=0;
        double cpu_context_rebuild_ms=0;
        double htp_context_release_ms=0;
        memory.sample("after_prefill");
        if(profile_graph)clocks["after_prefill"]=phone_clock_sample();
        J pre_handoff_resources=nullptr;
        if(force_any_hc)pre_handoff_resources=memory.checkpoint_process();
        if(force_path=="hc_compact")require(kv_will_reduce,"forced HC-Compact failed: KV reduction skipped (increase H+P or lower keep_tokens)");
        double handoff_start=now_ms();
        const double handoff_collection=memory.overhead_ms();
        size_t handoff_bytes=0;
        if(hc) {
            if(kv_will_reduce){
                double kv_start=now_ms();
                J kv_cfg={{"kv_reduce_policy",eff_policy},{"kv_keep_tokens",eff_keep},{"kv_keep_ratio",use_ratio?kv_keep_ratio:-1.0},{"kv_sink_tokens",kv_sink_tokens},
                          {"kv_pinned_tokens",pinned_tokens},{"kv_recent_tokens",kv_window.recent},{"kv_pool_window",kv_pool_window}};
                kv_info=evict_kv_by_prior(pre,active_after_prefill,kv_cfg,prior,int(suffix.size()),-1,-1,&live_pos);
                kv_reduce_ms=now_ms()-kv_start;
                llama_memory_clear(llama_get_memory(cpu->ctx),true);
                memory.sample("after_handoff_compaction");
            }
            auto peak=[&]{memory.sample("handoff_peak");};
            if(serial_context || memory_ledger_enabled){
                auto state=export_seq(pre);
                const auto max_pos=kv_verification==KvVerification::import_size
                    ? llama_memory_seq_pos_max(llama_get_memory(pre),0) : -1;
                if(serial_context){
                    double begin=now_ms();
                    htp->release_ctx();
                    htp_context_release_ms=now_ms()-begin;
                    begin=now_ms();
                    cpu->ensure_ctx();
                    cpu_context_rebuild_ms=now_ms()-begin;
                }
                handoff_bytes=import_seq(cpu->ctx,state,kv_verification,max_pos,peak);
            }else{
                handoff_bytes=transfer(pre,cpu->ctx,kv_verification);
            }
            if(!kv_info.is_null()){kv_info["compacted"]=true;kv_info["state_bytes_after"]=handoff_bytes;}
            if(!serial_context && !sample_first_on_htp && !retain_htp){
                const double begin=now_ms();
                htp->release_ctx();
                htp_context_release_ms=now_ms()-begin;
            }
        }
        double handoff_ms=now_ms()-handoff_start-kv_reduce_ms-(memory.overhead_ms()-handoff_collection);
        auto *decode=probe_htp_decode?pre:home().ctx;
        if(kv_will_reduce && !hc){
            double kv_start=now_ms();
            std::vector<float> first_logits;
            if(!frozen && limit>0){
                const auto *logits=llama_get_logits_ith(decode,-1);
                require(logits,"missing prefill logits");
                first_logits.assign(logits,logits+llama_vocab_n_tokens(vocab));
            }
            SessionSnapshot working;working.next_pos=pos;
            J kv_cfg={{"kv_reduce_policy",eff_policy},{"kv_keep_tokens",eff_keep},{"kv_keep_ratio",use_ratio?kv_keep_ratio:-1.0},{"kv_sink_tokens",kv_sink_tokens},
                      {"kv_pinned_tokens",pinned_tokens},{"kv_recent_tokens",kv_window.recent},{"kv_pool_window",kv_pool_window},{"kv_compact",true}};
            kv_info=run_kv_reduction(decode,working,kv_cfg,prior,int(suffix.size()),-1,-1,&live_pos);
            kv_info["state_bytes_before"]=next.state.empty()?size_t(0):next.state.size();
            if(!first_logits.empty()){
                auto *destination=llama_get_logits_ith(decode,-1);
                require(destination,"missing destination logits after kv reduction");
                std::copy(first_logits.begin(),first_logits.end(),destination);
            }
            kv_reduce_ms=now_ms()-kv_start;
        }
        memory.sample("before_decode");
        if(profile_graph)clocks["before_decode"]=phone_clock_sample();
        std::vector<llama_token> generated;
        double decode_ms;
        bool eog=false;
        J message={{"role","assistant"},{"content",nullptr}};
        J decode_resources=nullptr;
        std::string finish_reason="forced";
        if(frozen){
            const llama_pos before_decode=pos;
            auto resources_before=memory.checkpoint_process();
            double decode_start=now_ms();
            decode_graph.measure([&]{for(auto tok:frozen_ids)evaluate(decode,{tok},pos);});
            decode_ms=now_ms()-decode_start;
            if(profile_graph)clocks["after_decode"]=phone_clock_sample();
            memory.sample("after_decode");
            decode_resources={{"before",std::move(resources_before)},{"after",memory.checkpoint_process()}};
            require(pos-before_decode==limit,"incomplete frozen forward sequence");
            generated=std::move(frozen_ids);
        }else if(limit==0){
            decode_ms=0;
            memory.sample("after_decode");
            message["content"]="";
            finish_reason="length";
        }else{
            common_params_sampling sampling;sampling.temp=0;sampling.seed=body.value("seed",20261107);
            if(!chat.grammar.empty())sampling.grammar={COMMON_GRAMMAR_TYPE_TOOL_CALLS,chat.grammar};
            sampling.grammar_lazy=chat.grammar_lazy;
            sampling.grammar_triggers=chat.grammar_triggers;sampling.generation_prompt=chat.generation_prompt;
            for(const auto &text:chat.preserved_tokens){auto ts=tokens(vocab,text,false);if(ts.size()==1)sampling.preserved_tokens.insert(ts[0]);}
            auto sampler=std::unique_ptr<common_sampler,decltype(&common_sampler_free)>(common_sampler_init(home().model,sampling),common_sampler_free);
            require(bool(sampler),"sampler init failed");
            std::string raw;
            double decode_start=now_ms();
            for(int i=0;i<limit;++i){
                const bool first_on_htp=sample_first_on_htp && i==0;
                auto tok=common_sampler_sample(sampler.get(),first_on_htp?pre:decode,-1);
                if(first_on_htp){
                    const double release_start=now_ms();
                    htp->release_ctx();
                    const double release_ms=now_ms()-release_start;
                    handoff_ms+=release_ms;
                    decode_start+=release_ms;
                }
                common_sampler_accept(sampler.get(),tok,true);
                if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}
                generated.push_back(tok);raw+=piece(vocab,tok);evaluate(decode,{tok},pos);
            }
            decode_ms=now_ms()-decode_start;
            memory.sample("after_decode");
            common_chat_parser_params parser(chat);parser.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
            if(!chat.parser.empty())parser.parser.load(chat.parser);
            auto msg=common_chat_parse(raw,false,parser);msg.role="assistant";
            for(size_t i=0;i<msg.tool_calls.size();++i)msg.tool_calls[i].id="a2o_"+std::to_string(version)+"_"+std::to_string(i);
            finish_reason=!eog?"length":msg.tool_calls.empty()?"stop":"tool_calls";
            message=msg.to_json_oaicompat();
        }
        if(cpu && decode==cpu->ctx)cpu_kv_hwm=std::max(cpu_kv_hwm,int(live_pos.size()+generated.size()));
        lane.retire();lane.poll();
        if(defer_anchor || (compact_anchor && !kv_info.is_null())){
            double cs0=now_ms();
            auto *mem=llama_get_memory(decode);
            require(llama_memory_seq_rm(mem,0,llama_pos(anchor_toks.size()),-1),"anchor rollback seq_rm failed");
            next.state=export_seq(decode);
            next.next_pos=anchor_toks.size();next.evaluated_tokens=anchor_toks.size();
            live_pos.erase(std::lower_bound(live_pos.begin(),live_pos.end(),int(anchor_toks.size())),live_pos.end());
            next_live_pos=std::move(live_pos);
            snapshot_ms+=now_ms()-cs0;
            if(!(compact_anchor && !kv_info.is_null())){
                memory.sample("after_deferred_anchor");
            }
        }
        if(compact_anchor && !kv_info.is_null()){
            kv_info["compact_anchor"]=true;
            kv_info["anchor_state_bytes"]=next.state.size();
            kv_info["anchor_live_tokens"]=next_live_pos.size();
            const int kept_after=int(next_live_pos.size());
            memory.sample("after_compact_anchor");
            if(kv_hwm_trim && cpu && decode==cpu->ctx && cpu_kv_hwm-kept_after>=3072){
                double trim0=now_ms();
                int old_hwm=cpu_kv_hwm;
                cpu->release_ctx();
                cpu->ensure_ctx();
                cpu_kv_hwm=0;
                kv_info["kv_hwm_trimmed"]=true;
                kv_info["kv_hwm_before"]=old_hwm;
                kv_info["kv_trim_ms"]=now_ms()-trim0;
                memory.sample("after_kv_hwm_trim");
            }
        }
        if(force_path=="cc"){
            require(route=="cpu" && kv_info.is_null(),"forced CC validation failed: unexpected route or KV compaction");
        }else if(force_path=="hc_full"){
            require(route=="htp" && kv_info.is_null() && handoff_bytes>0,"forced HC-Full validation failed: silent CPU fallback or missing handoff");
        }else if(force_path=="hc_compact"){
            require(route=="htp" && !kv_info.is_null() && kv_info.value("compacted",false) && kv_info.value("evicted_count",0)>0 && handoff_bytes>0,
                    "forced HC-Compact validation failed: silent CPU fallback or skipped compaction");
        }else if(force_trim){
            require(route=="htp" && handoff_bytes>0 &&
                    (kv_admit ? !kv_info.is_null() && kv_info.value("compacted",false) : kv_info.is_null()),
                    "static HC trim validation failed: unexpected route or KV state");
        }
        memory.sample("turn_end");
        bool m1_online_updated=false;
        if(m1_explorer_enabled && condition=="route" && !cold_hc){
            const int h_active_obs=(!kv_info.is_null() && kv_info.contains("kept_count"))
                ? kv_info.at("kept_count").get<int>() : active_after_prefill;
            m1_online_updated=m1_explorer.observe(
                int(h_eff),int(p),route,h_active_obs,restore_ms,
                ready=="ready"?prepare_ms:-1.0,prefill_ms,handoff_ms,decode_ms,
                int(generated.size()),collect_prior);
        }
        J timing={{"prompt_ms",prefill_ms},{"predicted_ms",decode_ms},{"prompt_n",suffix.size()},
                  {"predicted_n",generated.size()},{"observation_ms",observation_ms},{"restore_ms",restore_ms},
                  {"handoff_ms",handoff_ms},{"handoff_bytes",handoff_bytes},{"snapshot_ms",snapshot_ms},
                  {"prior_ms",prior.kv_callback_ms},{"kv_reduce_ms",kv_reduce_ms},
                  {"deferred_hc_anchor",defer_hc_anchor},
                  {"kv_cost_decision",kv_cost_decision},{"kv_reduce",kv_info},{"total_ms",now_ms()-start},
                  {"route",route},{"decode_backend",(condition=="htp" || probe_htp_decode)?"htp":"cpu"},
                  {"route_reason",reason},{"availability",ready},{"H_anchor",h},
                  {"cpu_with_retiring_preparation",cpu_with_retiring_preparation},
                  {"H_effective",h_eff},{"pinned_tokens",pinned_tokens},{"kv_compact_anchor",compact_anchor},{"wait_ms",wait_ms},
                  {"P_evaluated",p},{"prepare_ms",ready=="ready"?prepare_ms:0},{"m1_online_updated",m1_online_updated},{"prediction",prediction}};
        timing["verify_kv_state"]=kv_verification==KvVerification::full_bytes;
        if(!pending_wait.is_null())timing["pending_wait"]=std::move(pending_wait);
        if(profile_graph)timing["graph_profile"]={{"prefill",prefill_graph.json()},
            {"decode",decode_graph.json()},{"clocks",std::move(clocks)}};
        if(!lifecycle_probe.empty())timing["hc_lifecycle_probe"]={{"mode",lifecycle_probe},
            {"retained_after_handoff",bool(htp->ctx)},{"prefill_chunks",std::move(prefill_chunks)}};
        if(!frozen)timing["decode_limit"]=limit;
        if(memory_ledger_enabled)timing["memory_ledger"]=memory.result();
        if(frozen){
            timing["frozen_validation_ms"]=frozen_validation_ms;
            timing["observation_excluding_frozen_validation_ms"]=observation_ms-frozen_validation_ms;
            timing["total_excluding_frozen_validation_ms"]=timing.at("total_ms").get<double>()-frozen_validation_ms;
            timing["decode_resources"]=std::move(decode_resources);
        }
        if(!gap_probe.is_null()){
            gap_probe["preparation_retirement_ms"]=preparation_retirement_ms;
            timing["gap_probe"]=std::move(gap_probe);
        }
        if(force_any_hc)timing["context_probe"]={{"serial_context",serial_context},
            {"cpu_context_rebuild_ms",cpu_context_rebuild_ms},{"htp_context_release_ms",htp_context_release_ms},
            {"before_handoff",std::move(pre_handoff_resources)}};
        J verbose_obj={{"tokens",generated},{"stop_type",frozen?"forced":eog?"eos":"limit"},{"truncated",context_limited && !eog}};
        if(!body.value("omit_verbose_prompt",false))verbose_obj["prompt"]=chat.prompt;
        timing["state_commit_ms"]=0.0;
        J response={{"id","a2o-"+std::to_string(version)},{"object","chat.completion"},{"created",0},{"model",model_id},
                {"choices",J::array({{{"index",0},{"message",message},{"finish_reason",finish_reason}}})},
                {"usage",{{"prompt_tokens",ids.size()},{"completion_tokens",generated.size()},{"total_tokens",ids.size()+generated.size()},
                          {"prompt_tokens_details",{{"cached_tokens",h}}}}},{"timings",std::move(timing)},
                {"__verbose",std::move(verbose_obj)}};
        const double commit_start=now_ms();
        anchor=std::make_shared<const SessionSnapshot>(std::move(next));
        anchor_ids=std::move(anchor_toks);
        anchor_live_positions=std::move(next_live_pos);
        anchor_pinned_tokens=pinned_tokens;
        response["timings"]["state_commit_ms"]=now_ms()-commit_start;
        return response;
    }
    #include "../profiling/stage_sweep.h"
#ifdef A2O_PREFILL_WEIGHT_AB
    #include "../profiling/prefill_weight_ab.h"
#endif
    #include "../profiling/state_cost_sweep.h"
    #include "../profiling/model_preflight.h"
    #include "../profiling/gap_replay.h"
    // Frozen-trajectory replay: evaluate supplied prompt and force supplied output
    // tokens.  No sampling or tool parsing is performed; this is a measurement
    // endpoint for host-collected traces.
    J replay(const J &body){
        auto prompt=body.at("prompt_ids").get<std::vector<llama_token>>();
        auto generated=body.value("generated_ids",J::array()).get<std::vector<llama_token>>();
        require(!prompt.empty(),"replay prompt is empty");
        require(prompt.size()+generated.size()<size_t(context_size),"replay context overflow");
        size_t h=body.value("H",size_t(0)), p=prompt.size()-h;
        require(h>0 && h<prompt.size(),"warm replay requires 0 < H < prompt length");
        require(cpu && htp,"paired replay requires route service");
        lane.close();
        anchor={};anchor_ids.clear();
        // Build/read/restore fixtures outside the measured return boundary.
        double bootstrap_start=now_ms();
        std::vector<llama_token> prefix(prompt.begin(),prompt.begin()+h);
        J cache_report;
        const char *cache_dir=std::getenv("A2O_REPLAY_CACHE_DIR");
        const auto &frozen=replay_cache->get(cpu->ctx,prefix,cache_dir?cache_dir:"",cache_report,kv_verification);
        llama_pos pos=h;
        double bootstrap_ms=now_ms()-bootstrap_start;
        if(body.value("cache_only",false))return {{"mode","cache_only"},{"cache",cache_report},{"H",h}};
        std::vector<llama_token> suffix(prompt.begin()+h,prompt.end());
        J results=J::array();
        auto order=body.value("condition_order",std::vector<std::string>{"cpu","htp","route"});
        if(htp)htp->ensure_ctx();
        auto sorted=order;std::sort(sorted.begin(),sorted.end());
        require(sorted==std::vector<std::string>({"cpu","htp","route"}),"condition_order must be a permutation of cpu/htp/route");
        for(const std::string &mode:order) {
            // All conditions begin with exactly the same authoritative prefix.
            double prepare_start=now_ms();
            if(mode=="htp" || mode=="route")restore_session(htp->ctx,frozen,nullptr,kv_verification);
            double preparation_ms=now_ms()-prepare_start;
            std::string route=mode=="htp"?"htp":"cpu", reason="fixed_backend";
            J prediction=nullptr;
            if(mode=="route") {
                require(!cost_model.is_null(),"paired replay requires cost model");
                auto f=[&](const char *k,double x){const auto &v=cost_model.at(k);return std::max(0.0,v.at("intercept_ms").get<double>()+v.at("slope_ms_per_token").get<double>()*x);};
                double cc=f("cpu_restore",h)+f("cpu_prefill",p),hc=f("htp_prefill",p)+f("htp_to_cpu",h+p);
                route=p>=cost_model.value("routing_p_min",size_t(16)) && hc<cc?"htp":"cpu";
                reason="linear_hp_cost";prediction={{"cc_ms",cc},{"hc_ms",hc},{"extrapolated",h>cost_model.at("h_max").get<size_t>()}};
            }
            auto *pre=route=="htp"?htp->ctx:cpu->ctx;
            double r0=now_ms();
            if(route=="cpu")restore_session(cpu->ctx,frozen,nullptr,kv_verification);
            double fixture_restore_ms=now_ms()-r0;
            double t0=now_ms();
            pos=h; double pf0=now_ms();evaluate(pre,suffix,pos);double prefill=now_ms()-pf0;
            double handoff0=now_ms();
            if(mode=="route" && route=="htp")transfer(pre,cpu->ctx,kv_verification);
            double handoff=now_ms()-handoff0;
            auto *dec=mode=="htp"?htp->ctx:cpu->ctx;double d0=now_ms();
            for(auto tok:generated)evaluate(dec,{tok},pos);
            double decode=now_ms()-d0;
            results.push_back({{"condition",mode},{"route",route},{"reason",reason},{"H",h},{"P",p},{"D",generated.size()},
                {"prepared_before_measurement_ms",preparation_ms},{"fixture_restore_ms",fixture_restore_ms},{"restore_ms",0.0},
                {"prefill_ms",prefill},{"handoff_ms",handoff},{"decode_ms",decode},{"total_ms",now_ms()-t0},{"prediction",prediction}});
        }
        anchor={};anchor_ids.clear();
        return {{"mode","paired_frozen_warm_replay"},{"bootstrap_ms",bootstrap_ms},{"cache",cache_report},{"results",results},
                {"scope","teacher-forced token evaluation; ready preparation excluded; not live E2E or quality measurement"}};
    }
};
struct AsyncChatRunner {
    PhoneService &svc;
    std::mutex mu;
    std::condition_variable cv;
    bool quit=false,has_pending=false;
    std::string cur_id,cur_status="idle",cur_error;
    J cur_req,cur_res;
    std::thread worker;
    explicit AsyncChatRunner(PhoneService &s):svc(s),worker([this]{run();}){}
    ~AsyncChatRunner(){
        {std::lock_guard<std::mutex> lk(mu);quit=true;cv.notify_all();}
        if(worker.joinable())worker.join();
    }
    void run(){
        for(;;){
            J req;std::string rid;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk,[&]{return has_pending||quit;});
                if(quit && !has_pending)break;
                req=std::move(cur_req);rid=cur_id;has_pending=false;
            }
            J out;std::string err;
            try{std::lock_guard<std::mutex> guard(svc.mutex);out=svc.infer(req);}
            catch(const std::exception &e){err=e.what();}
            std::lock_guard<std::mutex> lk(mu);
            if(cur_id==rid){
                if(err.empty()){cur_status="done";cur_res=std::move(out);}
                else{cur_status="error";cur_error=std::move(err);}
            }
            cv.notify_all();
        }
    }
    J start(J body){
        const std::string rid=body.at("request_id").get<std::string>();
        std::lock_guard<std::mutex> lk(mu);
        if(cur_id==rid)return J{{"request_id",rid},{"status",cur_status}};
        if(cur_status=="running")throw std::runtime_error("async chat runner busy");
        cur_id=rid;cur_status="running";cur_req=std::move(body);cur_res=nullptr;cur_error.clear();has_pending=true;
        cv.notify_all();
        return J{{"request_id",rid},{"status","running"}};
    }
    J wait_job(const J &body){
        const std::string rid=body.at("request_id").get<std::string>();
        const int wait_ms=std::clamp(body.value("wait_ms",15000),50,60000);
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk,std::chrono::milliseconds(wait_ms),[&]{return cur_id!=rid||cur_status!="running";});
        if(cur_id!=rid)return J{{"request_id",rid},{"status","unknown"}};
        if(cur_status=="done")return J{{"request_id",rid},{"status","done"},{"result",cur_res}};
        if(cur_status=="error")return J{{"request_id",rid},{"status","error"},{"error",cur_error}};
        return J{{"request_id",rid},{"status","running"}};
    }
    void wait_idle_and_clear(){
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk,[&]{return !has_pending && cur_status!="running";});
        cur_id.clear();cur_status="idle";cur_res=nullptr;cur_error.clear();
    }
};
int main(int argc,char **argv){try{
    require(argc==6,"usage: service model condition context port costs.json");
    context_size=std::stoi(argv[3]);batch_size=128;ggml_backend_load_all();
    const char *memory_flag=std::getenv("A2O_MEMORY_LEDGER");
    require(!memory_flag || std::string(memory_flag)=="0" || std::string(memory_flag)=="1", "invalid memory ledger flag");
    PhoneService service(argv[1],argv[2],memory_flag && std::string(memory_flag)=="1");
    AsyncChatRunner chat_runner(service);
    if(std::string(argv[5])!="none")service.costs=J::parse(read(argv[5])).get<std::vector<J>>();
    httplib::Server server;server.set_read_timeout(1800,0);server.set_write_timeout(1800,0);server.set_tcp_nodelay(true);
#ifndef _WIN32
    server.set_socket_options([](socket_t sock){
        httplib::default_socket_options(sock);
        int yes=1;
        ::setsockopt(sock,SOL_SOCKET,SO_KEEPALIVE,&yes,sizeof(yes));
#ifdef TCP_KEEPIDLE
        int idle=25,intvl=10,cnt=6;
        ::setsockopt(sock,IPPROTO_TCP,TCP_KEEPIDLE,&idle,sizeof(idle));
        ::setsockopt(sock,IPPROTO_TCP,TCP_KEEPINTVL,&intvl,sizeof(intvl));
        ::setsockopt(sock,IPPROTO_TCP,TCP_KEEPCNT,&cnt,sizeof(cnt));
#endif
    });
#endif
    auto handle=[&](auto fn){return [&,fn](const httplib::Request &req,httplib::Response &res){
        std::lock_guard<std::mutex> guard(service.mutex);
        try{auto body=req.body.empty()?J::object():J::parse(req.body);res.set_content(fn(body).dump(),"application/json");}
        catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    };};
    server.Get("/health",handle([&](const J &){return J{{"status","ok"},{"condition",service.condition},{"model",service.model_id},{"device",service.device_name},{"git_commit",A2O_GIT_COMMIT},{"build_id",A2O_BUILD_ID},{"single_resident",service.single_resident},{"shared_weights",service.shared_weights},{"cpu_repack",service.cpu_repack},{"htp_kv_offload",service.htp_kv_offload},{"htp_big_cores",service.htp_big_cores},{"flash_attention",service.flash_attention},{"kv_reduce",service.kv_reduce},{"kv_policy",service.kv_policy},{"kv_keep_tokens",service.kv_keep_tokens},{"kv_layer_stride",service.kv_layer_stride},{"kv_min_evict_tokens",service.kv_min_evict_tokens},{"kv_hwm_trim",service.kv_hwm_trim},{"memory_ledger_version",service.memory_ledger_enabled?1:0},{"verify_kv_state",service.kv_verification==KvVerification::full_bytes},{"kv_cost_loaded",service.kv_cost_table.valid},{"cold_cost_loaded",service.cold_costs.configured()},{"compact_anchor",service.compact_anchor},{"m1_explorer",service.m1_explorer_enabled},{"gap_ms",service.startup_gap_ms},{"expected_decode_tokens",service.expected_decode_tokens},{"init_ms",service.init_ms},{"context_size",context_size},{"frozen_decode_version",1},{"paired_truth_version",2},{"replay_cache_version",2},{"stage_sweep_version",1},{"cpu_threads",service.cpu_threads},{"htp_host_threads",4}};}));
    server.Post("/apply-template",handle([&](const J &b){return J{{"prompt",service.render(b).prompt}};}));
    server.Post("/tokenize",handle([&](const J &b){
        return J{{"tokens",tokens(service.vocab,b.at("content").get_ref<const std::string &>(),false)}};
    }));
    server.Post("/v1/chat/completions",handle([&](const J &b){return service.infer(b);}));
    server.Post("/a2o/chat-start",[&](const httplib::Request &req,httplib::Response &res){
        try{auto body=req.body.empty()?J::object():J::parse(req.body);res.set_content(chat_runner.start(std::move(body)).dump(),"application/json");}
        catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    });
    server.Post("/a2o/chat-wait",[&](const httplib::Request &req,httplib::Response &res){
        try{auto body=req.body.empty()?J::object():J::parse(req.body);res.set_content(chat_runner.wait_job(body).dump(),"application/json");}
        catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    });
    server.Post("/a2o/replay",handle([&](const J &b){return service.replay(b);}));
    server.Post("/a2o/stage-sweep",handle([&](const J &b){return service.stage_sweep(b);}));
#ifdef A2O_PREFILL_WEIGHT_AB
    server.Post("/a2o/prefill-weight-ab",handle([&](const J &b){return service.prefill_weight_ab(b);}));
#endif
    server.Post("/a2o/state-cost-sweep",handle([&](const J &b){return service.state_cost_sweep(b);}));
    server.Post("/a2o/state-residency",handle([&](const J &b){require(b.at("mode")=="dual","invalid residency mode");return service.enable_dual_state();}));
    server.Post("/a2o/model-preflight",handle([&](const J &b){return service.model_preflight(b);}));
    server.Post("/a2o/gap-replay",handle([&](const J &b){return service.gap_replay(b);}));
    server.Get("/a2o/memory-ledger",handle([&](const J &){return service.startup_memory;}));
    server.Post("/a2o/tool-start",handle([&](const J &b){return service.start_prepare(b);}));
    server.Post("/a2o/calibrate",handle([&](const J &b){return service.calibrate(b);}));
    server.Post("/a2o/costs",handle([&](const J &b){
        service.lane.close();
        if(b.value("clear_kv_model",false)){
            service.kv_cost_model=nullptr;service.kv_cost_table=KvReduceCostTable{};
            if(!b.contains("model") && !b.contains("rows") && !b.contains("kv_model") && !b.contains("cold_model"))
                return J{{"kv_model_cleared",true}};
        }
        J out=J::object();
        if(b.contains("cold_model")){
            const auto & data=b.at("cold_model");
            auto cold=data.is_null()?ColdRouteCosts{}:ColdRouteCosts::parse(data);
            require(!cold.configured() || (service.condition=="route" && service.cpu && service.htp &&
                    !service.single_resident && cold.model_id==service.model_id &&
                    cold.shared_weights==service.shared_weights),"cold route cost model identity or configuration mismatch");
            service.cold_costs=std::move(cold);
            out["cold_cost_loaded"]=service.cold_costs.configured();
        }
        if(b.contains("kv_model")){
            const auto &km=b.at("kv_model");
            service.kv_cost_table=parse_kv_reduce_cost_table(km);
            service.kv_cost_model=km;
            if(service.m1_explorer_enabled && service.m1_explorer.decode_per_tok.n_obs==0){
                service.m1_explorer.seed_from_linear_table(RouteLinearCostTable{},service.kv_cost_table);
            }
            out["kv_kind"]=km.at("kind");
            out["kv_model_id"]=km.value("model_id","");
        }
        if(b.contains("m1_explorer")){
            const auto &m1c=b.at("m1_explorer");
            if(m1c.is_boolean()){
                service.m1_explorer_enabled=m1c.get<bool>();
            }else if(m1c.is_object()){
                service.m1_explorer_enabled=m1c.value("enabled",true);
                service.m1_explorer.reset_defaults(m1c.value("forgetting",0.98));
                service.m1_explorer.routing_p_min=m1c.value("routing_p_min",16);
                service.m1_explorer.beta=m1c.value("beta",1.25);
                service.m1_explorer.max_explore_penalty_ms=m1c.value("max_explore_penalty_ms",4000.0);
                service.m1_explorer.exploration_enabled=m1c.value("exploration_enabled",true);
                service.m1_explorer.online_update_enabled=m1c.value("online_update_enabled",true);
                if(m1c.contains("probes") && m1c.at("probes").is_array()){
                    for(const auto &row:m1c.at("probes")){
                        const int ph=row.at("H").get<int>(), pp=row.at("P").get<int>();
                        if(row.contains("cc_restore_ms"))service.m1_explorer.cpu_restore.update({1.0,double(ph)},row.at("cc_restore_ms").get<double>());
                        if(row.contains("hc_prepare_ms"))service.m1_explorer.cpu_to_htp.update({1.0,double(ph)},row.at("hc_prepare_ms").get<double>());
                        if(row.contains("cpu_prefill_ms"))service.m1_explorer.cpu_prefill.update(service.m1_explorer.prefill_feat(ph,pp),row.at("cpu_prefill_ms").get<double>());
                        if(row.contains("htp_prefill_ms"))service.m1_explorer.htp_prefill.update(service.m1_explorer.prefill_feat(ph,pp),row.at("htp_prefill_ms").get<double>());
                        if(row.contains("htp_to_cpu_ms"))service.m1_explorer.htp_to_cpu.update({1.0,double(ph+pp)},row.at("htp_to_cpu_ms").get<double>());
                    }
                }
            }
            out["m1_explorer_enabled"]=service.m1_explorer_enabled;
        }
        if(b.contains("model")) {
            const auto &m=b.at("model");require(m.at("kind")=="linear_hp_v1","unsupported cost model");
            for(const char *key:{"cpu_restore","cpu_prefill","htp_prefill","htp_to_cpu","cpu_to_htp"}) {
                const auto &f=m.at(key);double a=f.at("intercept_ms"),s=f.at("slope_ms_per_token");
                double hp=f.contains("hp_slope_ms_per_1k_tokens")&&!f.at("hp_slope_ms_per_1k_tokens").is_null()?f.at("hp_slope_ms_per_1k_tokens").get<double>():0.0;
                require(std::isfinite(a)&&std::isfinite(s)&&std::isfinite(hp)&&a>=0&&s>=0&&hp>=0,"invalid linear fit");
            }
            require(m.at("h_max").get<size_t>()>=m.at("h_min").get<size_t>() &&
                    m.at("p_max").get<size_t>()>=m.at("p_min").get<size_t>(),"invalid model domain");
            service.cost_model=m;service.costs.clear();
            if(service.m1_explorer_enabled && service.m1_explorer.cpu_prefill.n_obs==0){
                service.m1_explorer.seed_from_linear_table(parse_route_linear_cost_table(m),service.kv_cost_table);
            }
            out["kind"]="linear_hp_v1";out["model_id"]=m.at("model_id");
            return out;
        }
        if(b.contains("rows")){
            service.cost_model=nullptr;service.costs=b.at("rows").get<std::vector<J>>();
            out["rows"]=service.costs.size();
            return out;
        }
        if(!out.empty())return out;
        throw std::runtime_error("missing model, cold_model, kv_model, or rows in /a2o/costs");
    }));
    server.Post("/a2o/reset",[&](const httplib::Request &req,httplib::Response &res){
        try{
            bool reset_explorer=false;
            if(!req.body.empty()){
                auto rb=J::parse(req.body);
                reset_explorer=rb.value("reset_explorer",false);
            }
            chat_runner.wait_idle_and_clear();
            std::lock_guard<std::mutex> guard(service.mutex);
            a2o::memory::Ledger memory(service.memory_ledger_enabled, service.kv_hwm_trim);
            memory.sample("before_reset");
            service.lane.close();service.anchor={};service.anchor_ids.clear();service.anchor_live_positions.clear();service.anchor_pinned_tokens=0;
            if(reset_explorer)service.m1_explorer.reset_defaults();
            if(service.kv_hwm_trim && service.cpu && service.cpu_kv_hwm>=3072){
                service.cpu->release_ctx();
                service.cpu->ensure_ctx();
                service.cpu_kv_hwm=0;
            }
            memory.sample("after_reset");
            J response={{"reset",true}};
            if(service.memory_ledger_enabled)response["memory_ledger"]=memory.result();
            res.set_content(response.dump(),"application/json");
        }catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    });
    J startup_meta={
        {"git_commit",A2O_GIT_COMMIT},
        {"build_id",A2O_BUILD_ID},
        {"device",service.device_name},
        {"model",service.model_id},
        {"shared_weights",service.shared_weights},
        {"compact_anchor",service.compact_anchor},
        {"kv_hwm_trim",service.kv_hwm_trim},
        {"memory_ledger_version",service.memory_ledger_enabled?1:0},
        {"verify_kv_state",service.kv_verification==KvVerification::full_bytes},
        {"kv_layer_stride",service.kv_layer_stride},
        {"kv_min_evict_tokens",service.kv_min_evict_tokens},
        {"gap_ms",service.startup_gap_ms}
    };
    std::cerr<<"A2O_META "<<startup_meta.dump()<<"\n";
    std::cerr<<"A2O_PHONE_READY "<<service.init_ms<<"\n";
    require(server.listen("0.0.0.0",std::stoi(argv[4])),"HTTP listen failed");return 0;
}catch(const std::exception &e){std::cerr<<"phone_service: "<<e.what()<<"\n";return 1;}}

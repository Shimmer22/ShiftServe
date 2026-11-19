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
#ifndef _WIN32
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif
#include "nlohmann/json.hpp"
using J=nlohmann::ordered_json;
#include "replay_cache.h"
#include "../scheduling/cold_route.h"
#include "../scheduling/greedy_return.h"
#include "../scheduling/kv_reduce_cost.h"
#include "../profiling/prefill_prior.h"
#include "../profiling/kv_reduce_probe.h"

struct PhoneService {
    std::string condition;
    std::unique_ptr<Engine> cpu,htp;
    common_chat_templates_ptr templates;
    const llama_vocab *vocab=nullptr;
    SessionSnapshot anchor;
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
    bool single_resident=false, shared_weights=false;
    int kv_keep_tokens=512, kv_sink_tokens=4, kv_recent_tokens=64, kv_pool_window=5, kv_tail_tokens=16, kv_layer_stride=4, kv_min_evict_tokens=256, expected_decode_tokens=96;
    bool kv_hwm_trim=true;
    int cpu_kv_hwm=0;
    std::string kv_policy="prefill_prior";
    PrefillPrior prior;
    std::string model_path;
    std::chrono::steady_clock::time_point prepare_started_at{};
    std::string model_id="qwen35-9b-q4_0";
    std::unique_ptr<FrozenReplayCache> replay_cache;
    PhoneService(const std::string &model,const std::string &which):condition(which) {
        model_path=model;
        if(const char *value=std::getenv("A2O_CPU_THREADS"))cpu_threads=std::stoi(value);
        require(cpu_threads>0 && cpu_threads<=16,"invalid CPU thread count");
        if(const char *value=std::getenv("A2O_CPU_REPACK")){require(std::string(value)=="0"||std::string(value)=="1","invalid CPU repack flag");cpu_repack=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_HTP_KV_OFFLOAD")){require(std::string(value)=="0"||std::string(value)=="1","invalid KV offload flag");htp_kv_offload=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_FLASH_ATTN")){require(std::string(value)=="0"||std::string(value)=="1","invalid flash attention flag");flash_attention=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_KV_REDUCE")){require(std::string(value)=="0"||std::string(value)=="1","invalid KV reduce flag");kv_reduce=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_KV_KEEP_TOKENS"))kv_keep_tokens=std::stoi(value);
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
        double start=now_ms();
        // kv_reduce 启用 prior 采集时挂载回调；prior.enabled=false 时 ask 阶段不抓取 tensor。
        if(which!="htp" && !shared_weights)cpu=std::make_unique<Engine>(model,false,cpu_repack,true,cpu_threads,flash_attention,need_prior_cb?&PrefillPrior::callback:nullptr,need_prior_cb?&prior:nullptr);
        if(which!="cpu" && !single_resident)htp=std::make_unique<Engine>(model,true,false,htp_kv_offload,4,flash_attention,(need_prior_cb && which=="route")?&PrefillPrior::callback:nullptr,(need_prior_cb && which=="route")?&prior:nullptr);
        if (shared_weights) {
            cpu = std::make_unique<Engine>(*htp, cpu_threads, flash_attention,
                    need_prior_cb ? &PrefillPrior::callback : nullptr, need_prior_cb ? &prior : nullptr);
        }
        require(cpu||htp,"single residency requires route or cpu startup");
        auto *m=cpu?cpu->model:htp->model;vocab=llama_model_get_vocab(m);
        templates=common_chat_templates_init(m,"");init_ms=now_ms()-start;
    }
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
        in.enable_thinking=false;in.chat_template_kwargs["enable_thinking"]="false";
        in.parallel_tool_calls=false;in.use_jinja=true;in.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
        return common_chat_templates_apply(templates.get(),in);
    }
    J start_prepare(){
        lane.poll();
        if(condition!="route" || anchor.state.empty())return {{"started",false},{"reason","disabled_or_no_anchor"}};
        if(lane.occupied())return {{"started",false},{"reason","previous_preparation_busy"}};
        auto state=std::make_shared<const SessionSnapshot>(anchor);
        prepare_started_at=std::chrono::steady_clock::now();
        bool started=lane.start(anchor.version,[this,state](const auto &valid){
            double start=now_ms();if(!valid())return false;
            restore_session(htp->ensure_ctx(),*state);prepare_ms=now_ms()-start;return valid();
        });
        return {{"started",started},{"version",anchor.version},{"H",anchor_ids.size()}};
    }
    J calibrate(const J &body){
        require(condition=="route" && cpu && htp,"calibration requires both explicit backends");
        lane.close();
        auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
        auto input=body.at("input_ids").get<std::vector<llama_token>>();
        int d=body.value("decode_evaluations",4);
        const bool cold=prefix.empty();
        require(!input.empty() && prefix.size()+input.size()+d<size_t(context_size),"invalid calibration shape");
        llama_pos pos=0;
        SessionSnapshot state;
        if(!cold){
            htp->ensure_ctx();
            llama_memory_clear(llama_get_memory(cpu->ctx),true);
            evaluate(cpu->ctx,prefix,pos);
            state.state=export_seq(cpu->ctx);
            state.next_pos=pos;
        }
        J rows=J::array();
        for(int repetition=0;repetition<body.value("repetitions",3);++repetition){
            J row={{"repetition",repetition}};
            std::vector<float> cpu_logits,htp_logits;
            if(cold)htp->release_ctx();
            for(bool accelerated:{false,true}){
                const double setup_start=now_ms();
                auto *ctx=accelerated?htp->ensure_ctx():cpu->ctx;
                const double setup_ms=now_ms()-setup_start;
                double ps=now_ms();restore_session(ctx,state);double preparation=now_ms()-ps;
                pos=prefix.size();double begin=now_ms();evaluate(ctx,input,pos);
                row[accelerated?"htp_prefill_ms":"cpu_prefill_ms"]=now_ms()-begin;
                auto logits=llama_get_logits_ith(ctx,-1);std::vector<float> values(logits,logits+llama_vocab_n_tokens(vocab));
                if(accelerated)htp_logits=values;else cpu_logits=values;
                auto first=greedy(ctx,llama_vocab_n_tokens(vocab));
                double handoff_start=now_ms();
                if(accelerated){transfer(ctx,cpu->ctx);auto *destination=llama_get_logits_ith(cpu->ctx,-1);require(destination,"missing migration logits");std::copy(values.begin(),values.end(),destination);}
                row[accelerated?"htp_to_cpu_ms":"cpu_handoff_ms"]=now_ms()-handoff_start;
                if(cold){
                    const double cost=setup_ms+preparation+now_ms()-begin;
                    row[accelerated?"cold_hc_ms":"cold_cc_ms"]=cost;
                }
                double decode_start=now_ms();
                auto *dec=cpu->ctx;std::vector<llama_token> ids;
                for(int i=0;i<d;++i){auto tok=i==0?first:greedy(dec,llama_vocab_n_tokens(vocab));ids.push_back(tok);evaluate(dec,{tok},pos);}
                row[accelerated?"hc_decode_ms":"cc_decode_ms"]=now_ms()-decode_start;
                row[accelerated?"hc_ms":"cc_ms"]=now_ms()-begin;
                row[accelerated?"hc_prepare_ms":"cc_restore_ms"]=preparation;
                row[accelerated?"hc_ids":"cc_ids"]=ids;
            }
            double dot=0,aa=0,bb=0,err=0;
            for(size_t i=0;i<cpu_logits.size();++i){require(std::isfinite(cpu_logits[i])&&std::isfinite(htp_logits[i]),"nonfinite calibration logits");dot+=cpu_logits[i]*htp_logits[i];aa+=cpu_logits[i]*cpu_logits[i];bb+=htp_logits[i]*htp_logits[i];err+=std::pow(cpu_logits[i]-htp_logits[i],2);}
            row["cosine"]=dot/std::sqrt(aa*bb);row["nrmse"]=std::sqrt(err/aa);
            row["passed"]=row["cosine"].get<double>()>=0.99 && row["cc_ids"]==row["hc_ids"];
            rows.push_back(row);
        }
        anchor={};anchor_ids.clear();
        return {{"H",prefix.size()},{"P",input.size()},{"D",d},{"rows",rows}};
    }
    J infer(const J &body){
        J gap_probe=nullptr;
        bool detach_preparation=false;
        bool force_probe_cc=false;
        if (body.contains("frozen_tool_gap_ms")) {
            require(condition=="route" && body.contains("frozen_decode_ids") && !anchor.state.empty(),
                    "device gap probe requires a frozen tool continuation on route");
            const double gap=body.at("frozen_tool_gap_ms").get<double>();
            require(std::isfinite(gap) && gap>=0.0 && gap<=10000.0, "invalid frozen tool gap");
            const auto mode=body.value("frozen_prepare_retirement",std::string("join"));
            require(mode=="join" || mode=="retire","invalid preparation retirement mode");
            detach_preparation=mode=="retire";
            force_probe_cc=body.value("frozen_force_cc_after_prepare",false);
            const auto begin=Clock::now();
            auto preparation=start_prepare();
            require(preparation.at("started").get<bool>(),"device gap probe failed to start preparation");
            std::this_thread::sleep_until(begin+std::chrono::duration<double,std::milli>(gap));
            const double actual=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            gap_probe={{"requested_gap_ms",gap},{"actual_gap_ms",actual},{"retirement",mode},
                       {"forced_cc",force_probe_cc},{"preparation",std::move(preparation)}};
        }
        double start=now_ms();auto chat=render(body);auto ids=tokens(vocab,chat.prompt,true);
        int limit=body.value("max_tokens",768);
        const bool frozen=body.contains("frozen_decode_ids");
        std::vector<llama_token> frozen_ids;
        if(frozen){
            frozen_ids=body.at("frozen_decode_ids").get<std::vector<llama_token>>();
            require(!frozen_ids.empty() && frozen_ids.size()<=size_t(context_size),"invalid frozen decode length");
            require(ids==body.at("frozen_prompt_ids").get<std::vector<llama_token>>(),"frozen prompt IDs differ from rendered input");
            const int n_vocab=llama_vocab_n_tokens(vocab);
            for(auto tok:frozen_ids)require(tok>=0 && tok<n_vocab,"frozen decode token out of vocabulary");
            limit=int(frozen_ids.size());
        }
        size_t common=0;while(common<std::min(ids.size(),anchor_ids.size()) && ids[common]==anchor_ids[common])++common;
        bool reuse=!anchor_ids.empty() && common==anchor_ids.size();
        if(!reuse){
            anchor_pinned_tokens=0;
            const auto &msgs=body.at("messages");
            if(msgs.is_array() && !msgs.empty() && msgs[0].value("role",std::string(""))=="system"){
                J sys_body={{"messages",J::array({msgs[0]})}};
                if(body.contains("tools") && !body["tools"].is_null())sys_body["tools"]=body["tools"];
                auto sys_chat=render(sys_body);
                auto sys_ids=tokens(vocab,sys_chat.prompt,true);
                size_t sys_common=0;
                while(sys_common<std::min(sys_ids.size(),ids.size()) && sys_ids[sys_common]==ids[sys_common])++sys_common;
                anchor_pinned_tokens=sys_common;
            }
        }
        std::vector<int> live_pos=reuse?anchor_live_positions:std::vector<int>{};
        const size_t h_eff=live_pos.size();
        size_t h=reuse?anchor_ids.size():0,p=ids.size()-h;
        const size_t physical_h=(reuse && compact_anchor)?h_eff:h;
        require(physical_h+p+limit<=size_t(context_size),"context overflow; refusing truncation");
        const bool use_kv_cost=kv_reduce && (body.value("use_kv_cost_model",false) || kv_policy=="adaptive_cost" || (kv_cost_table.valid && kv_policy!="recent"));
        const int exp_d=body.value("expected_decode_tokens",kv_cost_table.valid?kv_cost_table.expected_decode_tokens:expected_decode_tokens);
        if(use_kv_cost)require(kv_cost_table.valid,"adaptive_cost requires kv_model installed via /a2o/costs");
        std::string ready=lane.availability(anchor.version),route=condition=="htp"?"htp":"cpu",reason="fixed_backend";
        double wait_ms=0.0;
        J prediction=nullptr;
        J kv_cost_decision=nullptr;
        bool kv_admit=kv_reduce;
        int eff_keep=kv_keep_tokens;
        if (body.contains("frozen_kv_keep_tokens")) {
            require(frozen && kv_reduce && !use_kv_cost, "frozen KV budget requires fixed-policy frozen replay");
            eff_keep=body.at("frozen_kv_keep_tokens").get<int>();
            require(eff_keep>0 && eff_keep<=context_size, "frozen KV budget out of range");
        }
        std::string eff_policy=kv_policy;
        bool joint_decided=false;
        bool cold_hc=false;
        if(force_probe_cc){
            require(reuse,"forced CPU probe requires the prepared anchor");
            reason="frozen_probe_cc";
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
            if(reuse && ready=="pending" && !cost_model.is_null()){
                RouteLinearCostTable prep_tbl=parse_route_linear_cost_table(cost_model);
                JointRouteKvPlan jp_prep=plan_joint_route_kv(
                    prep_tbl,kv_cost_table,int(h_eff),int(p),exp_d,true,use_kv_cost,"cpu",int(h),int(anchor_pinned_tokens));
                if(jp_prep.route=="htp" && jp_prep.cpu_best_total_ms>0.0 && jp_prep.htp_best_total_ms>0.0){
                    double elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prepare_started_at).count();
                    double pred_prep=prep_tbl.cpu_to_htp.eval(double(h_eff));
                    GreedyReturnPlan gp=plan_greedy_return(false,elapsed_ms,pred_prep,jp_prep.cpu_best_total_ms,jp_prep.htp_best_total_ms);
                    if(gp.continue_prepare && gp.wait_budget_ms>0.0){
                        double w0=now_ms();
                        lane.wait_ready(anchor.version,gp.wait_budget_ms);
                        wait_ms=now_ms()-w0;
                        ready=lane.availability(anchor.version);
                    }
                }
            }
            reason=ready!="ready"?"candidate_"+ready:!reuse?"anchor_prefix_mismatch":"calibration_missing";
            if(reuse && ready=="ready" && !cost_model.is_null()) {
                RouteLinearCostTable route_tbl=parse_route_linear_cost_table(cost_model);
                JointRouteKvPlan jp=plan_joint_route_kv(
                    route_tbl,kv_cost_table,int(h_eff),int(p),exp_d,true,use_kv_cost,"cpu",int(h),int(anchor_pinned_tokens));
                double restore=route_tbl.cpu_restore.eval(double(h_eff));
                double cp=route_tbl.cpu_prefill.eval(double(p));
                double np=route_tbl.htp_prefill.eval(double(p));
                double back=route_tbl.htp_to_cpu.eval(double(h_eff+p));
                route=jp.route;reason=jp.reason;
                prediction={{"kind","linear_hp_v1"},{"cpu_restore_ms",restore},{"cpu_prefill_ms",cp},
                    {"htp_prefill_ms",np},{"handoff_ms",back},{"cc_ms",jp.cc_ms},{"hc_ms",jp.hc_ms},
                    {"prepare_ms_hidden",prepare_ms},{"model_id",cost_model.at("model_id")},
                    {"extrapolated",jp.extrapolated},
                    {"h_eff",h_eff},{"cpu_no_kv_total_ms",jp.cpu_no_kv_total_ms},{"htp_no_kv_total_ms",jp.htp_no_kv_total_ms},
                    {"cpu_best_total_ms",jp.cpu_best_total_ms},{"htp_best_total_ms",jp.htp_best_total_ms},
                    {"selected_total_ms",jp.selected_total_ms},{"joint_kv_route",use_kv_cost && kv_cost_table.valid}};
                prediction["routing_p_min"]=cost_model.value("routing_p_min",size_t(32));
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
        }else if(condition=="route" && route=="htp")lane.claim(anchor.version);
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
            if(reuse)restore_session(pre,anchor);else llama_memory_clear(llama_get_memory(pre),true);
        }
        std::vector<uint8_t>{}.swap(anchor.state);
        double restore_ms=now_ms()-restore_start;
        std::vector<llama_token> suffix(ids.begin()+h,ids.end());llama_pos pos=h;
        // 剥离末尾空 <think> 块作为多轮锚点分界，避免下一轮渲染清除空 <think> 后 H_anchor 回退到 0。
        std::vector<llama_token> anchor_toks=ids;
        const std::string empty_think="<think>\n\n</think>\n\n";
        if(chat.prompt.size()>empty_think.size() && chat.prompt.compare(chat.prompt.size()-empty_think.size(),empty_think.size(),empty_think)==0){
            auto cand=tokens(vocab,chat.prompt.substr(0,chat.prompt.size()-empty_think.size()),true);
            if(cand.size()>h && cand.size()<ids.size() && std::equal(cand.begin(),cand.end(),ids.begin()))anchor_toks=std::move(cand);
        }
        if(use_kv_cost && !joint_decided){
            KvReducePlan kv_plan=plan_kv_reduce(kv_cost_table,int(h_eff),int(p),exp_d,condition=="route" && route=="htp",int(h),int(anchor_pinned_tokens));
            kv_cost_decision=kv_reduce_plan_to_json<J>(kv_plan);
            kv_admit=kv_plan.admit;
            eff_keep=kv_plan.keep_tokens;
            eff_policy="prefill_prior";
        }
        const int active_after_prefill=int(h_eff+p);
        const int est_sink=std::min(active_after_prefill,int(std::max<size_t>(size_t(kv_sink_tokens),anchor_pinned_tokens)));
        const int est_recent=std::min(std::max(0,active_after_prefill-est_sink),kv_recent_tokens);
        const int est_effective_keep=std::max(eff_keep,est_sink+est_recent);
        const int est_evictable=std::max(0,active_after_prefill-est_effective_keep);
        const int est_prior_budget=std::max(0,eff_keep-(est_sink+est_recent));
        const bool kv_will_reduce=kv_admit && est_evictable>0 && active_after_prefill>eff_keep;
        // 若可淘汰 token 过少或无先验排序预算，直接退化为 recent，避免为极小裁剪量支付先验采集税。
        if(kv_will_reduce && eff_policy=="prefill_prior" && (est_prior_budget<=0 || est_evictable<kv_min_evict_tokens)){
            eff_policy="recent";
        }
        const bool collect_prior=kv_will_reduce && eff_policy=="prefill_prior";
        const size_t anchor_split_in_suffix=anchor_toks.size()-h;
        std::vector<int> next_live_pos=live_pos;
        next_live_pos.reserve(h_eff+anchor_split_in_suffix);
        for(size_t i=h;i<anchor_toks.size();++i)next_live_pos.push_back(int(i));
        SessionSnapshot next;double prefill_ms=0,snapshot_ms=0;
        if(anchor_split_in_suffix<suffix.size()){
            std::vector<llama_token> head_suf(suffix.begin(),suffix.begin()+anchor_split_in_suffix);
            std::vector<llama_token> tail_suf(suffix.begin()+anchor_split_in_suffix,suffix.end());
            // Acquire prior over the actual model input, before the template's empty thinking suffix.
            double pf0=now_ms();evaluate_with_prior_tail(pre,head_suf,pos,prior,collect_prior,kv_tail_tokens);prefill_ms+=now_ms()-pf0;
            if(!compact_anchor || !kv_will_reduce){
                double sn0=now_ms();next.state=export_seq(pre);next.next_pos=pos;next.evaluated_tokens=anchor_toks.size();next.version=++version;snapshot_ms=now_ms()-sn0;
            }else{next.version=++version;}
            double pf1=now_ms();evaluate(pre,tail_suf,pos);prefill_ms+=now_ms()-pf1;
        }else{
            double pf_start=now_ms();evaluate_with_prior_tail(pre,suffix,pos,prior,collect_prior,kv_tail_tokens);prefill_ms=now_ms()-pf_start;
            if(!compact_anchor || !kv_will_reduce){
                double sn0=now_ms();next.state=export_seq(pre);next.next_pos=pos;next.evaluated_tokens=anchor_toks.size();next.version=++version;snapshot_ms=now_ms()-sn0;
            }else{next.version=++version;}
        }
        live_pos.reserve(h_eff+p);
        for(size_t i=h;i<ids.size();++i)live_pos.push_back(int(i));
        // A fresh CPU context has no logits slot. Sample the first HC token
        // directly from HTP, then submit it and subsequent tokens to CPU.
        const bool sample_cold_htp=cold_hc && !frozen && limit>0;
        std::vector<float> first_logits;
        if(!frozen && !cold_hc && (kv_will_reduce || (condition=="route" && route=="htp"))){
            const float *fl=llama_get_logits_ith(pre,-1);require(fl,"missing prefill logits");
            first_logits.assign(fl,fl+llama_vocab_n_tokens(vocab));
        }
        J kv_info=nullptr;double kv_reduce_ms=0;
        double handoff_start=now_ms();
        size_t handoff_bytes=0;
        if(condition=="route" && route=="htp") {
            if(kv_will_reduce){
                double kv_start=now_ms();
                J kv_cfg={{"kv_reduce_policy",eff_policy},{"kv_keep_tokens",eff_keep},{"kv_sink_tokens",kv_sink_tokens},
                          {"kv_pinned_tokens",anchor_pinned_tokens},{"kv_recent_tokens",kv_recent_tokens},{"kv_pool_window",kv_pool_window}};
                kv_info=evict_kv_by_prior(pre,active_after_prefill,kv_cfg,prior,int(suffix.size()),-1,-1,&live_pos);
                kv_reduce_ms=now_ms()-kv_start;
                llama_memory_clear(llama_get_memory(cpu->ctx),true);
            }
            handoff_bytes=transfer(pre,cpu->ctx);
            if(!kv_info.is_null()){kv_info["compacted"]=true;kv_info["state_bytes_after"]=handoff_bytes;}
            if(!first_logits.empty()){
                auto *destination=llama_get_logits_ith(cpu->ctx,-1);
                require(destination,"missing destination logits after state migration");
                std::copy(first_logits.begin(),first_logits.end(),destination);
            }
            if(!sample_cold_htp)htp->release_ctx();
        }
        double handoff_ms=now_ms()-handoff_start-kv_reduce_ms;
        auto *decode=home().ctx;
        if(kv_will_reduce && !(condition=="route" && route=="htp")){
            double kv_start=now_ms();
            SessionSnapshot working;working.next_pos=pos;
            J kv_cfg={{"kv_reduce_policy",eff_policy},{"kv_keep_tokens",eff_keep},{"kv_sink_tokens",kv_sink_tokens},
                      {"kv_pinned_tokens",anchor_pinned_tokens},{"kv_recent_tokens",kv_recent_tokens},{"kv_pool_window",kv_pool_window},{"kv_compact",true}};
            kv_info=run_kv_reduction(decode,working,kv_cfg,prior,int(suffix.size()),-1,-1,&live_pos);
            kv_info["state_bytes_before"]=next.state.empty()?size_t(0):next.state.size();
            if(!first_logits.empty()){
                auto *destination=llama_get_logits_ith(decode,-1);
                require(destination,"missing destination logits after kv reduction");
                std::copy(first_logits.begin(),first_logits.end(),destination);
            }
            kv_reduce_ms=now_ms()-kv_start;
        }
        std::vector<llama_token> generated;
        double decode_ms;
        bool eog=false;
        J message={{"role","assistant"},{"content",nullptr}};
        J decode_resources=nullptr;
        std::string finish_reason="forced";
        if(frozen){
            const llama_pos before_decode=pos;
            auto resources_before=resource_sample();
            double decode_start=now_ms();
            for(auto tok:frozen_ids)evaluate(decode,{tok},pos);
            decode_ms=now_ms()-decode_start;
            decode_resources={{"before",resources_before},{"after",resource_sample()}};
            require(pos-before_decode==limit,"incomplete frozen forward sequence");
            generated=std::move(frozen_ids);
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
                const bool first_on_htp=sample_cold_htp && i==0;
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
            common_chat_parser_params parser(chat);parser.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
            if(!chat.parser.empty())parser.parser.load(chat.parser);
            auto msg=common_chat_parse(raw,false,parser);msg.role="assistant";
            for(size_t i=0;i<msg.tool_calls.size();++i)msg.tool_calls[i].id="a2o_"+std::to_string(version)+"_"+std::to_string(i);
            finish_reason=!eog?"length":msg.tool_calls.empty()?"stop":"tool_calls";
            message=msg.to_json_oaicompat();
        }
        if(cpu && decode==cpu->ctx)cpu_kv_hwm=std::max(cpu_kv_hwm,int(pos));
        lane.retire();lane.poll();
        if(compact_anchor && !kv_info.is_null()){
            double cs0=now_ms();
            auto *mem=llama_get_memory(decode);
            require(llama_memory_seq_rm(mem,0,llama_pos(anchor_toks.size()),-1),"compact_anchor rollback seq_rm failed");
            next.state=export_seq(decode);
            next.next_pos=anchor_toks.size();next.evaluated_tokens=anchor_toks.size();
            std::vector<int> trimmed;trimmed.reserve(live_pos.size());
            for(int lp:live_pos)if(lp<int(anchor_toks.size()))trimmed.push_back(lp);
            anchor_live_positions=std::move(trimmed);
            snapshot_ms+=now_ms()-cs0;
            kv_info["compact_anchor"]=true;
            kv_info["anchor_state_bytes"]=next.state.size();
            kv_info["anchor_live_tokens"]=anchor_live_positions.size();
            int kept_after=kv_info.value("kept_count",int(anchor_toks.size()));
            if(kv_hwm_trim && cpu && decode==cpu->ctx && cpu_kv_hwm-kept_after>=3072){
                double trim0=now_ms();
                int old_hwm=cpu_kv_hwm;
                cpu->release_ctx();
                cpu->ensure_ctx();
                cpu_kv_hwm=kept_after;
                kv_info["kv_hwm_trimmed"]=true;
                kv_info["kv_hwm_before"]=old_hwm;
                kv_info["kv_trim_ms"]=now_ms()-trim0;
            }
        }else{
            anchor_live_positions=std::move(next_live_pos);
        }
        anchor=std::move(next);anchor_ids=std::move(anchor_toks);
        J timing={{"prompt_ms",prefill_ms},{"predicted_ms",decode_ms},{"prompt_n",suffix.size()},
                  {"predicted_n",generated.size()},{"observation_ms",observation_ms},{"restore_ms",restore_ms},
                  {"handoff_ms",handoff_ms},{"handoff_bytes",handoff_bytes},{"snapshot_ms",snapshot_ms},{"kv_reduce_ms",kv_reduce_ms},
                  {"kv_cost_decision",kv_cost_decision},{"kv_reduce",kv_info},{"total_ms",now_ms()-start},
                  {"route",route},{"route_reason",reason},{"availability",ready},{"H_anchor",h},
                  {"cpu_with_retiring_preparation",cpu_with_retiring_preparation},
                  {"H_effective",h_eff},{"pinned_tokens",anchor_pinned_tokens},{"kv_compact_anchor",compact_anchor},{"wait_ms",wait_ms},
                  {"P_evaluated",p},{"prepare_ms",ready=="ready"?prepare_ms:0},{"prediction",prediction}};
        if(frozen)timing["decode_resources"]=std::move(decode_resources);
        if(!gap_probe.is_null()){
            gap_probe["preparation_retirement_ms"]=preparation_retirement_ms;
            timing["gap_probe"]=std::move(gap_probe);
        }
        J verbose_obj={{"tokens",generated},{"stop_type",frozen?"forced":eog?"eos":"limit"},{"truncated",false}};
        if(!body.value("omit_verbose_prompt",false))verbose_obj["prompt"]=chat.prompt;
        return {{"id","a2o-"+std::to_string(version)},{"object","chat.completion"},{"created",0},{"model",model_id},
                {"choices",J::array({{{"index",0},{"message",message},{"finish_reason",finish_reason}}})},
                {"usage",{{"prompt_tokens",ids.size()},{"completion_tokens",generated.size()},{"total_tokens",ids.size()+generated.size()},
                          {"prompt_tokens_details",{{"cached_tokens",h}}}}},{"timings",timing},
                {"__verbose",verbose_obj}};
    }
    #include "../profiling/stage_sweep.h"
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
        auto frozen=replay_cache->get(cpu->ctx,prefix,cache_dir?cache_dir:"",cache_report);
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
            if(mode=="htp" || mode=="route")restore_session(htp->ctx,frozen);
            double preparation_ms=now_ms()-prepare_start;
            std::string route=mode=="htp"?"htp":"cpu", reason="fixed_backend";
            J prediction=nullptr;
            if(mode=="route") {
                require(!cost_model.is_null(),"paired replay requires cost model");
                auto f=[&](const char *k,double x){const auto &v=cost_model.at(k);return std::max(0.0,v.at("intercept_ms").get<double>()+v.at("slope_ms_per_token").get<double>()*x);};
                double cc=f("cpu_restore",h)+f("cpu_prefill",p),hc=f("htp_prefill",p)+f("htp_to_cpu",h+p);
                route=p>=cost_model.value("routing_p_min",size_t(32)) && hc<cc?"htp":"cpu";
                reason="linear_hp_cost";prediction={{"cc_ms",cc},{"hc_ms",hc},{"extrapolated",h>cost_model.at("h_max").get<size_t>()}};
            }
            auto *pre=route=="htp"?htp->ctx:cpu->ctx;
            double r0=now_ms();
            if(route=="cpu")restore_session(cpu->ctx,frozen);
            double fixture_restore_ms=now_ms()-r0;
            double t0=now_ms();
            pos=h; double pf0=now_ms();evaluate(pre,suffix,pos);double prefill=now_ms()-pf0;
            double handoff0=now_ms();
            if(mode=="route" && route=="htp")transfer(pre,cpu->ctx);
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
    J start(const J &body){
        const std::string rid=body.at("request_id").get<std::string>();
        std::lock_guard<std::mutex> lk(mu);
        if(cur_id==rid)return J{{"request_id",rid},{"status",cur_status}};
        if(cur_status=="running")throw std::runtime_error("async chat runner busy");
        cur_id=rid;cur_status="running";cur_req=body;cur_res=nullptr;cur_error.clear();has_pending=true;
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
    PhoneService service(argv[1],argv[2]);
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
    server.Get("/health",handle([&](const J &){return J{{"status","ok"},{"condition",service.condition},{"model",service.model_id},{"single_resident",service.single_resident},{"shared_weights",service.shared_weights},{"cpu_repack",service.cpu_repack},{"htp_kv_offload",service.htp_kv_offload},{"flash_attention",service.flash_attention},{"kv_reduce",service.kv_reduce},{"kv_policy",service.kv_policy},{"kv_keep_tokens",service.kv_keep_tokens},{"kv_layer_stride",service.kv_layer_stride},{"kv_min_evict_tokens",service.kv_min_evict_tokens},{"kv_cost_loaded",service.kv_cost_table.valid},{"cold_cost_loaded",service.cold_costs.configured()},{"compact_anchor",service.compact_anchor},{"expected_decode_tokens",service.expected_decode_tokens},{"init_ms",service.init_ms},{"context_size",context_size},{"frozen_decode_version",1},{"replay_cache_version",2},{"stage_sweep_version",1},{"cpu_threads",service.cpu_threads},{"htp_host_threads",4},{"build_id",__DATE__ " " __TIME__}};}));
    server.Post("/apply-template",handle([&](const J &b){return J{{"prompt",service.render(b).prompt}};}));
    server.Post("/tokenize",handle([&](const J &b){return J{{"tokens",tokens(service.vocab,b.at("content"),false)}};}));
    server.Post("/v1/chat/completions",handle([&](const J &b){return service.infer(b);}));
    server.Post("/a2o/chat-start",[&](const httplib::Request &req,httplib::Response &res){
        try{auto body=req.body.empty()?J::object():J::parse(req.body);res.set_content(chat_runner.start(body).dump(),"application/json");}
        catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    });
    server.Post("/a2o/chat-wait",[&](const httplib::Request &req,httplib::Response &res){
        try{auto body=req.body.empty()?J::object():J::parse(req.body);res.set_content(chat_runner.wait_job(body).dump(),"application/json");}
        catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    });
    server.Post("/a2o/replay",handle([&](const J &b){return service.replay(b);}));
    server.Post("/a2o/stage-sweep",handle([&](const J &b){return service.stage_sweep(b);}));
    server.Post("/a2o/state-cost-sweep",handle([&](const J &b){return service.state_cost_sweep(b);}));
    server.Post("/a2o/state-residency",handle([&](const J &b){require(b.at("mode")=="dual","invalid residency mode");return service.enable_dual_state();}));
    server.Post("/a2o/model-preflight",handle([&](const J &b){return service.model_preflight(b);}));
    server.Post("/a2o/gap-replay",handle([&](const J &b){return service.gap_replay(b);}));
    server.Post("/a2o/tool-start",handle([&](const J &){return service.start_prepare();}));
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
            out["kv_kind"]=km.at("kind");
            out["kv_model_id"]=km.value("model_id","");
        }
        if(b.contains("model")) {
            const auto &m=b.at("model");require(m.at("kind")=="linear_hp_v1","unsupported cost model");
            for(const char *key:{"cpu_restore","cpu_prefill","htp_prefill","htp_to_cpu","cpu_to_htp"}) {
                const auto &f=m.at(key);double a=f.at("intercept_ms"),s=f.at("slope_ms_per_token");
                require(std::isfinite(a)&&std::isfinite(s)&&a>=0&&s>=0,"invalid linear fit");
            }
            require(m.at("h_max").get<size_t>()>=m.at("h_min").get<size_t>() &&
                    m.at("p_max").get<size_t>()>=m.at("p_min").get<size_t>(),"invalid model domain");
            service.cost_model=m;service.costs.clear();
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
            (void)req;
            chat_runner.wait_idle_and_clear();
            std::lock_guard<std::mutex> guard(service.mutex);
                        service.lane.close();service.anchor={};service.anchor_ids.clear();service.anchor_live_positions.clear();service.anchor_pinned_tokens=0;
            if(service.kv_hwm_trim && service.cpu && service.cpu_kv_hwm>=3072){
                service.cpu->release_ctx();
                service.cpu->ensure_ctx();
                service.cpu_kv_hwm=0;
            }
            res.set_content(J{{"reset",true}}.dump(),"application/json");
        }catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    });
    std::cerr<<"A2O_PHONE_READY "<<service.init_ms<<"\n";
    require(server.listen("0.0.0.0",std::stoi(argv[4])),"HTTP listen failed");return 0;
}catch(const std::exception &e){std::cerr<<"phone_service: "<<e.what()<<"\n";return 1;}}

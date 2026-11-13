// 私有研究服务：复用原工程的推理/状态机制和 llama.cpp 官方模板、解析及采样器。
#define main a2o_legacy_main
#define json a2o_json_string
#include "../agent_runtime.cpp"
#undef json
#undef main
#include "chat.h"
#include "sampling.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
using J=nlohmann::ordered_json;
#include "replay_cache.h"
#include "../scheduling/greedy_return.h"

struct PhoneService {
    std::string condition;
    std::unique_ptr<Engine> cpu,htp;
    common_chat_templates_ptr templates;
    const llama_vocab *vocab=nullptr;
    SessionSnapshot anchor;
    std::vector<llama_token> anchor_ids;
    PreparationLane lane;
    std::mutex mutex;
    double init_ms=0,prepare_ms=0;
    std::vector<J> costs;
    J cost_model=nullptr;
    uint64_t version=0;
    int cpu_threads=4;
    bool htp_kv_offload=false, flash_attention=false;
    bool single_resident=false;
    std::string model_path;
    std::string model_id="qwen35-9b-q4_0";
    std::unique_ptr<FrozenReplayCache> replay_cache;
    PhoneService(const std::string &model,const std::string &which):condition(which) {
        model_path=model;
        if(const char *value=std::getenv("A2O_CPU_THREADS"))cpu_threads=std::stoi(value);
        require(cpu_threads>0 && cpu_threads<=16,"invalid CPU thread count");
        if(const char *value=std::getenv("A2O_HTP_KV_OFFLOAD")){require(std::string(value)=="0"||std::string(value)=="1","invalid KV offload flag");htp_kv_offload=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_FLASH_ATTN")){require(std::string(value)=="0"||std::string(value)=="1","invalid flash attention flag");flash_attention=std::string(value)=="1";}
        if(const char *value=std::getenv("A2O_MODEL_ID"))model_id=value;
        if(const char *value=std::getenv("A2O_SINGLE_RESIDENT")){require(std::string(value)=="0"||std::string(value)=="1","invalid residency flag");single_resident=std::string(value)=="1";}
        replay_cache=std::make_unique<FrozenReplayCache>(model+":"+std::to_string(std::filesystem::file_size(model))+":"+std::to_string(static_cast<long long>(std::filesystem::last_write_time(model).time_since_epoch().count()))+":"+__DATE__+":"+__TIME__+":cpu_threads="+std::to_string(cpu_threads));
        double start=now_ms();
        if(which!="htp")cpu=std::make_unique<Engine>(model,false,false,true,cpu_threads,flash_attention);
        if(which!="cpu" && !single_resident)htp=std::make_unique<Engine>(model,true,false,htp_kv_offload,4,flash_attention);
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
        bool started=lane.start(anchor.version,[this,state](const auto &valid){
            double start=now_ms();if(!valid())return false;
            restore_session(htp->ctx,*state);prepare_ms=now_ms()-start;return valid();
        });
        return {{"started",started},{"version",anchor.version},{"H",anchor_ids.size()}};
    }
    J calibrate(const J &body){
        require(condition=="route","calibration requires both explicit backends");
        lane.close();
        auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
        auto input=body.at("input_ids").get<std::vector<llama_token>>();
        int d=body.value("decode_evaluations",4);
        require(!input.empty() && prefix.size()+input.size()+d<size_t(context_size),"invalid calibration shape");
        llama_memory_clear(llama_get_memory(cpu->ctx),true);llama_pos pos=0;evaluate(cpu->ctx,prefix,pos);
        SessionSnapshot state;state.state=export_seq(cpu->ctx);state.next_pos=pos;
        J rows=J::array();
        for(int repetition=0;repetition<body.value("repetitions",3);++repetition){
            J row={{"repetition",repetition}};
            std::vector<float> cpu_logits,htp_logits;
            for(bool accelerated:{false,true}){
                auto *ctx=accelerated?htp->ctx:cpu->ctx;
                double ps=now_ms();restore_session(ctx,state);double preparation=now_ms()-ps;
                pos=prefix.size();double begin=now_ms();evaluate(ctx,input,pos);
                row[accelerated?"htp_prefill_ms":"cpu_prefill_ms"]=now_ms()-begin;
                auto logits=llama_get_logits_ith(ctx,-1);std::vector<float> values(logits,logits+llama_vocab_n_tokens(vocab));
                if(accelerated)htp_logits=values;else cpu_logits=values;
                auto first=greedy(ctx,llama_vocab_n_tokens(vocab));
                double handoff_start=now_ms();
                if(accelerated){transfer(ctx,cpu->ctx);auto *destination=llama_get_logits_ith(cpu->ctx,-1);require(destination,"missing migration logits");std::copy(values.begin(),values.end(),destination);}
                row[accelerated?"htp_to_cpu_ms":"cpu_handoff_ms"]=now_ms()-handoff_start;
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
        double start=now_ms();auto chat=render(body);auto ids=tokens(vocab,chat.prompt,true);
        int limit=body.value("max_tokens",768);
        require(ids.size()+limit<=size_t(context_size),"context overflow; refusing truncation");
        size_t common=0;while(common<std::min(ids.size(),anchor_ids.size()) && ids[common]==anchor_ids[common])++common;
        bool reuse=!anchor_ids.empty() && common==anchor_ids.size();
        std::string ready=lane.availability(anchor.version),route=condition=="htp"?"htp":"cpu",reason="fixed_backend";
        J prediction=nullptr;
        size_t h=reuse?anchor_ids.size():0,p=ids.size()-h;
        if(condition=="route"){
            reason=ready!="ready"?"candidate_"+ready:!reuse?"anchor_prefix_mismatch":"calibration_missing";
            if(reuse && ready=="ready" && !cost_model.is_null()) {
                auto predict=[&](const char *key,double x){
                    const auto &fit=cost_model.at(key);
                    return std::max(0.0,fit.at("intercept_ms").get<double>()+fit.at("slope_ms_per_token").get<double>()*x);
                };
                double restore=predict("cpu_restore",h),cp=predict("cpu_prefill",p);
                double np=predict("htp_prefill",p),back=predict("htp_to_cpu",h+p);
                double cc=restore+cp,hc=np+back;
                prediction={{"kind","linear_hp_v1"},{"cpu_restore_ms",restore},{"cpu_prefill_ms",cp},
                    {"htp_prefill_ms",np},{"handoff_ms",back},{"cc_ms",cc},{"hc_ms",hc},
                    {"prepare_ms_hidden",prepare_ms},{"model_id",cost_model.at("model_id")},
                    {"extrapolated",h<cost_model.at("h_min").get<size_t>() || h>cost_model.at("h_max").get<size_t>() ||
                        p<cost_model.at("p_min").get<size_t>() || p>cost_model.at("p_max").get<size_t>()}};
                route=hc<cc?"htp":"cpu";reason="linear_hp_cost";
                if(p<cost_model.value("routing_p_min",size_t(32))) {route="cpu";reason="below_routing_p_min";}
                prediction["routing_p_min"]=cost_model.value("routing_p_min",size_t(32));
            } else if(reuse && ready=="ready") {
                // 兼容历史结果；新实验只安装线性模型。
                for(const auto &row:costs) if(h==row.at("h_min").get<size_t>() && p==row.at("p_min").get<size_t>()) {
                    prediction=row;route=row.at("hc_ms").get<double>()<row.at("cc_ms").get<double>()?"htp":"cpu";
                    reason="legacy_exact_calibration";break;
                }
            }
        }
        double observation_ms=now_ms()-start,restore_start=now_ms();
        llama_context *pre=route=="htp"?htp->ctx:home().ctx;
        if(condition=="route" && route=="htp")lane.claim(anchor.version);
        else {
            lane.retire();
            if(reuse)restore_session(pre,anchor);else llama_memory_clear(llama_get_memory(pre),true);
        }
        double restore_ms=now_ms()-restore_start;
        std::vector<llama_token> suffix(ids.begin()+h,ids.end());llama_pos pos=h;
        double pf_start=now_ms();evaluate(pre,suffix,pos);double prefill_ms=now_ms()-pf_start;
        std::vector<float> first_logits(llama_get_logits_ith(pre,-1),llama_get_logits_ith(pre,-1)+llama_vocab_n_tokens(vocab));
        double handoff_start=now_ms();
        if(condition=="route" && route=="htp") {
            transfer(pre,cpu->ctx);
            auto *destination=llama_get_logits_ith(cpu->ctx,-1);
            require(destination,"missing destination logits after state migration");
            std::copy(first_logits.begin(),first_logits.end(),destination);
        }
        double handoff_ms=now_ms()-handoff_start;
        auto *decode=home().ctx;
        // 保存完整 prompt 的 recurrent+conv+KV 状态。下一轮以此为稳定锚点，
        // 重评估 assistant 工具调用的规范化文本；不尝试裁剪不可逆的 DeltaNet 状态。
        double snapshot_start=now_ms();
        SessionSnapshot next;next.state=export_seq(decode);next.next_pos=pos;
        next.evaluated_tokens=ids.size();next.version=++version;
        double snapshot_ms=now_ms()-snapshot_start;
        common_params_sampling sampling;sampling.temp=0;sampling.seed=body.value("seed",20261107);
        if(!chat.grammar.empty())sampling.grammar={COMMON_GRAMMAR_TYPE_TOOL_CALLS,chat.grammar};
        sampling.grammar_lazy=chat.grammar_lazy;
        sampling.grammar_triggers=chat.grammar_triggers;sampling.generation_prompt=chat.generation_prompt;
        for(const auto &text:chat.preserved_tokens){auto ts=tokens(vocab,text,false);if(ts.size()==1)sampling.preserved_tokens.insert(ts[0]);}
        auto sampler=std::unique_ptr<common_sampler,decltype(&common_sampler_free)>(common_sampler_init(home().model,sampling),common_sampler_free);
        require(bool(sampler),"sampler init failed");
        std::string raw;std::vector<llama_token> generated;bool eog=false;
        double decode_start=now_ms();
        for(int i=0;i<limit;++i){
            auto tok=common_sampler_sample(sampler.get(),decode,-1);
            common_sampler_accept(sampler.get(),tok,true);
            if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}
            generated.push_back(tok);raw+=piece(vocab,tok);evaluate(decode,{tok},pos);
        }
        double decode_ms=now_ms()-decode_start;
        common_chat_parser_params parser(chat);parser.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
        if(!chat.parser.empty())parser.parser.load(chat.parser);
        auto msg=common_chat_parse(raw,false,parser);msg.role="assistant";
        for(size_t i=0;i<msg.tool_calls.size();++i)msg.tool_calls[i].id="a2o_"+std::to_string(version)+"_"+std::to_string(i);
        lane.retire();lane.poll();anchor=std::move(next);anchor_ids=ids;
        auto message=msg.to_json_oaicompat();
        J timing={{"prompt_ms",prefill_ms},{"predicted_ms",decode_ms},{"prompt_n",suffix.size()},
                  {"predicted_n",generated.size()},{"observation_ms",observation_ms},{"restore_ms",restore_ms},
                  {"handoff_ms",handoff_ms},{"snapshot_ms",snapshot_ms},{"total_ms",now_ms()-start},
                  {"route",route},{"route_reason",reason},{"availability",ready},{"H_anchor",h},
                  {"P_evaluated",p},{"prepare_ms",ready=="ready"?prepare_ms:0},{"prediction",prediction}};
        return {{"id","a2o-"+std::to_string(version)},{"object","chat.completion"},{"created",0},{"model",model_id},
                {"choices",J::array({{{"index",0},{"message",message},{"finish_reason",!eog?"length":msg.tool_calls.empty()?"stop":"tool_calls"}}})},
                {"usage",{{"prompt_tokens",ids.size()},{"completion_tokens",generated.size()},{"total_tokens",ids.size()+generated.size()},
                          {"prompt_tokens_details",{{"cached_tokens",h}}}}},{"timings",timing},
                {"__verbose",{{"prompt",chat.prompt},{"tokens",generated},{"stop_type",eog?"eos":"limit"},{"truncated",false}}}};
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
int main(int argc,char **argv){try{
    require(argc==6,"usage: service model condition context port costs.json");
    context_size=std::stoi(argv[3]);batch_size=128;ggml_backend_load_all();
    PhoneService service(argv[1],argv[2]);
    if(std::string(argv[5])!="none")service.costs=J::parse(read(argv[5])).get<std::vector<J>>();
    httplib::Server server;server.set_read_timeout(1800,0);server.set_write_timeout(1800,0);
    auto handle=[&](auto fn){return [&,fn](const httplib::Request &req,httplib::Response &res){
        std::lock_guard<std::mutex> guard(service.mutex);
        try{auto body=req.body.empty()?J::object():J::parse(req.body);res.set_content(fn(body).dump(),"application/json");}
        catch(const std::exception &e){res.status=500;res.set_content(J{{"error",e.what()}}.dump(),"application/json");}
    };};
    server.Get("/health",handle([&](const J &){return J{{"status","ok"},{"condition",service.condition},{"model",service.model_id},{"single_resident",service.single_resident},{"htp_kv_offload",service.htp_kv_offload},{"flash_attention",service.flash_attention},{"init_ms",service.init_ms},{"context_size",context_size},{"replay_cache_version",2},{"stage_sweep_version",1},{"cpu_threads",service.cpu_threads},{"htp_host_threads",4},{"build_id",__DATE__ " " __TIME__}};}));
    server.Post("/apply-template",handle([&](const J &b){return J{{"prompt",service.render(b).prompt}};}));
    server.Post("/tokenize",handle([&](const J &b){return J{{"tokens",tokens(service.vocab,b.at("content"),false)}};}));
    server.Post("/v1/chat/completions",handle([&](const J &b){return service.infer(b);}));
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
        if(b.contains("model")) {
            const auto &m=b.at("model");require(m.at("kind")=="linear_hp_v1","unsupported cost model");
            for(const char *key:{"cpu_restore","cpu_prefill","htp_prefill","htp_to_cpu","cpu_to_htp"}) {
                const auto &f=m.at(key);double a=f.at("intercept_ms"),s=f.at("slope_ms_per_token");
                require(std::isfinite(a)&&std::isfinite(s)&&a>=0&&s>=0,"invalid linear fit");
            }
            require(m.at("h_max").get<size_t>()>=m.at("h_min").get<size_t>() &&
                    m.at("p_max").get<size_t>()>=m.at("p_min").get<size_t>(),"invalid model domain");
            service.cost_model=m;service.costs.clear();
            return J{{"kind","linear_hp_v1"},{"model_id",m.at("model_id")}};
        }
        service.cost_model=nullptr;service.costs=b.at("rows").get<std::vector<J>>();
        return J{{"rows",service.costs.size()}};
    }));
    server.Post("/a2o/reset",handle([&](const J &){service.lane.close();service.anchor={};service.anchor_ids.clear();return J{{"reset",true}};}));
    std::cerr<<"A2O_PHONE_READY "<<service.init_ms<<"\n";
    require(server.listen("0.0.0.0",std::stoi(argv[4])),"HTTP listen failed");return 0;
}catch(const std::exception &e){std::cerr<<"phone_service: "<<e.what()<<"\n";return 1;}}

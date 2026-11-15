// 冻结 token + 实际观测 G 的异步准备重放；不执行生成工具，也不测任务质量。
J gap_replay(const J &body) {
    require(cpu && htp,"gap replay requires explicit CPU and HTP");
    auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
    auto input=body.at("input_ids").get<std::vector<llama_token>>();
    auto generated=body.at("generated_ids").get<std::vector<llama_token>>();
    auto order=body.at("order").get<std::vector<std::string>>();
    bool allow_partial_order=body.value("allow_partial_order",false);
    if(allow_partial_order){
        require(!order.empty(),"empty gap replay order");
        for(const auto &m:order)require(m=="cpu"||m=="htp"||m=="route","invalid gap replay mode: "+m);
    }else{
        auto sorted=order;std::sort(sorted.begin(),sorted.end());
        require(sorted==std::vector<std::string>({"cpu","htp","route"}),"invalid gap replay order");
    }
    const bool kv_reduce_req=body.value("kv_reduce",false);
    const bool kv_reduce_cpu=body.value("kv_reduce_on_cpu",false);
    const std::string kv_pol=body.value("kv_reduce_policy",kv_policy);
    const int kv_keep=body.value("kv_keep_tokens",kv_keep_tokens);
    const int kv_sink=body.value("kv_sink_tokens",kv_sink_tokens);
    const int kv_recent=body.value("kv_recent_tokens",kv_recent_tokens);
    const int kv_pool=body.value("kv_pool_window",kv_pool_window);
    const int kv_tail=body.value("kv_tail_tokens",kv_tail_tokens);
    const bool verify_free=body.value("verify_free_generation",false);
    const int free_tokens=body.value("free_tokens",128);
    if(kv_reduce_req && kv_pol=="prefill_prior")require(kv_reduce,"gap_replay prefill_prior requires service A2O_KV_REDUCE=1");
    require(!prefix.empty() && !input.empty() && !generated.empty() &&
        prefix.size()+input.size()+generated.size()<size_t(context_size),"invalid gap replay shape");
    double gap=body.at("gap_ms").get<double>();
    require(std::isfinite(gap) && gap>=0 && gap<120000,"invalid observed gap");
    const auto predicted=body.at("prediction");
    const auto policy=body.value("return_policy",std::string("ready_only"));
    require(policy=="ready_only" || policy=="greedy","unsupported return policy");
    if(policy=="greedy")require(predicted.value("validated",false) || body.value("allow_unvalidated_prediction",false),"unvalidated prediction not explicitly enabled");
    const double cc=predicted.at("cpu_prefill_ms").get<double>()+predicted.value("cpu_restore_ms",0.0),hc=predicted.at("htp_prefill_handoff_ms");
    require(std::isfinite(cc) && std::isfinite(hc) && cc>0 && hc>0,"invalid replay prediction");
    lane.close();anchor={};anchor_ids.clear();
    J cache;double begin=now_ms();const char *dir=std::getenv("A2O_REPLAY_CACHE_DIR");
    auto frozen=replay_cache->get(cpu->ctx,prefix,dir?dir:"",cache);
    double bootstrap=now_ms()-begin;
    J rows=J::array();
    for(const auto &mode:order) {
        lane.close();
        auto *resident=mode=="htp"?htp->ctx:cpu->ctx;
        begin=now_ms();restore_session(resident,frozen);double fixture=now_ms()-begin;
        const uint64_t generation=++version;
        double preparation_ms=0;bool started=false;
        begin=now_ms();
        if(mode=="route") {
            // frozen.state 已由夹具持有；生产服务也在上一响应保存锚点。
            started=lane.start(generation,[&](const auto &valid){
                double start=now_ms();if(!valid())return false;
                restore_session(htp->ctx,frozen);preparation_ms=now_ms()-start;return valid();
            });
            require(started,"always-prepare failed to start on empty lane");
        }
        std::this_thread::sleep_until(std::chrono::steady_clock::now()+std::chrono::duration<double,std::milli>(gap));
        double actual_gap=now_ms()-begin;
        double returned=now_ms();
        std::string availability=mode=="route"?lane.availability(generation):"not_applicable";
        const auto availability_at_return=availability;
        double wait_ms=0;J decision=nullptr;
        std::string route=mode=="htp"?"htp":"cpu",reason="fixed_backend";
        if(mode=="route") {
            if(policy=="greedy") {
                require(availability=="ready" || availability=="pending","invalid greedy candidate state");
                auto plan=plan_greedy_return(availability=="ready",returned-begin,
                    predicted.at("prepare_ms").get<double>(),cc,hc);
                decision={{"cpu_cost_ms",cc},{"htp_cost_ms",hc},{"elapsed_prepare_ms",returned-begin},
                    {"remaining_prepare_ms",plan.remaining_ms},{"continue_cost_ms",plan.continue_cost_ms},
                    {"continue_prepare",plan.continue_prepare},{"wait_budget_ms",plan.wait_budget_ms}};
                reason=plan.reason;
                if(plan.continue_prepare && availability=="pending") {
                    double start=now_ms();bool ready=lane.wait_ready(generation,plan.wait_budget_ms);
                    wait_ms=now_ms()-start;availability=lane.availability(generation);
                    if(!ready)reason="greedy_wait_timeout";
                }
                if(plan.continue_prepare && availability=="ready") {route="htp";lane.claim(generation);}
                else lane.retire();
            } else {
                if(availability=="ready" && hc<cc){route="htp";reason="ready_and_lower_predicted_cost";lane.claim(generation);}
                else {reason=availability=="ready"?"cpu_lower_predicted_cost":"preparation_"+availability;lane.retire();}
            }
        }
        auto res_before=resource_sample();
        auto *pre=route=="htp"?htp->ctx:cpu->ctx;llama_pos pos=prefix.size();
        // 当前生产服务在返回时恢复上一输入锚点，回滚已经生成的 assistant。
        // ready HC 的恢复已在 G 内完成；CPU 与 Full HTP 的真实恢复仍计入响应。
        double stage=now_ms();
        if(!(mode=="route" && route=="htp"))restore_session(pre,frozen);
        double restore=now_ms()-stage;
        bool collect_prior=kv_reduce_req && mode=="route" && route=="htp" && kv_pol=="prefill_prior" && int(prefix.size()+input.size())>kv_keep;
        stage=now_ms();evaluate_with_prior_tail(pre,input,pos,prior,collect_prior,kv_tail);double prefill=now_ms()-stage;
        std::vector<float> first_logits;
        if(verify_free || (kv_reduce_req && int(pos)>kv_keep)){
            const float *fl=llama_get_logits_ith(pre,-1);require(fl,"missing prefill logits");
            first_logits.assign(fl,fl+llama_vocab_n_tokens(vocab));
        }
        J kv_info=nullptr;double evict_ms=0;size_t handoff_bytes=0;
        if(kv_reduce_req && mode=="route" && route=="htp" && int(pos)>kv_keep){
            J kv_cfg={{"kv_reduce_policy",kv_pol},{"kv_keep_tokens",kv_keep},{"kv_sink_tokens",kv_sink},
                      {"kv_recent_tokens",kv_recent},{"kv_pool_window",kv_pool}};
            kv_info=evict_kv_by_prior(pre,int(pos),kv_cfg,prior,int(input.size()));
            evict_ms=kv_info.at("evict_ms").get<double>();
        }
        stage=now_ms();
        if(mode=="route" && route=="htp"){
            if(!kv_info.is_null())llama_memory_clear(llama_get_memory(cpu->ctx),true);
            handoff_bytes=transfer(pre,cpu->ctx);
            if(!first_logits.empty()){
                auto *dst=llama_get_logits_ith(cpu->ctx,-1);require(dst,"missing dst logits after handoff");
                std::copy(first_logits.begin(),first_logits.end(),dst);
            }
            if(!kv_info.is_null()){kv_info["compacted"]=true;kv_info["handoff_bytes"]=handoff_bytes;}
        }
        double handoff=now_ms()-stage;
        auto *dec=mode=="htp"?htp->ctx:cpu->ctx;
        if(kv_reduce_cpu && kv_reduce_req && mode=="cpu" && int(pos)>kv_keep && kv_pol=="recent"){
            double kv_s=now_ms();
            SessionSnapshot tmp;tmp.next_pos=pos;
            J kv_cfg={{"kv_reduce_policy","recent"},{"kv_keep_tokens",kv_keep},{"kv_sink_tokens",kv_sink},
                      {"kv_recent_tokens",kv_recent},{"kv_pool_window",kv_pool},{"kv_compact",true}};
            kv_info=run_kv_reduction(dec,tmp,kv_cfg,prior,int(input.size()));
            evict_ms=now_ms()-kv_s;
            auto *dst=llama_get_logits_ith(dec,-1);require(dst,"missing dst logits after cpu reduce");
            std::copy(first_logits.begin(),first_logits.end(),dst);
        }
        int decode_n_kv=!kv_info.is_null()?kv_info.at("kept_count").get<int>():int(pos);
        stage=now_ms();
        for(auto tok:generated)evaluate(dec,{tok},pos);
        double decode=now_ms()-stage;double response_ms=now_ms()-returned;
        auto res_after=resource_sample();
        J free_gen=nullptr;
        if(verify_free){
            int n_voc=llama_vocab_n_tokens(vocab);
            llama_pos rollback=llama_pos(prefix.size()+input.size());
            require(llama_memory_seq_rm(llama_get_memory(dec),0,rollback,-1),"free-gen rollback seq_rm failed");
            pos=rollback;
            auto *dst=llama_get_logits_ith(dec,-1);require(dst,"missing dst logits for free gen");
            std::copy(first_logits.begin(),first_logits.end(),dst);
            std::vector<llama_token> f_ids;std::string f_text;bool hit_eog=false;
            llama_token top1=-1;
            for(int i=0;i<free_tokens;++i){
                auto tok=greedy(dec,n_voc);
                if(i==0)top1=tok;
                if(llama_vocab_is_eog(vocab,tok)){hit_eog=true;break;}
                f_ids.push_back(tok);f_text+=piece(vocab,tok);evaluate(dec,{tok},pos);
            }
            free_gen={{"tokens",f_ids},{"text",f_text},{"top1_first",top1},{"eog",hit_eog}};
        }
        // 单边界夹具退出必须等后台线程结束；尾部单列，不能悄悄抹去准备工作。
        stage=now_ms();lane.close();double tail=now_ms()-stage;
        rows.push_back({{"condition",mode},{"route",route},{"route_reason",reason},{"availability",availability},
            {"availability_at_return",availability_at_return},{"return_policy",policy},{"wait_ms",wait_ms},{"decision",decision},
            {"H",prefix.size()},{"P",input.size()},{"D",generated.size()},{"decode_n_kv",decode_n_kv},{"always_prepare_started",started},
            {"fixture_restore_ms",fixture},{"requested_gap_ms",gap},{"actual_gap_ms",actual_gap},
            {"prepare_ms",preparation_ms},{"restore_ms",restore},{"prefill_ms",prefill},{"evict_ms",evict_ms},
            {"handoff_ms",handoff},{"handoff_bytes",handoff_bytes},{"decode_ms",decode},{"post_return_ms",response_ms},
            {"retirement_tail_ms",tail},{"gap_to_response_ms",actual_gap+response_ms},{"serialized_total_ms",actual_gap+response_ms+tail},
            {"kv_reduce",kv_info},{"resources_before",res_before},{"resources_after",res_after},{"free_generation",free_gen}});
    }
    return {{"mode",policy=="greedy"?"always_prepare_gap_replay_v2":"always_prepare_gap_replay_v1"},{"bootstrap_ms",bootstrap},{"cache",cache},{"results",rows},
        {"prediction",predicted},{"scope","teacher-forced replay with observed external gap duration; asynchronous prepare is real, external work is a timed wait; no task-quality or live E2E claim"}};
}

// 冻结 token + 实际观测 G 的异步准备重放；不执行生成工具，也不测任务质量。
J gap_replay(const J &body) {
    require(cpu && htp,"gap replay requires explicit CPU and HTP");
    auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
    auto input=body.at("input_ids").get<std::vector<llama_token>>();
    auto generated=body.at("generated_ids").get<std::vector<llama_token>>();
    auto order=body.at("order").get<std::vector<std::string>>();
    auto sorted=order;std::sort(sorted.begin(),sorted.end());
    require(sorted==std::vector<std::string>({"cpu","htp","route"}),"invalid gap replay order");
    require(!prefix.empty() && !input.empty() && !generated.empty() &&
        prefix.size()+input.size()+generated.size()<size_t(context_size),"invalid gap replay shape");
    double gap=body.at("gap_ms").get<double>();
    require(std::isfinite(gap) && gap>=0 && gap<120000,"invalid observed gap");
    const auto predicted=body.at("prediction");
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
        std::string route=mode=="htp"?"htp":"cpu",reason="fixed_backend";
        if(mode=="route") {
            if(availability=="ready" && hc<cc){route="htp";reason="ready_and_lower_predicted_cost";lane.claim(generation);}
            else {reason=availability=="ready"?"cpu_lower_predicted_cost":"preparation_"+availability;lane.retire();}
        }
        auto *pre=route=="htp"?htp->ctx:cpu->ctx;llama_pos pos=prefix.size();
        // 当前生产服务在返回时恢复上一输入锚点，回滚已经生成的 assistant。
        // ready HC 的恢复已在 G 内完成；CPU 与 Full HTP 的真实恢复仍计入响应。
        double stage=now_ms();
        if(!(mode=="route" && route=="htp"))restore_session(pre,frozen);
        double restore=now_ms()-stage;
        stage=now_ms();evaluate(pre,input,pos);double prefill=now_ms()-stage;
        stage=now_ms();if(mode=="route" && route=="htp")transfer(htp->ctx,cpu->ctx);double handoff=now_ms()-stage;
        auto *dec=mode=="htp"?htp->ctx:cpu->ctx;stage=now_ms();
        for(auto tok:generated)evaluate(dec,{tok},pos);
        double decode=now_ms()-stage;double response_ms=now_ms()-returned;
        // 单边界夹具退出必须等后台线程结束；尾部单列，不能悄悄抹去准备工作。
        stage=now_ms();lane.close();double tail=now_ms()-stage;
        rows.push_back({{"condition",mode},{"route",route},{"route_reason",reason},{"availability",availability},
            {"H",prefix.size()},{"P",input.size()},{"D",generated.size()},{"always_prepare_started",started},
            {"fixture_restore_ms",fixture},{"requested_gap_ms",gap},{"actual_gap_ms",actual_gap},
            {"prepare_ms",preparation_ms},{"restore_ms",restore},{"prefill_ms",prefill},{"handoff_ms",handoff},
            {"decode_ms",decode},{"post_return_ms",response_ms},{"retirement_tail_ms",tail},
            {"gap_to_response_ms",actual_gap+response_ms},{"serialized_total_ms",actual_gap+response_ms+tail}});
    }
    return {{"mode","always_prepare_gap_replay_v1"},{"bootstrap_ms",bootstrap},{"cache",cache},{"results",rows},
        {"prediction",predicted},{"scope","teacher-forced replay with observed external gap duration; asynchronous prepare is real, external work is a timed wait; no task-quality or live E2E claim"}};
}

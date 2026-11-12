// 独立状态成本：快照导出、always-prepare 的恢复、真实 HTP→CPU handoff。
J state_cost_sweep(const J &body) {
    require(cpu && htp,"state cost sweep requires CPU and HTP");
    auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
    auto input=body.at("input_ids").get<std::vector<llama_token>>();
    require(!prefix.empty() && !input.empty() && prefix.size()+input.size()<size_t(context_size),"invalid state sweep shape");
    const int repetitions=body.value("repetitions",5);
    require(repetitions>0 && repetitions<=20,"invalid repetitions");
    lane.close();anchor={};anchor_ids.clear();
    J cache;double begin=now_ms();const char *dir=std::getenv("A2O_REPLAY_CACHE_DIR");
    auto frozen=replay_cache->get(cpu->ctx,prefix,dir?dir:"",cache);
    double bootstrap=now_ms()-begin;
    J rows=J::array();
    for(int r=-1;r<repetitions;++r) {
        restore_session(cpu->ctx,frozen);
        begin=now_ms();auto snapshot=export_seq(cpu->ctx);double snapshot_ms=now_ms()-begin;
        require(snapshot==frozen.state,"state sweep snapshot mismatch");
        double import_work=0;
        begin=now_ms();restore_session(htp->ctx,frozen,&import_work);double preparation=now_ms()-begin;
        llama_pos pos=prefix.size();begin=now_ms();evaluate(htp->ctx,input,pos);double prefill=now_ms()-begin;
        begin=now_ms();auto bytes=transfer(htp->ctx,cpu->ctx);double handoff=now_ms()-begin;
        require(bytes>0,"empty handoff state");
        begin=now_ms();restore_session(cpu->ctx,frozen);double cpu_restore=now_ms()-begin;
        rows.push_back({{"repetition",r},{"warmup",r<0},{"H",prefix.size()},{"P",input.size()},
            {"snapshot_bytes",snapshot.size()},{"handoff_bytes",bytes},{"snapshot_ms",snapshot_ms},
            {"prepare_ms",preparation},{"prepare_import_work_ms",import_work},
            {"prepare_verification_ms",preparation-import_work},{"htp_prefill_ms",prefill},
            {"htp_to_cpu_ms",handoff},{"cpu_restore_ms",cpu_restore},{"state_verified",true}});
    }
    return {{"mode","state_cost_sweep_v1"},{"bootstrap_ms",bootstrap},{"cache",cache},{"rows",rows},
        {"scope","production snapshot/restore/transfer helpers, including their byte verification; first repetition is warmup; fixture setup excluded"}};
}

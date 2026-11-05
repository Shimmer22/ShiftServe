#pragma once
// 短历史机制验收：可选单次实测成本参考，不做性能校准；同一对上下文，不运行生成动作。
static int run_execution_probe(const std::map<std::string,std::string> &args) {
    auto get=[&](const std::string &key){return args.at(key);};
    require(get("--mode")=="hybrid" && get("--policy")=="none" && get("--pre-move")=="0" && batch_size==64,"execution probe requires explicit hybrid/none/batch64");
    auto begin=now_ms();Engine cpu(get("--model"),false,false),htp(get("--model"),true,false);
    double initialize_ms=now_ms()-begin;auto vocab=llama_model_get_vocab(cpu.model);int nv=llama_vocab_n_tokens(vocab);
    ExecutionPreparation plan(htp.ctx,true);ToolEvents events(get("--out")+".events.jsonl");
    auto prefix=tokens(vocab,"<|im_start|>system\nYou are a GUI agent. For every new request return only {\"action\":\"wait\"}. No markdown.<|im_end|>\n<|im_start|>user\nPast observations:\n",true);
    auto closing=tokens(vocab,"\nDone.<|im_end|>\n<|im_start|>assistant\n{\"action\":\"wait\"}<|im_end|>",false);
    const int h=256;auto fill=tokens(vocab," note note note note note note note note",false);
    while(prefix.size()+closing.size()<size_t(h)) prefix.push_back(fill[(prefix.size())%fill.size()]);
    prefix.insert(prefix.end(),closing.begin(),closing.end());require(prefix.size()==h,"probe history mismatch");
    llama_pos pos=0;begin=now_ms();evaluate(htp.ctx,prefix,pos);double history_ms=now_ms()-begin;
    SessionSnapshot base;base.state=export_seq(htp.ctx);base.next_pos=h;base.evaluated_tokens=h;base.version=7;
    auto input=[&](int p) {
        std::string padding;
        for(int n=0;n<=p;n++) {
            auto text="\n<|im_start|>user\n"+padding+"\nWait.<|im_end|>\n<|im_start|>assistant\n";
            auto result=tokens(vocab,text,false);if(result.size()==size_t(p))return result;padding+=" note";
        }
        throw std::runtime_error("probe input fixture unavailable");
    };
    struct Result {std::vector<float> logits;std::vector<uint8_t> state,final_state;std::vector<llama_token> ids;std::string text;};
    std::map<std::pair<bool,int>,Result> reference;
    std::ofstream output(get("--out"));require(bool(output),"probe output unavailable");
    events.emit("session_ready",-1,"\"model_context_init_ms\":"+std::to_string(initialize_ms)+",\"history_build_ms\":"+std::to_string(history_ms));
    const bool calibrate=args.count("--self-calibration-probe") && args.at("--self-calibration-probe")=="1";
    OnlineCostConfig online_config;online_config.enabled=calibrate;
    OnlineRouteCosts online(online_config);CostRouter router;router.source="mechanism-fixture";
    bool online_checked=false;double ema=0;
    struct Case {const char *name;int actual,predicted;bool stale,cancel;uint64_t prefault;bool asynchronous=false,cpu=false;};
    std::vector<Case> cases={
        {"ordinary_64",64,0,false,false,0},{"ordinary_32",32,0,false,false,0},
        {"hit_64",64,64,false,false,0},{"hit_32",32,32,false,false,0},
        {"shape_miss",32,64,false,false,0},{"version_miss",64,64,true,false,0},
        {"discard_before_bind",64,64,false,true,0},{"prefault_hit",64,64,false,false,1024*1024},
        {"repeat_hit",64,64,false,false,0},
        {"async_hit",64,64,false,false,0,true},{"async_cancel",64,64,false,true,0,true}};
    if(calibrate) {
        cases.insert(cases.begin(),{"ordinary_cpu_64",64,0,false,false,0,false,true});
        cases.insert(cases.begin()+3,{{"online_hc_repeat_1",64,0,false,false,0},
                                     {"online_hc_repeat_2",64,0,false,false,0}});
    }
    int id=0;
    for(const auto &c:cases) {
        plan.discard();restore_session(htp.ctx,base);restore_session(cpu.ctx,base);auto suffix=input(c.actual);
        events.emit("execution_case_started",id,"\"name\":"+json(c.name));
        a2o_execution_stats prepared{},bound{},consumed{};bool unchanged=true;
        double prepare_ms=0;
        if(c.predicted) {
            begin=now_ms();
            if(c.asynchronous) {
                PreparationBarrier barrier;PreparationLane lane;
                try {
                    require(lane.start(base.version,[&](const auto &valid) {
                        try {
                            if(!plan.prepare_cancellable(valid,base.version,h,c.predicted,c.prefault,&prepared)) return false;
                            if(c.cancel) barrier.enter();return valid();
                        } catch(...) {barrier.fail(std::current_exception());throw;}
                    }),"probe lane was busy");
                    if(c.cancel) {barrier.wait_entered();lane.retire();barrier.release();lane.close();}
                    else {lane.wait_for_test();lane.claim(base.version);}
                } catch(...) {barrier.release();lane.close();throw;}
            } else prepared=plan.prepare(base.version,h,c.predicted,c.prefault);
            prepare_ms=now_ms()-begin;
            unchanged=export_seq(htp.ctx)==base.state && export_seq(cpu.ctx)==base.state;require(unchanged,"prepare mutated KV state");
            events.emit("execution_prepared",id,"\"name\":"+json(c.name)+",\"graph_nodes\":"+std::to_string(prepared.graph_nodes)
                +",\"descriptor_ops\":"+std::to_string(prepared.descriptor_ops)+",\"mapped_bytes\":"+std::to_string(prepared.mapped_bytes));
            if(c.cancel) plan.discard();
            else bound=plan.bind(base.version+(c.stale?1:0),h,c.actual);
        }
        auto pre=c.cpu?cpu.ctx:htp.ctx;
        pos=h;begin=now_ms();evaluate(pre,suffix,pos);double prefill_ms=now_ms()-begin;
        consumed=plan.query();
        events.emit("execution_consumed",id,"\"name\":"+json(c.name)+",\"status\":"+std::to_string(consumed.status)
            +",\"reason\":"+json(consumed.reason)+",\"graph_hits\":"+std::to_string(consumed.graph_hits)
            +",\"descriptor_hits\":"+std::to_string(consumed.descriptor_hits)+",\"descriptor_misses\":"+std::to_string(consumed.descriptor_misses));
        Result current;auto ptr=llama_get_logits_ith(pre,-1);current.logits.assign(ptr,ptr+nv);current.state=export_seq(pre);
        auto first=greedy(pre,nv);size_t bytes=0;double handoff_ms=0,handoff_work_ms=0;
        if(!c.cpu) {
            begin=now_ms();SessionSnapshot updated;updated.state=export_seq(pre);
            double exported=now_ms()-begin,imported=0;
            bytes=restore_session(cpu.ctx,updated,&imported);handoff_work_ms=exported+imported;handoff_ms=now_ms()-begin;
        }
        bool eog=false;begin=now_ms();
        DecodeHorizon horizon(calibrate,4,begin);
        for(int n=0;n<32;n++) {auto token=n?greedy(cpu.ctx,nv):first;current.ids.push_back(token);evaluate(cpu.ctx,{token},pos);horizon.evaluated(n+1,now_ms);
            if(llama_vocab_is_eog(vocab,token)) {eog=true;break;}current.text+=piece(vocab,token);}
        double decode_ms=now_ms()-begin;require(eog,"probe response did not reach EOG");current.final_state=export_seq(cpu.ctx);
        bool same=true;
        auto reference_key=std::make_pair(c.cpu,c.actual);
        if(!reference.count(reference_key)) reference[reference_key]=current;
        else {const auto &r=reference.at(reference_key);same=current.logits==r.logits && current.state==r.state && current.final_state==r.final_state && current.ids==r.ids && current.text==r.text;}
        require(same,"prepared result differs from ordinary HTP");
        bool expect_hit=c.predicted && !c.cancel && !c.stale && c.actual==c.predicted;
        bool mechanism=expect_hit?(bound.status==A2O_EXEC_BOUND && consumed.graph_hits>0 && consumed.descriptor_hits==prepared.descriptor_ops):consumed.graph_hits==0 && consumed.descriptor_hits==0;
        require(mechanism,"execution plan was not consumed as expected");
        if(c.predicted)require(prepared.descriptor_ops>0 && prepared.mapped_buffers>0,"preparation counters missing");
        if(c.prefault)require(prepared.prefault_bytes>0 && prepared.prefault_bytes<=c.prefault,"prefault budget invalid");
        std::ostringstream row;row<<"{\"id\":"<<id<<",\"name\":"<<json(c.name)<<",\"P\":"<<c.actual<<",\"passed\":true,\"eog\":true,\"state_unchanged\":"<<(unchanged?"true":"false")
            <<",\"exact_logits_and_state\":"<<(same?"true":"false")<<",\"output\":"<<json(current.text)<<",\"handoff_bytes\":"<<bytes
            <<",\"graph_hits\":"<<consumed.graph_hits<<",\"descriptor_hits\":"<<consumed.descriptor_hits<<",\"descriptor_ops\":"<<prepared.descriptor_ops
            <<",\"mapped_buffers\":"<<prepared.mapped_buffers<<",\"mapped_bytes\":"<<prepared.mapped_bytes<<",\"prefault_bytes\":"<<prepared.prefault_bytes
            <<",\"prefill_backend\":"<<json(c.cpu?"cpu":"htp")
            <<",\"prepare_ms\":"<<prepare_ms<<",\"prefill_ms\":"<<prefill_ms<<",\"handoff_ms\":"<<handoff_ms
            <<",\"handoff_work_ms\":"<<handoff_work_ms<<",\"decode_prefix_ms\":"<<optional_number(horizon.prefix_ms())<<",\"decode_ms\":"<<decode_ms<<"}";
        output<<row.str()<<'\n';output.flush();events.emit("execution_case_completed",id,"\"result\":"+row.str());id++;
        if(calibrate && !c.predicted && c.actual==64) {
            require(std::isfinite(horizon.prefix_ms()),"response shorter than calibration D");
            double future=prefill_ms+handoff_work_ms+horizon.prefix_ms();
            if(c.cpu) router.rows.push_back({h,64,4,1,"CC",future,0});
            else {
                if(!router.find(h,64,4,"HC")) router.rows.push_back({h,64,4,1,"HC",future,0});
                auto before=router.choose(h,64,4,"ready",&online,now_ms());
                require(before.cpu_source=="offline","unobserved CC was overwritten");
                OnlineCostKey key{router.source,"ordinary","HC",h,64,4};
                require(online.sample(key,now_ms()),"cold bucket was not sampled");
                auto count=online.samples(key,now_ms());ema=count?ema+.2*(future-ema):future;
                online.update(key,future,now_ms());
                auto after=router.choose(h,64,4,"ready",&online,now_ms());
                require(after.htp_source==(count<2?"offline":"online_corrected"),"online readiness mismatch");
                if(count==2) {
                    require(std::abs(after.htp_ms-ema)<1e-6 && after.cpu_source=="offline","EMA or candidate isolation failed");
                    require(router.choose(h,64,4,"pending",&online,now_ms()).backend=="cpu","pending candidate was selected");
                    require(!router.choose(h,65,4,"ready",&online,now_ms()).cost_covered,"unmeasured shape extrapolated");
                    online_checked=true;
                }
                events.emit("self_calibration_observed",id-1,"\"samples\":"+std::to_string(count+1)
                    +",\"cpu_source\":"+json(after.cpu_source)+",\"htp_source\":"+json(after.htp_source)
                    +",\"cpu_ms\":"+optional_number(after.cpu_ms)+",\"htp_ms\":"+optional_number(after.htp_ms)
                    +",\"observed_future_ms\":"+optional_number(future));
            }
        }
    }
    require(!calibrate || online_checked,"self calibration mechanism incomplete");
    plan.discard();events.emit("session_closed",-1,"\"cases\":"+std::to_string(id));return 0;
}

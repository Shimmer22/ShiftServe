#pragma once
// 短历史机制验收：无成本校准，不运行生成动作；所有对照沿用同一对 CPU/HTP 上下文。
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
    std::map<int,Result> reference;
    std::ofstream output(get("--out"));require(bool(output),"probe output unavailable");
    events.emit("session_ready",-1,"\"model_context_init_ms\":"+std::to_string(initialize_ms)+",\"history_build_ms\":"+std::to_string(history_ms));
    struct Case {const char *name;int actual,predicted;bool stale,cancel;uint64_t prefault;bool asynchronous=false;};
    const Case cases[]={
        {"ordinary_64",64,0,false,false,0},{"ordinary_32",32,0,false,false,0},
        {"hit_64",64,64,false,false,0},{"hit_32",32,32,false,false,0},
        {"shape_miss",32,64,false,false,0},{"version_miss",64,64,true,false,0},
        {"discard_before_bind",64,64,false,true,0},{"prefault_hit",64,64,false,false,1024*1024},
        {"repeat_hit",64,64,false,false,0},
        {"async_hit",64,64,false,false,0,true},{"async_cancel",64,64,false,true,0,true}};
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
                            prepared=plan.prepare(base.version,h,c.predicted,c.prefault);
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
        pos=h;begin=now_ms();evaluate(htp.ctx,suffix,pos);double prefill_ms=now_ms()-begin;
        consumed=plan.query();
        events.emit("execution_consumed",id,"\"name\":"+json(c.name)+",\"status\":"+std::to_string(consumed.status)
            +",\"reason\":"+json(consumed.reason)+",\"graph_hits\":"+std::to_string(consumed.graph_hits)
            +",\"descriptor_hits\":"+std::to_string(consumed.descriptor_hits)+",\"descriptor_misses\":"+std::to_string(consumed.descriptor_misses));
        Result current;auto ptr=llama_get_logits_ith(htp.ctx,-1);current.logits.assign(ptr,ptr+nv);current.state=export_seq(htp.ctx);
        auto first=greedy(htp.ctx,nv);begin=now_ms();size_t bytes=transfer(htp.ctx,cpu.ctx);double handoff_ms=now_ms()-begin;
        bool eog=false;begin=now_ms();
        for(int n=0;n<32;n++) {auto token=n?greedy(cpu.ctx,nv):first;current.ids.push_back(token);evaluate(cpu.ctx,{token},pos);
            if(llama_vocab_is_eog(vocab,token)) {eog=true;break;}current.text+=piece(vocab,token);}
        double decode_ms=now_ms()-begin;require(eog,"probe response did not reach EOG");current.final_state=export_seq(cpu.ctx);
        bool same=true;
        if(!reference.count(c.actual)) reference[c.actual]=current;
        else {const auto &r=reference.at(c.actual);same=current.logits==r.logits && current.state==r.state && current.final_state==r.final_state && current.ids==r.ids && current.text==r.text;}
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
            <<",\"prepare_ms\":"<<prepare_ms<<",\"prefill_ms\":"<<prefill_ms<<",\"handoff_ms\":"<<handoff_ms<<",\"decode_ms\":"<<decode_ms<<"}";
        output<<row.str()<<'\n';output.flush();events.emit("execution_case_completed",id,"\"result\":"+row.str());id++;
    }
    plan.discard();events.emit("session_closed",-1,"\"cases\":"+std::to_string(id));return 0;
}

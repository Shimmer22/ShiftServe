#pragma once
#include "preparation_lane.h"
#include "router.h"

// 协议输出与原始 JSONL 同时发布；worker 与主线程共用有锁的事件出口。
class ToolEvents {
public:
    explicit ToolEvents(const std::string & path): file_(path) { require(bool(file_),"tool event file unavailable"); }
    void emit(const std::string & name, int turn, const std::string & fields="") {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream row;
        row << "{\"event\":" << json(name) << ",\"turn\":" << turn << ",\"monotonic_ms\":"
            << std::fixed << now_ms() << (fields.empty()?"":","+fields) << "}";
        file_ << row.str() << '\n'; file_.flush(); require(bool(file_),"tool event write failed");
        std::cout << "A2O\t" << row.str() << '\n'; std::cout.flush();
        require(bool(std::cout),"tool protocol output failed");
    }
private:
    std::mutex mutex_;
    std::ofstream file_;
};

static std::string optional_number(double value) {
    if(!std::isfinite(value)) return "null";
    std::ostringstream out; out << value; return out.str();
}

// 文本 Tool 的持久进程：START -> OBS -> RUN；Tool 由 host 白名单适配器实际执行。
static int run_tool_session(const std::map<std::string,std::string> & args) {
    auto get=[&](const std::string & k,const std::string & d="") {auto it=args.find(k);return it==args.end()?d:it->second;};
    require(get("--mode")=="hybrid" && get("--policy")=="none" && get("--pre-move")=="0",
            "tool session requires explicit hybrid without compression/premove");
    const bool mechanism=get("--mechanism-test","0")=="1";
    int d=std::stoi(get("--router-d","4")), max_tokens=std::stoi(get("--max-tokens","64"));
    require(d>0 && max_tokens>0,"invalid decode budget");
    auto router=CostRouter::load(get("--router-table"));
    require(router.context==context_size && router.batch==batch_size,"router context/batch mismatch");
    require(router.queue==std::stoi(get("--htp-queue")) && router.opbatch==std::stoi(get("--htp-opbatch")),
            "router HTP configuration mismatch");
    double init_start=now_ms();
    Engine cpu(get("--model"),false,false), htp(get("--model"),true,false);
    auto vocab=llama_model_get_vocab(cpu.model); int nv=llama_vocab_n_tokens(vocab);
    auto events=std::make_shared<ToolEvents>(get("--out")+".events.jsonl");
    PreparationLane lane; // 必须先于 Engine 析构。
    std::vector<std::shared_ptr<PreparationBarrier>> barriers;
    struct ReleaseOnExit {
        std::vector<std::shared_ptr<PreparationBarrier>> & barriers;
        ~ReleaseOnExit() { for(auto & barrier:barriers) barrier->release(); }
    } release_on_exit{barriers}; // 异常时先释放测试屏障，再让 lane 安全关闭。
    SessionSnapshot snapshot;
    std::vector<std::pair<std::string,std::string>> conversation={{"system",read(get("--system"))}};
    std::vector<std::string> results;
    std::string line, full, suffix, mode;
    std::vector<llama_token> input;
    int round=0; bool started=false, observed=false;
    double observation_processing_ms=0;
    auto poll=[&] {
        auto old=lane.reaped(); lane.poll();
        if(lane.reaped()!=old) events->emit("lane_reaped",round,"\"reaped\":"+std::to_string(lane.reaped()));
    };
    events->emit("session_ready",-1,"\"model_context_init_ms\":"+std::to_string(now_ms()-init_start));
    while(std::getline(std::cin,line)) {
        std::vector<std::string> fields; std::istringstream row(line); std::string field;
        while(std::getline(row,field,'\t')) fields.push_back(field);
        require(!fields.empty(),"empty tool command");
        poll();
        if(fields[0]=="CLOSE") {
            require(fields.size()==1 && !started,"cannot close unfinished tool turn");
            events->emit("shutdown_started",round);
            for(auto & barrier:barriers) barrier->release();
            lane.close();
            std::ofstream out(get("--out")); require(bool(out),"tool summary unavailable");
            out << "{\"session\":true,\"tool_session\":true,\"timing_valid_for_performance\":false,\"turns\":[";
            for(size_t i=0;i<results.size();++i) out << (i?",":"") << results[i];
            out << "],\"image_encode_calls\":0,\"started_tasks\":" << lane.started()
                << ",\"reaped_tasks\":" << lane.reaped() << ",\"max_background_tasks\":1}";
            out.close(); require(bool(out),"tool summary write failed");
            events->emit("session_closed",round,"\"started_tasks\":"+std::to_string(lane.started())
                         +",\"reaped_tasks\":"+std::to_string(lane.reaped()));
            return 0;
        }
        if(fields[0]=="RELEASE") {
            require(mechanism && fields.size()==1,"RELEASE is mechanism-test only");
            for(auto & barrier:barriers) barrier->release();
            barriers.clear();
            events->emit("barrier_released",round); continue;
        }
        if(fields[0]=="DRAIN") {
            require(mechanism && fields.size()==1 && !started,"DRAIN is mechanism-test cleanup only");
            lane.close(); events->emit("test_cleanup_drained",round); continue;
        }
        require(fields.size()>=2 && std::stoi(fields[1])==round,"tool command turn mismatch");
        if(fields[0]=="START") {
            require(fields.size()==3 && !started,"invalid START transition");
            mode=fields[2];
            require(mode=="auto" || mode=="none" || mode=="always" || (mechanism && mode=="hold"),"invalid preparation mode");
            bool want=!snapshot.state.empty() && mode!="none"
                && (mode!="auto" || router.should_prepare(snapshot.evaluated_tokens,d));
            bool launch=false; std::string skip;
            if(want && !lane.occupied()) {
                auto immutable=std::make_shared<const SessionSnapshot>(snapshot);
                auto barrier=mode=="hold"?std::make_shared<PreparationBarrier>():nullptr;
                if(barrier) barriers.push_back(barrier);
                const int task_turn=round;
                launch=lane.start(snapshot.version,[immutable,barrier,events,task_turn,&htp](const auto & valid) {
                    try {
                    if(!valid()) return false;
                    events->emit("prepare_started",task_turn);
                    double prepare_begin=now_ms();
                    llama_synchronize(htp.ctx);
                    if(!valid()) return false;
                    llama_memory_clear(llama_get_memory(htp.ctx),true);
                    if(!valid()) return false;
                    require(llama_state_seq_set_data(htp.ctx,immutable->state.data(),immutable->state.size(),0)==immutable->state.size(),
                            "tool preparation import failed");
                    if(!valid()) return false;
                    require(export_seq(htp.ctx)==immutable->state,"tool preparation state/position mismatch");
                    events->emit("prepare_import_verified",task_turn,"\"bytes\":"+std::to_string(immutable->state.size())
                                 +",\"prepare_checked_ms\":"+std::to_string(now_ms()-prepare_begin));
                    if(barrier) { events->emit("barrier_entered",task_turn); barrier->enter(); }
                    bool complete=valid();
                    events->emit(complete?"prepare_work_ready":"prepare_work_cancelled",task_turn);
                    return complete;
                    } catch(...) { if(barrier) barrier->fail(std::current_exception()); throw; }
                });
                if(barrier) barrier->wait_entered();
            }
            skip=launch?"started":lane.occupied()?"lane_busy":snapshot.state.empty()?"no_history":mode=="none"?"disabled":"no_supported_benefit";
            started=true; observed=false;
            events->emit("tool_started",round,"\"preparation\":"+json(skip)+",\"version\":"+std::to_string(snapshot.version)
                         +",\"H\":"+std::to_string(snapshot.evaluated_tokens)+",\"D\":"+std::to_string(d));
        } else if(fields[0]=="WAIT") {
            require(mechanism && fields.size()==2 && started && !observed,"WAIT is mechanism-test only before OBS");
            lane.wait_for_test(); events->emit("candidate_waited",round);
        } else if(fields[0]=="OBS") {
            require(fields.size()==3 && started && !observed,"invalid OBS transition");
            double observation_begin=now_ms();
            // 只有真实 Tool 返回后，host 才创建并传入此文件。
            conversation.push_back({"user",read(fields[2])});
            std::vector<llama_chat_message> messages;
            for(const auto & m:conversation) messages.push_back({m.first.c_str(),m.second.c_str()});
            full=format(cpu.model,messages,true);
            require(full.compare(0,snapshot.rendered_prefix.size(),snapshot.rendered_prefix)==0,"tool session prefix mismatch");
            suffix=full.substr(snapshot.rendered_prefix.size()); input=tokens(vocab,suffix,round==0);
            require(snapshot.evaluated_tokens+int(input.size())+max_tokens<=context_size,"tool session context capacity exceeded");
            std::ofstream(get("--out")+".turn-"+std::to_string(round)+".prompt.txt") << full;
            observed=true;
            observation_processing_ms=now_ms()-observation_begin;
            events->emit("observation_ready",round,"\"H\":"+std::to_string(snapshot.evaluated_tokens)
                         +",\"P\":"+std::to_string(input.size())+",\"D\":"+std::to_string(d));
        } else if(fields[0]=="RUN") {
            require(started && observed && (fields.size()==2 || (mechanism && fields.size()==3)),"invalid RUN transition");
            auto current_router=fields.size()==3?CostRouter::load(fields[2],true):router;
            require(fields.size()!=3 || current_router.source=="mechanism-fixture","fixture costs must be explicitly labelled");
            require(current_router.context==context_size && current_router.batch==batch_size
                    && current_router.queue==router.queue && current_router.opbatch==router.opbatch,"router fixture configuration mismatch");
            auto availability=lane.availability(snapshot.version);
            auto decision=current_router.choose(snapshot.evaluated_tokens,input.size(),d,availability);
            if(mode=="none") {decision.backend="cpu";decision.reason="fixed_cpu";}
            bool authority=snapshot.state.empty() || export_seq(cpu.ctx)==snapshot.state;
            require(authority,"background task changed CPU authority");
            events->emit("route_selected",round,"\"backend\":"+json(decision.backend)+",\"reason\":"+json(decision.reason)
                         +",\"availability\":"+json(availability)+",\"source\":"+json(current_router.source)
                         +",\"progress\":"+json(lane.progress())+",\"candidate_version\":"+std::to_string(lane.version())
                         +",\"cost_covered\":"+(decision.cost_covered?"true":"false")
                         +",\"cpu_ms\":"+optional_number(decision.cpu_ms)+",\"htp_ms\":"+optional_number(decision.htp_ms));
            llama_context * pre=cpu.ctx;
            double restore_begin=now_ms();
            if(decision.backend=="htp") { lane.claim(snapshot.version); pre=htp.ctx; }
            else { lane.retire(); restore_session(cpu.ctx,snapshot); }
            double restore_checked_ms=now_ms()-restore_begin;
            events->emit("prefill_started",round,"\"backend\":"+json(decision.backend));
            auto pos=snapshot.next_pos; double begin=now_ms(); evaluate(pre,input,pos); double prefill_ms=now_ms()-begin;
            auto first=greedy(pre,nv); size_t handoff_bytes=0;
            double handoff_ms=0;
            if(pre!=cpu.ctx) {
                begin=now_ms();SessionSnapshot updated;updated.state=export_seq(pre);handoff_bytes=restore_session(cpu.ctx,updated);
                handoff_ms=now_ms()-begin;
            }
            std::vector<llama_token> ids; std::string output,closing; bool eog=false;
            begin=now_ms();
            for(int i=0;i<max_tokens;++i) {
                auto tok=i==0?first:greedy(cpu.ctx,nv);
                if(llama_vocab_is_eog(vocab,tok)) events->emit("action_complete",round);
                evaluate(cpu.ctx,{tok},pos);
                if(llama_vocab_is_eog(vocab,tok)) {eog=true;closing=piece(vocab,tok);break;}
                ids.push_back(tok);output+=piece(vocab,tok);
            }
            double decode_ms=now_ms()-begin;
            require(eog && !output.empty(),"tool response incomplete/empty");
            auto old_version=snapshot.version;
            auto old_tokens=snapshot.evaluated_tokens;
            snapshot.version++; snapshot.next_pos=pos;
            snapshot.evaluated_tokens+=input.size()+ids.size()+1;
            snapshot.rendered_prefix=full+output+closing; snapshot.state=export_seq(cpu.ctx);
            conversation.push_back({"assistant",output});
            std::ofstream(get("--out")+".turn-"+std::to_string(round)+".prefix.txt") << snapshot.rendered_prefix;
            lane.retire(); // 状态推进后旧候选只会退役，绝不隐式提交。
            events->emit("cpu_state_committed",round,"\"version\":"+std::to_string(snapshot.version));
            std::ostringstream response;
            response << "{\"turn\":" << round << ",\"prefill_backend\":" << json(decision.backend)
                << ",\"decode_backend\":\"cpu\",\"route_reason\":" << json(decision.reason)
                << ",\"availability\":" << json(availability) << ",\"cost_source\":" << json(current_router.source)
                << ",\"cpu_cost_ms\":" << optional_number(decision.cpu_ms) << ",\"htp_cost_ms\":" << optional_number(decision.htp_ms)
                << ",\"cost_covered\":" << (decision.cost_covered?"true":"false")
                << ",\"cached_tokens\":" << old_tokens << ",\"new_tokens\":" << input.size() << ",\"router_D\":" << d
                << ",\"snapshot_version\":" << old_version << ",\"committed_version\":" << snapshot.version
                << ",\"authority_checked\":true,\"prefix_match\":true,\"restore_checked\":true,\"eog_committed\":true,\"compared\":false"
                << ",\"image_encode_calls\":0,\"handoff_bytes\":" << handoff_bytes << ",\"next_position\":" << pos
                << ",\"evaluated_tokens\":" << snapshot.evaluated_tokens << ",\"output\":" << json(output) << ",\"generated_ids\":[";
            for(size_t i=0;i<ids.size();++i) response << (i?",":"") << ids[i];
            response << "],\"timing_ms\":{\"observation_processing_ms\":" << observation_processing_ms
                     << ",\"restore_checked_ms\":" << restore_checked_ms << ",\"handoff_checked_ms\":" << handoff_ms
                     << ",\"vision_init_ms\":0,\"prefill_ms\":" << prefill_ms << ",\"decode_ms\":" << decode_ms
                     << "},\"background_occupied_at_publish\":" << (lane.occupied()?"true":"false") << "}";
            results.push_back(response.str());
            // 在任何回收/下一轮等待之前真正 flush 到消费者。
            events->emit("response_published",round,"\"result\":"+response.str());
            ++round; started=false; observed=false;
        } else throw std::runtime_error("unknown tool session command");
    }
    throw std::runtime_error("tool protocol EOF without CLOSE");
}

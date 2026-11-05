#pragma once
#include "state/preparation_lane.h"
#include "scheduling/router.h"
#include "scheduling/execution_preparation.h"

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
    const bool prepare_execution=get("--htp-execution-prepare","0")=="1";
    const int predicted_input=std::stoi(get("--htp-predict-input","64"));
    const uint64_t prefault_bytes=std::stoull(get("--htp-prefault-bytes","0"));
    OnlineCostConfig online_config;
    online_config.enabled=get("--self-calibrating","0")=="1";
    online_config.alpha=std::stod(get("--online-alpha","0.2"));
    online_config.min_samples=std::stoul(get("--online-min-samples","3"));
    online_config.every=std::stoul(get("--online-every","8"));
    online_config.capacity=std::stoul(get("--online-capacity","256"));
    online_config.ttl_ms=std::stod(get("--online-ttl-ms","300000"));
    OnlineRouteCosts online(online_config);
    // 首期 native 仅普通执行成本；准备计划命中/失配不混入其 EMA，机制夹具也不训练。
    const OnlineRouteCosts * corrections=!prepare_execution && !mechanism && online.enabled()?&online:nullptr;
    require(!prepare_execution || (predicted_input>0 && predicted_input<=batch_size && predicted_input<=64),"predicted HTP microbatch must be 1..min(batch,64)");
    require(!prepare_execution || prefault_bytes<=64ull*1024*1024,"prefault budget exceeds 64MiB");
    ExecutionPreparation execution(htp.ctx,prepare_execution);
    PreparationLane lane; // 先 join lane，再销毁 execution 和 Engine。
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
        if(lane.reaped()!=old) {
            execution.discard(); // worker 已退出，覆盖最后一个检查点之后到达的取消。
            events->emit("lane_reaped",round,"\"reaped\":"+std::to_string(lane.reaped()));
        }
    };
    events->emit("session_ready",-1,"\"model_context_init_ms\":"+std::to_string(now_ms()-init_start)
                 +",\"self_calibrating\":"+(online.enabled()?"true":"false")
                 +",\"online_scope\":"+json(corrections?"ordinary_exact":"inactive"));
    while(std::getline(std::cin,line)) {
        std::vector<std::string> fields; std::istringstream row(line); std::string field;
        while(std::getline(row,field,'\t')) fields.push_back(field);
        require(!fields.empty(),"empty tool command");
        poll();
        if(fields[0]=="CLOSE") {
            require(fields.size()==1 && !started,"cannot close unfinished tool turn");
            events->emit("shutdown_started",round);
            for(auto & barrier:barriers) barrier->release();
            lane.close();execution.discard();
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
            lane.close();execution.discard();events->emit("test_cleanup_drained",round); continue;
        }
        require(fields.size()>=2 && std::stoi(fields[1])==round,"tool command turn mismatch");
        if(fields[0]=="START") {
            require(fields.size()==3 && !started,"invalid START transition");
            mode=fields[2];
            require(mode=="auto" || mode=="none" || mode=="always" || (mechanism && mode=="hold"),"invalid preparation mode");
            bool want=!snapshot.state.empty() && mode!="none"
                && (mode!="auto" || router.should_prepare(snapshot.evaluated_tokens,d,corrections,now_ms()));
            bool launch=false; std::string skip;
            if(want && !lane.occupied()) {
                auto immutable=std::make_shared<const SessionSnapshot>(snapshot);
                auto barrier=mode=="hold"?std::make_shared<PreparationBarrier>():nullptr;
                if(barrier) barriers.push_back(barrier);
                const int task_turn=round;
                launch=lane.start(snapshot.version,[immutable,barrier,events,task_turn,&htp,&execution,predicted_input,prefault_bytes](const auto & valid) {
                    try {
                    if(!valid()) return false;
                    events->emit("prepare_started",task_turn);
                    double prepare_begin=now_ms();
                    llama_synchronize(htp.ctx);
                    if(!valid()) return false;
                    execution.discard();
                    llama_memory_clear(llama_get_memory(htp.ctx),true);
                    if(!valid()) return false;
                    require(llama_state_seq_set_data(htp.ctx,immutable->state.data(),immutable->state.size(),0)==immutable->state.size(),
                            "tool preparation import failed");
                    if(!valid()) return false;
                    require(export_seq(htp.ctx)==immutable->state,"tool preparation state/position mismatch");
                    events->emit("prepare_import_verified",task_turn,"\"bytes\":"+std::to_string(immutable->state.size())
                                 +",\"prepare_checked_ms\":"+std::to_string(now_ms()-prepare_begin));
                    if(execution.enabled()) {
                        if(!valid()) return false;
                        a2o_execution_stats plan{};
                        if(!execution.prepare_cancellable(valid,immutable->version,immutable->next_pos,predicted_input,prefault_bytes,&plan))
                            return false;
                        events->emit("execution_plan_prepared",task_turn,
                            "\"version\":"+std::to_string(immutable->version)+",\"predicted_microbatch\":"+std::to_string(predicted_input)
                            +",\"graph_nodes\":"+std::to_string(plan.graph_nodes)+",\"htp_nodes\":"+std::to_string(plan.htp_nodes)
                            +",\"cpu_nodes\":"+std::to_string(plan.cpu_nodes)+",\"descriptor_ops\":"+std::to_string(plan.descriptor_ops)+",\"descriptor_bytes\":"+std::to_string(plan.descriptor_bytes)
                            +",\"mapped_buffers\":"+std::to_string(plan.mapped_buffers)+",\"mapped_bytes\":"+std::to_string(plan.mapped_bytes)
                            +",\"prefault_bytes\":"+std::to_string(plan.prefault_bytes)
                            +",\"graph_ms\":"+std::to_string(plan.graph_ms)+",\"allocate_ms\":"+std::to_string(plan.allocate_ms)
                            +",\"backend_ms\":"+std::to_string(plan.backend_ms));
                    }
                    if(barrier) { events->emit("barrier_entered",task_turn); barrier->enter(); }
                    bool complete=valid();
                    if(!complete && execution.enabled()) execution.discard();
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
            const double cost_now=now_ms();
            auto decision=current_router.choose(snapshot.evaluated_tokens,input.size(),d,availability,corrections,cost_now);
            if(mode=="none") {decision.backend="cpu";decision.reason="fixed_cpu";}
            OnlineCostKey cost_key{current_router.source,"ordinary",decision.backend=="htp"?"HC":"CC",
                                   snapshot.evaluated_tokens,int(input.size()),d};
            bool sample_cost=corrections && mode!="none"
                && current_router.find(cost_key.h,cost_key.p,cost_key.d,cost_key.path)
                && online.sample(cost_key,cost_now);
            bool authority=snapshot.state.empty() || export_seq(cpu.ctx)==snapshot.state;
            require(authority,"background task changed CPU authority");
            events->emit("route_selected",round,"\"backend\":"+json(decision.backend)+",\"reason\":"+json(decision.reason)
                         +",\"availability\":"+json(availability)+",\"source\":"+json(current_router.source)
                         +",\"progress\":"+json(lane.progress())+",\"candidate_version\":"+std::to_string(lane.version())
                         +",\"cost_covered\":"+(decision.cost_covered?"true":"false")
                         +",\"cpu_ms\":"+optional_number(decision.cpu_ms)+",\"htp_ms\":"+optional_number(decision.htp_ms)
                         +",\"cpu_cost_source\":"+json(decision.cpu_source)+",\"htp_cost_source\":"+json(decision.htp_source));
            llama_context * pre=cpu.ctx;
            double restore_begin=now_ms();
            if(decision.backend=="htp") {
                lane.claim(snapshot.version);pre=htp.ctx;
                if(execution.enabled()) {
                    auto plan=execution.bind(snapshot.version,snapshot.next_pos,std::min(size_t(batch_size),input.size()));
                    events->emit("execution_plan_bound",round,"\"status\":"+std::to_string(plan.status)+",\"reason\":"+json(plan.reason)
                        +",\"bind_ms\":"+std::to_string(plan.bind_ms));
                }
            }
            else { lane.retire(); restore_session(cpu.ctx,snapshot); }
            double restore_checked_ms=now_ms()-restore_begin;
            events->emit("prefill_started",round,"\"backend\":"+json(decision.backend));
            auto pos=snapshot.next_pos; double begin=now_ms(); evaluate(pre,input,pos); double prefill_ms=now_ms()-begin;
            if(pre==htp.ctx && execution.enabled()) {
                auto plan=execution.query();
                events->emit("execution_plan_consumed",round,"\"status\":"+std::to_string(plan.status)+",\"reason\":"+json(plan.reason)
                    +",\"graph_hits\":"+std::to_string(plan.graph_hits)+",\"descriptor_hits\":"+std::to_string(plan.descriptor_hits)
                    +",\"descriptor_misses\":"+std::to_string(plan.descriptor_misses));
            }
            auto first=greedy(pre,nv); size_t handoff_bytes=0;
            double handoff_ms=0,handoff_work_ms=0;
            if(pre!=cpu.ctx) {
                begin=now_ms();SessionSnapshot updated;updated.state=export_seq(pre);
                double export_ms=sample_cost?now_ms()-begin:0,import_ms=0;
                handoff_bytes=restore_session(cpu.ctx,updated,sample_cost?&import_ms:nullptr);
                handoff_work_ms=export_ms+import_ms;
                handoff_ms=now_ms()-begin;
            }
            std::vector<llama_token> ids; std::string output,closing; bool eog=false;
            begin=now_ms();
            DecodeHorizon horizon(sample_cost,d,begin);
            for(int i=0;i<max_tokens;++i) {
                auto tok=i==0?first:greedy(cpu.ctx,nv);
                evaluate(cpu.ctx,{tok},pos);
                horizon.evaluated(i+1,now_ms);
                if(llama_vocab_is_eog(vocab,tok)) {events->emit("action_complete",round);eog=true;closing=piece(vocab,tok);break;}
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
                << ",\"cpu_cost_source\":" << json(decision.cpu_source) << ",\"htp_cost_source\":" << json(decision.htp_source)
                << ",\"online_sampled\":" << (sample_cost?"true":"false")
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
                     << ",\"decode_prefix_ms\":" << optional_number(horizon.prefix_ms())
                     << ",\"handoff_work_ms\":" << (sample_cost?optional_number(handoff_work_ms):"null")
                     << "},\"background_occupied_at_publish\":" << (lane.occupied()?"true":"false") << "}";
            results.push_back(response.str());
            // 在任何回收/下一轮等待之前真正 flush 到消费者。
            events->emit("response_published",round,"\"result\":"+response.str());
            // 已发布响应后更新，下一轮生效。短于 D 的输出没有固定 D 样本。
            if(sample_cost) {
                bool updated=std::isfinite(horizon.prefix_ms()) && online.update(
                    cost_key,prefill_ms+handoff_work_ms+horizon.prefix_ms(),now_ms());
                events->emit("online_cost_updated",round,"\"updated\":"+(updated?std::string("true"):"false")
                    +",\"reason\":"+json(updated?"observed_path":"decode_horizon_incomplete")
                    +",\"path\":"+json(cost_key.path)+",\"H\":"+std::to_string(cost_key.h)
                    +",\"P\":"+std::to_string(cost_key.p)+",\"D\":"+std::to_string(cost_key.d)
                    +",\"samples\":"+std::to_string(online.samples(cost_key,now_ms())));
            }
            ++round; started=false; observed=false;
        } else throw std::runtime_error("unknown tool session command");
    }
    throw std::runtime_error("tool protocol EOF without CLOSE");
}

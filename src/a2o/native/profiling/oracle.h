#pragma once
#include <future>
#include "profiling/text_fixture.h"
#include "scheduling/pending_wait.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cerrno>

// 仅访问 adb reverse 的本地只读 HTTP Tool；完整 body 才作为新增输入。
static std::string oracle_http(int port,int p,int gap,int id) {
    int fd=socket(AF_INET,SOCK_STREAM,0);require(fd>=0,"HTTP socket failed");
    struct Close { int fd; ~Close(){::close(fd);} } close{fd};
    timeval timeout{30,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
    sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(port);addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    require(connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0,"HTTP connect failed");
    std::string req="GET /tool?p="+std::to_string(p)+"&gap="+std::to_string(gap)+"&id="+std::to_string(id)+" HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    for(size_t n=0;n<req.size();) {auto sent=send(fd,req.data()+n,req.size()-n,0);if(sent<0 && errno==EINTR)continue;require(sent>0,"HTTP send failed");n+=sent;}
    std::string raw;char buffer[4096];ssize_t n=0;size_t total=std::string::npos,split=std::string::npos;
    // 以 Content-Length 的完整正文为 Tool 返回点，不等待 adb reverse 的连接关闭。
    for(;;) {
        n=recv(fd,buffer,sizeof(buffer),0);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)break;
        raw.append(buffer,n);require(raw.size()<65536,"HTTP response too large");
        if(total==std::string::npos && (split=raw.find("\r\n\r\n"))!=std::string::npos) {
            require(raw.find("HTTP/1.0 200 OK\r\n")==0,"HTTP status failed");
            auto length=raw.find("\r\nContent-Length: ");
            require(length!=std::string::npos && length<split,"HTTP Content-Length missing");
            auto begin=length+18,end=raw.find("\r\n",begin);std::string value=raw.substr(begin,end-begin);
            require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid HTTP Content-Length");
            total=split+4+std::stoul(value);require(total<65536,"HTTP body too large");
        }
        if(total!=std::string::npos && raw.size()>=total)break;
    }
    require(total!=std::string::npos && raw.size()==total,"HTTP incomplete response bytes="+std::to_string(raw.size())+" expected="+std::to_string(total)+" recv="+std::to_string(n)+" errno="+std::to_string(errno));
    return raw.substr(split+4);
}

static int run_oracle(const std::map<std::string,std::string> & args) {
    auto get=[&](const std::string & k){return args.at(k);};
    require(get("--mode")=="hybrid" && get("--policy")=="none" && get("--pre-move")=="0","explicit hybrid/none required");
    const std::string candidate_backend=args.count("--oracle-candidate-backend")?get("--oracle-candidate-backend"):"htp";
    require(candidate_backend=="htp" || candidate_backend=="opencl","unsupported candidate backend");
    double start=now_ms();Engine cpu(get("--model"),false,false),htp(get("--model"),true,false,candidate_backend=="opencl"?"GPUOpenCL":"HTP0");
    double init=now_ms()-start;auto vocab=llama_model_get_vocab(cpu.model);int nv=llama_vocab_n_tokens(vocab);
    const int h=std::stoi(args.count("--profile-history")?get("--profile-history"):"4096");
    const double min_cosine=std::stod(args.count("--oracle-min-cosine")?get("--oracle-min-cosine"):"0.999");
    std::string error_limit=args.count("--oracle-max-nrmse")?get("--oracle-max-nrmse"):"0.05";
    const double max_nrmse=error_limit=="none"?std::numeric_limits<double>::infinity():std::stod(error_limit);
    require(h>0 && min_cosine>=0 && min_cosine<=1 && max_nrmse>=0,"invalid oracle history/thresholds");
    std::vector<int> input_lengths;std::istringstream lengths(args.count("--profile-inputs")?get("--profile-inputs"):"16,64");std::string length;
    while(std::getline(lengths,length,',')) {int p=std::stoi(length);require(p>=16 && p<=2048,"P must be within 16..2048");require(std::find(input_lengths.begin(),input_lengths.end(),p)==input_lengths.end(),"duplicate P");input_lengths.push_back(p);}
    require(!input_lengths.empty(),"missing P conditions");
    const bool prepack=args.count("--htp-kv-prepack") && get("--htp-kv-prepack")=="1";
    auto reg=ggml_backend_reg_by_name("HTP");
    if(!reg) reg=ggml_backend_reg_by_name("Hexagon");
    using KvControl=int(*)(ggml_backend_reg_t,int,int,uint64_t*);
    KvControl kv_control=prepack && reg?reinterpret_cast<KvControl>(ggml_backend_reg_get_proc_address(reg,"a2o_kv_control_v1")):nullptr;
    require(candidate_backend=="htp" || !prepack,"OpenCL cannot use HTP prepack");
    require(!prepack || kv_control,"private HTP prepack bridge unavailable");
    if(prepack) require(h%32==0,"prepack H must align 32");
    const bool release_tiles=args.count("--htp-kv-release-before-decode") && get("--htp-kv-release-before-decode")=="1";
    require(!release_tiles || prepack,"release requires private prepack bridge");
    const bool release_calibration=args.count("--oracle-release-calibration-state") && get("--oracle-release-calibration-state")=="1";
    int port=std::stoi(get("--http-port"));
    double history_fixture_start=now_ms();
    auto prefix=observation_history(vocab,h);
    double history_fixture_ms=now_ms()-history_fixture_start;
    const bool history_htp=args.count("--oracle-history-backend") && get("--oracle-history-backend")=="htp";
    llama_pos pos=0;start=now_ms();evaluate(history_htp?htp.ctx:cpu.ctx,prefix,pos);double history_ms=now_ms()-start;
    double history_handoff_ms=0;
    if(history_htp) {start=now_ms();transfer(htp.ctx,cpu.ctx);history_handoff_ms=now_ms()-start;}
    SessionSnapshot base;base.state=export_seq(cpu.ctx);base.next_pos=h;base.evaluated_tokens=h;
    double http_fixture_start=now_ms();
    for(int p:input_lengths) {
        std::string text;
        if(args.count("--oracle-fixture") && get("--oracle-fixture")=="neutral_padding_v1") {
            const std::string head="\n<|im_start|>user\n",tail="\nWait.<|im_end|>\n<|im_start|>assistant\n";
            int guess=p-int(tokens(vocab,head+tail,false).size());
            for(int n=std::max(0,guess-2);n<=guess+2;++n) {
                std::string fill;for(int i=0;i<n;++i)fill+=" note";
                std::string candidate=head+fill+tail;
                if(tokens(vocab,candidate,false).size()==size_t(p)){text=candidate;break;}
            }
            std::string fill;
            for(int n=0;text.empty() && n<=p;++n) {
                std::string candidate="\n<|im_start|>user\n"+fill+"\nWait.<|im_end|>\n<|im_start|>assistant\n";
                if(tokens(vocab,candidate,false).size()==size_t(p)){text=candidate;break;}
                fill+=" note";
            }
        } else {
        auto head=tokens(vocab,"\n<|im_start|>user\n",false);
        auto tail=tokens(vocab,"Wait.<|im_end|>\n<|im_start|>assistant\n",false);
        auto padding=tokens(vocab,"The current screen remains unchanged. No action is needed. ",false);
        require(size_t(p)>=head.size()+tail.size(),"P too small for complete user turn");
        size_t n=p-head.size()-tail.size();std::string original_fill;
        for(size_t i=0;i<n;++i)original_fill+=piece(vocab,padding[i%padding.size()]);
        std::string fill;for(int i=0;i<30;++i)fill+="The current screen remains unchanged. No action is needed. ";
        size_t distance=fill.size()+1;
        for(size_t chars=0;chars<=fill.size();++chars) {
            std::string candidate="\n<|im_start|>user\n"+fill.substr(0,chars)+"Wait.<|im_end|>\n<|im_start|>assistant\n";
            size_t delta=chars>original_fill.size()?chars-original_fill.size():original_fill.size()-chars;
            if(delta<distance && tokens(vocab,candidate,false).size()==size_t(p)){text=candidate;distance=delta;}
        }
        }
        require(!text.empty(),"cannot construct exact HTTP P fixture");
        std::ofstream(get("--out")+".P"+std::to_string(p)+".txt")<<text;
    }
    std::ofstream prefix_file(get("--out")+".prefix.txt");for(auto tok:prefix)prefix_file<<piece(vocab,tok);prefix_file.close();
    ToolEvents events(get("--out")+".events.jsonl");
    events.emit("session_ready",-1,"\"model_context_init_ms\":"+std::to_string(init)+",\"history_build_ms\":"+std::to_string(history_ms)+",\"history_backend\":"+json(history_htp?"htp":"cpu")+",\"history_handoff_ms\":"+std::to_string(history_handoff_ms)+",\"history_fixture_ms\":"+std::to_string(history_fixture_ms)+",\"http_fixture_ms\":"+std::to_string(now_ms()-http_fixture_start));
    std::vector<llama_token> reference_ids;std::string reference_output;
    if(args.count("--oracle-reference-ids")) {
        std::istringstream values(get("--oracle-reference-ids"));std::string value;
        while(std::getline(values,value,',')) {int id=std::stoi(value);require(id>=0 && id<nv,"invalid reference token");reference_ids.push_back(id);}
        require(!reference_ids.empty() && llama_vocab_is_eog(vocab,reference_ids.back()),"reference must end with EOG");
        reference_output=get("--oracle-reference-output");std::string decoded;
        for(size_t i=0;i+1<reference_ids.size();i++) decoded+=piece(vocab,reference_ids[i]);
        require(decoded==reference_output,"reference text/token mismatch");
    }
    std::map<int,std::vector<float>> reference_pre,reference_final,htp_reference_pre;
    std::map<int,std::vector<uint8_t>> htp_reference_state;
    std::ofstream output(get("--out"));require(bool(output),"oracle output unavailable");
    bool measurement_started=false;
    std::string line;
    while(std::getline(std::cin,line)) {
        if(line=="CLOSE") {events.emit("session_closed",-1);return 0;}
        std::istringstream cmd(line);std::string op,strategy,route,phase;int id,p,g,want,qualified;double cc,hc;
        require(bool(cmd>>op>>id>>p>>g>>strategy>>route>>phase>>cc>>hc>>want>>qualified) && op=="TRIAL","invalid oracle command");
        require(phase=="calibration" || phase=="measurement","invalid trial phase");
        require(!measurement_started || phase=="measurement","calibration after measurement is unsupported");
        if(phase=="measurement" && !measurement_started) {
            measurement_started=true;
            if(release_calibration) {
                size_t state_bytes=0;for(const auto &r:htp_reference_state)state_bytes+=r.second.size();
                htp_reference_state.clear();htp_reference_pre.clear();reference_pre.clear();reference_final.clear();
                events.emit("calibration_buffers_released",id,"\"state_bytes\":"+std::to_string(state_bytes));
            }
        }
        int pack_request=-1;cmd>>pack_request;require(pack_request>=-1 && pack_request<=1,"invalid pack request");
        int cpu_prefill_threads=4;cmd>>cpu_prefill_threads;
        int bounded_wait=0;double prepare_upper_ms=0,wait_cap_ms=0,wait_margin_ms=0;
        if(cmd>>bounded_wait) {
            require(bool(cmd>>prepare_upper_ms>>wait_cap_ms>>wait_margin_ms),"missing bounded wait costs");
            require(bounded_wait==0 || bounded_wait==1,"invalid bounded wait mode");
        }
        require(cpu_prefill_threads>=1 && cpu_prefill_threads<=sysconf(_SC_NPROCESSORS_ONLN),"invalid CPU prefill threads");
        // 单token decode 保持4线程，仅改变多token prefill线程；历史建立仍为4线程。
        llama_set_n_threads(cpu.ctx,4,cpu_prefill_threads);
        require((std::find(input_lengths.begin(),input_lengths.end(),p)!=input_lengths.end()) && g>=0 && (strategy=="after_return"||strategy=="always"||strategy=="cost_aware") && (route=="CC"||route=="HC"||route=="router"),"invalid condition");
        require(h+p+64<=context_size,"oracle context too small");
        if(prepack) require(kv_control(reg,0,0,nullptr)==0,"prepack reset failed");
        bool pack_trial=prepack && (route!="CC" || strategy!="after_return") && !(phase=="calibration" && route=="HC" && !htp_reference_pre.count(p));
        if(pack_request>=0) pack_trial=prepack && pack_request==1;
        uint64_t pack_stats[6]={};double pack_ms=0;
        restore_session(cpu.ctx,base);llama_memory_clear(llama_get_memory(htp.ctx),true);
        bool early=strategy=="always" || (strategy=="cost_aware" && want==1);
        double prepare_start=0,prepare_end=0;std::future<void> worker;
        auto prepare=[&] {
            prepare_start=now_ms();restore_session(htp.ctx,base);
            if(pack_trial) {double t=now_ms();require(kv_control(reg,h,phase=="calibration",nullptr)==0,"KV tile preparation failed");pack_ms=now_ms()-t;}
            prepare_end=now_ms();
        };
        double issued=now_ms(),active_issued=active_ms();
        if(early)worker=std::async(std::launch::async,prepare);
        auto body=oracle_http(port,p,g,id);double returned=now_ms(),active_returned=active_ms();
        bool ready=worker.valid() && worker.wait_for(std::chrono::milliseconds(0))==std::future_status::ready;
        double obs_start=now_ms();auto suffix=tokens(vocab,body,false);require(suffix.size()==size_t(p),"real HTTP P mismatch");double obs_ms=now_ms()-obs_start;
        std::string selected=route;
        PendingWaitPlan wait_plan;
        bool wait_attempted=false,wait_succeeded=false,wait_timed_out=false;
        double pending_wait_ms=0;
        std::string route_reason="forced_path";
        if(route=="router") {
            CostRouter current;
            current.rows={{h,p,int(reference_ids.size()),3,"CC",cc,0}};
            if(qualified)current.rows.push_back({h,p,int(reference_ids.size()),3,"HC",hc,0});
            // 仅实验基线：Tool 返回后按包含准备成本的独立校准选路，允许按需准备。
            bool post_tool=strategy=="after_return" && args.count("--oracle-post-tool-routing") && get("--oracle-post-tool-routing")=="1";
            auto decision=current.choose(h,p,reference_ids.size(),post_tool?"ready":ready?"ready":early?"pending":"absent");
            selected=decision.backend=="htp"?"HC":"CC";
            route_reason=decision.reason;
            if(bounded_wait && early && !ready && !post_tool) {
                wait_plan=plan_pending_wait(qualified,cc,hc,prepare_upper_ms,wait_cap_ms,wait_margin_ms);
                route_reason=wait_plan.reason;
                if(wait_plan.eligible) {
                    wait_attempted=true;double t=now_ms();
                    wait_succeeded=await_pending(worker,wait_plan);
                    pending_wait_ms=now_ms()-t;wait_timed_out=!wait_succeeded;
                    if(wait_succeeded) {selected="HC";route_reason="pending_ready_within_budget";}
                    else {selected="CC";route_reason="pending_wait_timeout";}
                }
            }
        }
        double wait_start=now_ms();
        if(selected=="HC") {if(worker.valid())worker.get();else prepare();}
        double wait_ms=pending_wait_ms+now_ms()-wait_start;
        auto pre=selected=="HC"?htp.ctx:cpu.ctx;pos=h;
        start=now_ms();evaluate(pre,suffix,pos);double pre_ms=now_ms()-start;
        double pack_release_ms=0;
        if(pack_trial && selected=="HC" && release_tiles) {
            double t=now_ms();require(kv_control(reg,-1,0,pack_stats)==0,"prepack release failed");
            pack_release_ms=now_ms()-t;
        }
        auto first=greedy(pre,nv);std::vector<float> pre_logits;
        if(phase=="calibration") {auto ptr=llama_get_logits_ith(pre,-1);pre_logits.assign(ptr,ptr+nv);}
        double handoff=0;size_t bytes=0;
        if(pre!=cpu.ctx) {start=now_ms();bytes=transfer(pre,cpu.ctx);handoff=now_ms()-start;}
        std::vector<llama_token> ids;std::string text;bool eog=false;
        rusage decode_before{},decode_after{};
        require(getrusage(RUSAGE_SELF,&decode_before)==0,"decode resource audit unavailable");
        start=now_ms();for(int i=0;i<64;++i) {auto tok=i?greedy(cpu.ctx,nv):first;ids.push_back(tok);evaluate(cpu.ctx,{tok},pos);if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}text+=piece(vocab,tok);}
        double decode=now_ms()-start;double completed=now_ms();
        require(getrusage(RUSAGE_SELF,&decode_after)==0,"decode resource audit unavailable");
        auto cpu_ms=[](const rusage &r) {return 1000.*(r.ru_utime.tv_sec+r.ru_stime.tv_sec)+(r.ru_utime.tv_usec+r.ru_stime.tv_usec)/1000.;};
        events.emit("response_published",id,"\"output\":"+json(text)+",\"eog\":"+(eog?"true":"false")
                    +",\"response_completed_ms\":"+std::to_string(completed));
        double published=now_ms(),active_published=active_ms();
        if(reference_ids.empty()) {require(phase=="calibration" && selected=="CC","CPU reference must run first");reference_ids=ids;reference_output=text;}
        bool same=ids==reference_ids && text==reference_output;
        double cosine=1,error=0,htp_cosine=1,htp_error=0;
        if(phase=="calibration") {
            auto ptr=llama_get_logits_ith(cpu.ctx,-1);std::vector<float> final(ptr,ptr+nv);
            if(!reference_pre.count(p)){require(selected=="CC","missing CPU logits reference");reference_pre[p]=pre_logits;reference_final[p]=final;}
            if(prepack && selected=="HC") {
                if(!htp_reference_pre.count(p)) htp_reference_pre[p]=pre_logits;
                else {auto d=path_difference(htp_reference_pre[p],pre_logits);htp_cosine=d.first;htp_error=d.second;
                    require(htp_error==0,"prepacked HTP logits differ from original HTP");}
            }
            auto a=path_difference(reference_pre[p],pre_logits),b=path_difference(reference_final[p],final);
            cosine=std::min(a.first,b.first);error=std::max(a.second,b.second);
        }
        // CC 的响应计时必须先结束，再回收尚未退出的准备线程；争用保留在真实耗时内。
        double drain_start=now_ms();if(worker.valid())worker.get();double drain=now_ms()-drain_start;
        if(pack_trial && selected=="HC") {
            if(!release_tiles) require(kv_control(reg,0,0,pack_stats)==0,"prepack stats failed");
            else {uint64_t released[6]={};require(kv_control(reg,0,0,released)==0 && released[4]==0 && released[5]==0,"prepack release not verified");}
            if(p>=32) require(pack_stats[0]>0 && pack_stats[1]>0 && pack_stats[2]>=pack_stats[4]/2 && pack_stats[3]>=pack_stats[4]/2 && pack_stats[5]==72,"KV tiles not actually reused: K_hits="+std::to_string(pack_stats[0])+", V_hits="+std::to_string(pack_stats[1])+", K_bytes="+std::to_string(pack_stats[2])+", V_bytes="+std::to_string(pack_stats[3])+", cache_bytes="+std::to_string(pack_stats[4])+", entries="+std::to_string(pack_stats[5]));
        }
        bool state_compared=false;
        if(prepack && selected=="HC" && phase=="calibration") {
            auto state=export_seq(cpu.ctx);
            if(!htp_reference_state.count(p)) htp_reference_state[p]=std::move(state);
            else {require(state==htp_reference_state[p],"prepacked final KV differs from original HTP");state_compared=true;}
        }
        bool numerical=cosine>=min_cosine && error<=max_nrmse;
        bool legacy_numerical=cosine>=.999 && error<=.05;
        bool passed=eog && same;
        std::ostringstream row;row<<"{\"id\":"<<id<<",\"phase\":"<<json(phase)<<",\"H\":"<<h<<",\"P\":"<<p<<",\"D\":"<<ids.size()<<",\"gap_requested_ms\":"<<g
          <<",\"cpu_prefill_threads\":"<<cpu_prefill_threads<<",\"cpu_decode_threads\":4"
          <<",\"candidate_backend\":"<<json(candidate_backend)<<",\"strategy\":"<<json(strategy)<<",\"requested_route\":"<<json(route)<<",\"route\":"<<json(selected)<<",\"prepared_early\":"<<(early?"true":"false")<<",\"ready_at_return\":"<<(ready?"true":"false")
          <<",\"want_prepare\":"<<(want?"true":"false")<<",\"wait_policy\":"<<json(bounded_wait?"bounded":"ready_only")
          <<",\"route_reason\":"<<json(route_reason)<<",\"pending_wait_eligible\":"<<(wait_plan.eligible?"true":"false")
          <<",\"pending_wait_attempted\":"<<(wait_attempted?"true":"false")<<",\"pending_wait_succeeded\":"<<(wait_succeeded?"true":"false")
          <<",\"pending_wait_timed_out\":"<<(wait_timed_out?"true":"false")<<",\"pending_wait_budget_ms\":"<<std::fixed<<wait_plan.budget_ms
          <<",\"pending_wait_ms\":"<<pending_wait_ms<<",\"calibrated_prepare_upper_ms\":"<<prepare_upper_ms
          <<",\"calibrated_cc_ms\":"<<cc<<",\"calibrated_hc_ms\":"<<hc
          <<",\"http_issued_ms\":"<<std::fixed<<issued<<",\"http_returned_ms\":"<<returned<<",\"response_completed_ms\":"<<published<<",\"prepare_started_ms\":"<<prepare_start<<",\"prepare_completed_ms\":"<<prepare_end
          <<",\"actual_gap_ms\":"<<returned-issued<<",\"response_after_return_ms\":"<<published-returned<<",\"tool_to_response_ms\":"<<published-issued
          <<",\"active_actual_gap_ms\":"<<active_returned-active_issued<<",\"active_tool_to_response_ms\":"<<active_published-active_issued<<",\"suspend_during_tool_to_response_ms\":"<<(published-issued)-(active_published-active_issued)
          <<",\"response_compute_completed_ms\":"<<completed<<",\"prepare_ms\":"<<(prepare_end-prepare_start)<<",\"wait_or_prepare_after_return_ms\":"<<wait_ms<<",\"observation_processing_ms\":"<<obs_ms<<",\"prefill_ms\":"<<pre_ms<<",\"handoff_checked_ms\":"<<handoff<<",\"decode_ms\":"<<decode<<",\"drain_after_response_ms\":"<<drain
          <<",\"handoff_bytes\":"<<bytes<<",\"state_bytes\":"<<base.state.size()<<",\"vision_init_ms\":0,\"image_encode_calls\":0,\"eog\":"<<(eog?"true":"false")<<",\"generated_ids_equal\":"<<(same?"true":"false")<<",\"boundary_min_cosine\":"<<cosine<<",\"boundary_max_nrmse\":"<<error
          <<",\"numerical_compared\":"<<(phase=="calibration"?"true":"false")<<",\"numerical_passed\":"<<(phase=="calibration"?(numerical?"true":"false"):"null")
          <<",\"legacy_numerical_passed\":"<<(phase=="calibration"?(legacy_numerical?"true":"false"):"null")
          <<",\"cost_covered\":"<<(qualified?"true":"false")<<",\"passed\":"<<(passed?"true":"false")<<",\"output\":"<<json(text)<<",\"generated_ids\":[";
        for(size_t i=0;i<ids.size();++i)row<<(i?",":"")<<ids[i];row<<"]}";
        std::string result=row.str();result.pop_back();
        size_t retained_calibration=0;for(const auto &r:htp_reference_state)retained_calibration+=r.second.size();
        std::ostringstream extra;extra<<",\"kv_prepack_enabled\":"<<(pack_trial?"true":"false")<<",\"kv_prepack_ms\":"<<pack_ms
          <<",\"calibration_state_retained_bytes\":"<<retained_calibration
          <<",\"kv_prepack_released_before_decode\":"<<((pack_trial && release_tiles)?"true":"false")<<",\"kv_prepack_release_ms\":"<<pack_release_ms
          <<",\"kv_prepack_k_hits\":"<<pack_stats[0]<<",\"kv_prepack_v_hits\":"<<pack_stats[1]
          <<",\"kv_prepack_k_bytes\":"<<pack_stats[2]<<",\"kv_prepack_v_bytes\":"<<pack_stats[3]
          <<",\"kv_prepack_cache_bytes\":"<<pack_stats[4]<<",\"kv_prepack_entries\":"<<pack_stats[5]
          <<",\"decode_cpu_ms\":"<<cpu_ms(decode_after)-cpu_ms(decode_before)
          <<",\"decode_minor_faults\":"<<decode_after.ru_minflt-decode_before.ru_minflt
          <<",\"decode_major_faults\":"<<decode_after.ru_majflt-decode_before.ru_majflt
          <<",\"decode_voluntary_switches\":"<<decode_after.ru_nvcsw-decode_before.ru_nvcsw
          <<",\"decode_involuntary_switches\":"<<decode_after.ru_nivcsw-decode_before.ru_nivcsw
          <<",\"htp_baseline_state_compared\":"<<(state_compared?"true":"false")<<",\"htp_baseline_compared\":"<<((prepack && selected=="HC" && phase=="calibration" && pack_trial)?"true":"false")<<",\"htp_baseline_cosine\":"<<htp_cosine<<",\"htp_baseline_nrmse\":"<<htp_error<<"}";
        result+=extra.str();
        output<<result<<'\n';output.flush();events.emit("trial_completed",id,"\"result\":"+result);
        require(passed || (selected=="HC" && (phase=="calibration" || !qualified)),"qualified oracle response failed");
    }
    throw std::runtime_error("oracle EOF without CLOSE");
}

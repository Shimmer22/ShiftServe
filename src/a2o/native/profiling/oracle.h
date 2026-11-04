#pragma once
#include <future>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// 仅访问 adb reverse 的本地只读 HTTP Tool；完整 body 才作为新增输入。
static std::string oracle_http(int port,int p,int gap,int id) {
    int fd=socket(AF_INET,SOCK_STREAM,0);require(fd>=0,"HTTP socket failed");
    struct Close { int fd; ~Close(){::close(fd);} } close{fd};
    timeval timeout{30,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
    sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(port);addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    require(connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0,"HTTP connect failed");
    std::string req="GET /tool?p="+std::to_string(p)+"&gap="+std::to_string(gap)+"&id="+std::to_string(id)+" HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    for(size_t n=0;n<req.size();) {auto sent=send(fd,req.data()+n,req.size()-n,0);require(sent>0,"HTTP send failed");n+=sent;}
    std::string raw;char buffer[4096];ssize_t n;
    while((n=recv(fd,buffer,sizeof(buffer),0))>0) {raw.append(buffer,n);require(raw.size()<65536,"HTTP response too large");}
    require(n==0 && raw.find("HTTP/1.0 200 OK\r\n")==0,"HTTP read/status failed");
    auto split=raw.find("\r\n\r\n");require(split!=std::string::npos,"HTTP headers invalid");return raw.substr(split+4);
}

static int run_oracle(const std::map<std::string,std::string> & args) {
    auto get=[&](const std::string & k){return args.at(k);};
    require(get("--mode")=="hybrid" && get("--policy")=="none" && get("--pre-move")=="0","explicit hybrid/none required");
    double start=now_ms();Engine cpu(get("--model"),false,false),htp(get("--model"),true,false);
    double init=now_ms()-start;auto vocab=llama_model_get_vocab(cpu.model);int nv=llama_vocab_n_tokens(vocab);
    const int h=std::stoi(args.count("--profile-history")?get("--profile-history"):"4096");
    const double min_cosine=std::stod(args.count("--oracle-min-cosine")?get("--oracle-min-cosine"):"0.999");
    std::string error_limit=args.count("--oracle-max-nrmse")?get("--oracle-max-nrmse"):"0.05";
    const double max_nrmse=error_limit=="none"?std::numeric_limits<double>::infinity():std::stod(error_limit);
    require(h>0 && min_cosine>=0 && min_cosine<=1 && max_nrmse>=0,"invalid oracle history/thresholds");
    std::vector<int> input_lengths;std::istringstream lengths(args.count("--profile-inputs")?get("--profile-inputs"):"16,64");std::string length;
    while(std::getline(lengths,length,',')) {int p=std::stoi(length);require(p>=16 && p<=128,"P must be within 16..128");require(std::find(input_lengths.begin(),input_lengths.end(),p)==input_lengths.end(),"duplicate P");input_lengths.push_back(p);}
    require(!input_lengths.empty(),"missing P conditions");
    const bool prepack=args.count("--htp-kv-prepack") && get("--htp-kv-prepack")=="1";
    auto reg=ggml_backend_reg_by_name("HTP");
    if(!reg) reg=ggml_backend_reg_by_name("Hexagon");
    using KvControl=int(*)(ggml_backend_reg_t,int,int,uint64_t*);
    KvControl kv_control=prepack && reg?reinterpret_cast<KvControl>(ggml_backend_reg_get_proc_address(reg,"a2o_kv_control_v1")):nullptr;
    require(!prepack || kv_control,"private HTP prepack bridge unavailable");
    if(prepack) require(h%32==0,"prepack H must align 32");
    const bool release_tiles=args.count("--htp-kv-release-before-decode") && get("--htp-kv-release-before-decode")=="1";
    require(!release_tiles || prepack,"release requires private prepack bridge");
    const bool release_calibration=args.count("--oracle-release-calibration-state") && get("--oracle-release-calibration-state")=="1";
    int port=std::stoi(get("--http-port"));
    auto prefix=tokens(vocab,"<|im_start|>system\nYou are a GUI agent. For every new request return only {\"action\":\"wait\"}. Do not repeat observation text. No markdown.<|im_end|>\n<|im_start|>user\nPast screen observations:\n",true);
    auto closing=tokens(vocab,"\nThe tool completed successfully. Continue waiting for the next request.<|im_end|>\n<|im_start|>assistant\n{\"action\":\"wait\"}<|im_end|>",false);
    // 复用四路径已验收历史夹具，固定内容以便比较 backend 数值准入。
    std::string records;
    const char * screens[]={"Settings shows battery level and network status.","Developer options shows logging, graphics and debugging controls.","The application displays a list of saved documents.","A confirmation dialog has been dismissed and the toolbar is visible."};
    for(int i=0;i<h;++i)records+="Observation "+std::to_string(i)+": "+screens[i%4]+" Tool result: success.\n";
    require(size_t(h)>=prefix.size()+closing.size(),"H too small for structured history");
    auto body_tokens=tokens(vocab,records,false);body_tokens.resize(h-prefix.size()-closing.size());
    prefix.insert(prefix.end(),body_tokens.begin(),body_tokens.end());prefix.insert(prefix.end(),closing.begin(),closing.end());
    require(prefix.size()==h,"H fixture mismatch");
    llama_pos pos=0;start=now_ms();evaluate(cpu.ctx,prefix,pos);double history_ms=now_ms()-start;
    SessionSnapshot base;base.state=export_seq(cpu.ctx);base.next_pos=h;base.evaluated_tokens=h;
    for(int p:input_lengths) {
        std::string text;
        if(args.count("--oracle-fixture") && get("--oracle-fixture")=="neutral_padding_v1") {
            std::string fill;
            for(int n=0;n<=256;++n) {
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
    events.emit("session_ready",-1,"\"model_context_init_ms\":"+std::to_string(init)+",\"history_build_ms\":"+std::to_string(history_ms));
    std::vector<llama_token> reference_ids;std::string reference_output;
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
        double issued=now_ms();
        if(early)worker=std::async(std::launch::async,prepare);
        auto body=oracle_http(port,p,g,id);double returned=now_ms();
        bool ready=worker.valid() && worker.wait_for(std::chrono::milliseconds(0))==std::future_status::ready;
        double obs_start=now_ms();auto suffix=tokens(vocab,body,false);require(suffix.size()==size_t(p),"real HTTP P mismatch");double obs_ms=now_ms()-obs_start;
        std::string selected=route;
        if(route=="router") {
            CostRouter current;
            current.rows={{h,p,int(reference_ids.size()),3,"CC",cc,0}};
            if(qualified)current.rows.push_back({h,p,int(reference_ids.size()),3,"HC",hc,0});
            auto decision=current.choose(h,p,reference_ids.size(),ready?"ready":early?"pending":"absent");
            selected=decision.backend=="htp"?"HC":"CC";
        }
        double wait_start=now_ms();
        if(selected=="HC") {if(worker.valid())worker.get();else prepare();}
        double wait_ms=now_ms()-wait_start;
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
        double published=now_ms();
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
            if(p>=32) require(pack_stats[0]>0 && pack_stats[1]>0 && pack_stats[2]>=pack_stats[4]/2 && pack_stats[3]>=pack_stats[4]/2 && pack_stats[5]==72,"KV tiles not actually reused");
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
          <<",\"strategy\":"<<json(strategy)<<",\"requested_route\":"<<json(route)<<",\"route\":"<<json(selected)<<",\"prepared_early\":"<<(early?"true":"false")<<",\"ready_at_return\":"<<(ready?"true":"false")
          <<",\"http_issued_ms\":"<<std::fixed<<issued<<",\"http_returned_ms\":"<<returned<<",\"response_completed_ms\":"<<published<<",\"prepare_started_ms\":"<<prepare_start<<",\"prepare_completed_ms\":"<<prepare_end
          <<",\"actual_gap_ms\":"<<returned-issued<<",\"response_after_return_ms\":"<<published-returned<<",\"tool_to_response_ms\":"<<published-issued
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

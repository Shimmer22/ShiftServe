// B 阶段：相同 CPU 历史快照上的成对 prefill，分开计时实际准备及返回。
struct ProfileMeasurement {
    double export_ms=0, clear_ms=0, import_ms=0, prefill_ms=0;
    double return_export_ms=0, return_clear_ms=0, return_import_ms=0, decode_ms=0;
    double fixture_restore_ms=0, verification_ms=0;
    size_t upload_bytes=0, return_bytes=0;
    std::string output;
    std::vector<llama_token> generated;
    std::vector<float> logits;
    llama_token first_token=-1;
    bool first_is_eog=false;
};

static void import_profile_state(llama_context * ctx,const std::vector<uint8_t> & blob) {
    require(llama_state_seq_set_data(ctx,blob.data(),blob.size(),0)==blob.size(),"profile state import failed");
    llama_synchronize(ctx);
}

static std::vector<llama_token> profile_tokens(const llama_vocab * vocab,int count,bool history) {
    auto head=tokens(vocab,history?
        "<|im_start|>system\nYou are a GUI agent. Return only a JSON object with action set to wait.<|im_end|>\n<|im_start|>user\n":
        "\n<|im_start|>user\n",history);
    auto tail=tokens(vocab,history?
        "<|im_end|>\n<|im_start|>assistant\n{\"action\":\"wait\"}<|im_end|>":
        "\nReturn only a JSON object with action set to wait.<|im_end|>\n<|im_start|>assistant\n",false);
    require(count>=int(head.size()+tail.size()),"token budget too small for template");
    auto pad=tokens(vocab," note",false);
    head.insert(head.end(),count-head.size()-tail.size(),pad.at(0));
    head.insert(head.end(),tail.begin(),tail.end()); return head;
}

static int run_profile(const std::map<std::string,std::string> & args) {
    auto get=[&](std::string key,std::string fallback="") {
        auto it=args.find(key); return it==args.end()?fallback:it->second;
    };
    int history=std::stoi(get("--profile-history")), repetitions=std::stoi(get("--profile-repeats","3"));
    std::vector<int> lengths; std::istringstream in(get("--profile-inputs","64,256,560,1024")); std::string part;
    while(std::getline(in,part,',')) lengths.push_back(std::stoi(part));
    double begin=now_ms(); Engine cpu(get("--model"),false,false), htp(get("--model"),true,false);
    double init_ms=now_ms()-begin;
    auto vocab=llama_model_get_vocab(cpu.model); int nv=llama_vocab_n_tokens(vocab);
    auto prefix=profile_tokens(vocab,history,true); llama_pos pos=0;
    { std::ofstream file(get("--out")+".prefix.txt"); for(auto token:prefix) file << piece(vocab,token); }
    std::cerr << "PROFILE build_history H=" << history << '\n';
    begin=now_ms(); evaluate(cpu.ctx,prefix,pos); double history_ms=now_ms()-begin;
    auto base=export_seq(cpu.ctx);
    auto base_resources=resource_sample();
    auto measure=[&](bool accelerator,const std::vector<llama_token> & suffix,bool restore_fixture=true) {
        ProfileMeasurement m;
        double t=now_ms();
        if(restore_fixture) {
            llama_memory_clear(llama_get_memory(cpu.ctx),false);
            import_profile_state(cpu.ctx,base); m.fixture_restore_ms=now_ms()-t;
        }
        auto ctx=accelerator?htp.ctx:cpu.ctx;
        if(accelerator) {
            t=now_ms(); auto blob=export_seq(cpu.ctx); m.export_ms=now_ms()-t; m.upload_bytes=blob.size();
            t=now_ms(); llama_memory_clear(llama_get_memory(htp.ctx),false); m.clear_ms=now_ms()-t;
            t=now_ms(); import_profile_state(htp.ctx,blob); m.import_ms=now_ms()-t;
            t=now_ms(); require(export_seq(htp.ctx)==blob,"profile history restore mismatch"); m.verification_ms+=now_ms()-t;
        }
        llama_pos next=history;
        t=now_ms(); evaluate(ctx,suffix,next); m.prefill_ms=now_ms()-t;
        auto first=greedy(ctx,nv); auto logits=llama_get_logits_ith(ctx,-1); m.logits.assign(logits,logits+nv);
        m.first_token=first; m.first_is_eog=llama_vocab_is_eog(vocab,first);
        if(accelerator) {
            t=now_ms(); auto blob=export_seq(htp.ctx); m.return_export_ms=now_ms()-t; m.return_bytes=blob.size();
            t=now_ms(); llama_memory_clear(llama_get_memory(cpu.ctx),false); m.return_clear_ms=now_ms()-t;
            t=now_ms(); import_profile_state(cpu.ctx,blob); m.return_import_ms=now_ms()-t;
            t=now_ms(); require(export_seq(cpu.ctx)==blob,"profile return mismatch"); m.verification_ms+=now_ms()-t;
        }
        t=now_ms();
        for(int i=0;i<24;++i) {
            auto token=i==0?first:greedy(cpu.ctx,nv);
            evaluate(cpu.ctx,{token},next);
            if(llama_vocab_is_eog(vocab,token)) break;
            m.generated.push_back(token); m.output+=piece(vocab,token);
        }
        m.decode_ms=now_ms()-t; return m;
    };
    // Diagnostic only: continue the original CPU context before any snapshot restore.
    // This distinguishes a bad long-context fixture from corruption introduced by restore.
    ProfileMeasurement direct;
    bool direct_check=get("--profile-direct-check","0")=="1";
    if(direct_check) direct=measure(false,profile_tokens(vocab,lengths.at(0),false),false);
    std::ofstream rows(get("--out")); require(bool(rows),"profile output unavailable");
    for(int rep=0;rep<repetitions;++rep) {
        auto order=lengths; if(rep%2) std::reverse(order.begin(),order.end());
        for(size_t j=0;j<order.size();++j) {
            int p=order[j]; require(history+p+24<=context_size,"profile context capacity exceeded");
            auto suffix=profile_tokens(vocab,p,false);
            { std::ofstream file(get("--out")+".P"+std::to_string(p)+".suffix.txt"); for(auto token:suffix) file << piece(vocab,token); }
            bool htp_first=(rep+j)%2;
            std::cerr << "PROFILE H=" << history << " P=" << p << " repetition=" << rep << " first=" << (htp_first?"htp":"cpu") << '\n';
            ProfileMeasurement c,h;
            if(htp_first) {h=measure(true,suffix);c=measure(false,suffix);} else {c=measure(false,suffix);h=measure(true,suffix);}
            double aa=0,bb=0,ab=0,dd=0,max_abs=0;
            for(int k=0;k<nv;++k) {
                double a=c.logits[k],b=h.logits[k],d=a-b;
                aa+=a*a;bb+=b*b;ab+=a*b;dd+=d*d;max_abs=std::max(max_abs,std::abs(d));
            }
            double cosine=ab/std::sqrt(aa*bb),nrmse=std::sqrt(dd/aa);
            bool responses=!c.output.empty() && !h.output.empty();
            bool numerical=cosine>=.999 && nrmse<=.05;
            auto emit=[&](const char * backend,const ProfileMeasurement & m) {
                rows << json(backend) << ":{\"state_export_ms\":" << m.export_ms << ",\"state_clear_ms\":" << m.clear_ms
                     << ",\"tiling_ms\":null,\"state_import_ms\":" << m.import_ms << ",\"prefill_ms\":" << m.prefill_ms
                     << ",\"return_export_ms\":" << m.return_export_ms << ",\"return_clear_ms\":" << m.return_clear_ms
                     << ",\"return_import_ms\":" << m.return_import_ms << ",\"decode_ms\":" << m.decode_ms
                     << ",\"fixture_restore_ms\":" << m.fixture_restore_ms << ",\"verification_ms\":" << m.verification_ms
                     << ",\"upload_bytes\":" << m.upload_bytes << ",\"return_bytes\":" << m.return_bytes
                     << ",\"first_token\":" << m.first_token << ",\"first_is_eog\":" << (m.first_is_eog?"true":"false")
                     << ",\"output\":" << json(m.output) << '}';
            };
            rows << "{\"history_tokens\":" << history << ",\"new_tokens\":" << p << ",\"repetition\":" << rep
                 << ",\"first_backend\":" << json(htp_first?"htp":"cpu") << ",\"model_context_init_ms\":" << init_ms
                 << ",\"history_build_ms\":" << history_ms << ",\"history_state_bytes\":" << base.size()
                 << ",\"workload\":\"text_suffix\",\"kv_type\":\"F16\",\"tiling_status\":\"no_explicit_transform; backend_internal_work_in_import\","
                 << "\"responses_passed\":" << (responses?"true":"false") << ",\"numerical_passed\":" << (numerical?"true":"false")
                 << ",\"generated_ids_equal\":" << (c.generated==h.generated?"true":"false")
                 << ",\"logits_cosine\":" << cosine << ",\"logits_nrmse\":" << nrmse << ",\"logits_max_abs\":" << max_abs << ',';
            emit("cpu",c);rows << ',';emit("htp",h);
            if(direct_check && rep==0 && j==0) {
                double a2=0,d2=0; bool exact=true;
                for(int k=0;k<nv;++k) {
                    double d=direct.logits[k]-c.logits[k]; a2+=double(direct.logits[k])*direct.logits[k];d2+=d*d;
                    exact=exact && direct.logits[k]==c.logits[k];
                }
                rows << ',';emit("direct_cpu",direct);
                rows << ",\"direct_restore_logits_exact\":" << (exact?"true":"false")
                     << ",\"direct_restore_nrmse\":" << std::sqrt(d2/a2)
                     << ",\"direct_restore_ids_equal\":" << (direct.generated==c.generated?"true":"false");
            }
            rows << ",\"resources\":{"; bool comma=false;
            for(auto & metric:resource_sample()) { rows << (comma?",":"") << json(metric.first) << ':' << metric.second;comma=true; }
            rows << "}}\n"; rows.flush();
            std::cerr << "PROFILE completed H=" << history << " P=" << p << " rep=" << rep
                      << " CPU_ms=" << c.prefill_ms << " HTP_ms=" << h.prefill_ms << " responses=" << responses << " numerical=" << numerical << '\n';
        }
    }
    return 0;
}

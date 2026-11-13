// 短decode归因：同步后端split、跨后端copy、同步等待与剩余主机工作。
#define A2O_SPARSE_PROBE_NO_MAIN
#include "sparse_feasibility.cpp"
#include <dlfcn.h>

static J observe_cpu_state(){
    J j;for(int core:{0,4,7}){std::string p="/sys/devices/system/cpu/cpu"+std::to_string(core)+"/cpufreq/scaling_cur_freq";std::ifstream f(p);long hz=0;if(f>>hz)j["cpu"+std::to_string(core)+"_kHz"]=hz;}
    std::ifstream f("/proc/self/cgroup");if(f)j["cgroup"]=std::string(std::istreambuf_iterator<char>(f),{});
    rusage usage{};if(getrusage(RUSAGE_SELF,&usage)==0){j["user_cpu_ms"]=usage.ru_utime.tv_sec*1000.+usage.ru_utime.tv_usec/1000.;j["system_cpu_ms"]=usage.ru_stime.tv_sec*1000.+usage.ru_stime.tv_usec/1000.;}return j;
}

int main(int argc,char **argv){try{
    require(argc==5,"usage: decode_attribution MODEL CONFIG BACKEND OUTPUT");
    std::string model=argv[1],backend=argv[3],path=argv[4];J cfg=J::parse(read(argv[2]));
    require(backend=="cpu"||backend=="htp","explicit backend");bool htp=backend=="htp";
    setenv("LLAMA_GRAPH_REUSE_DISABLE",cfg.value("graph_reuse_disable",false)?"1":"0",1);
    batch_size=128;context_size=cfg.at("context_size");ggml_backend_load_all();
    auto begin=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"a2o_attr_begin"));
    auto finish=reinterpret_cast<const char*(*)()>(dlsym(RTLD_DEFAULT,"a2o_attr_end"));
    bool instrument=cfg.value("host_instrumentation",false);require(!instrument||(begin&&finish),"missing private attribution API");
    J result={{"config",cfg},{"backend",backend},{"rows",J::array()}};double start=now_ms();
    Engine engine(model,htp,false,false,htp?cfg.value("htp_threads",4):6);auto *ctx=engine.ctx;auto *vocab=llama_model_get_vocab(engine.model);int nv=llama_vocab_n_tokens(vocab);
    result["model_context_init_ms"]=now_ms()-start;
    auto normal=tokens(vocab,cfg.at("normal_prompt"),true);llama_pos pos=0;evaluate(ctx,normal,pos);
    std::string text;bool eog=false;for(int i=0;i<16;++i){auto tok=greedy(ctx,nv);if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}text+=piece(vocab,tok);evaluate(ctx,{tok},pos);}
    result["normal_response"]={{"text",text},{"passed",eog&&text.find("OK")!=std::string::npos}};save_probe(path,result);require(result["normal_response"]["passed"],"normal response failed");
    auto prefix=cfg.at("prefix_ids").get<std::vector<llama_token>>(),input=cfg.at("input_ids").get<std::vector<llama_token>>(),output=cfg.at("generated_ids").get<std::vector<llama_token>>();
    if(cfg.contains("history_tokens")){
        int h=cfg.at("history_tokens");require(h>0&&h<=int(prefix.size()),"invalid diagnostic history length");prefix.resize(h);
        llama_memory_clear(llama_get_memory(ctx),true);pos=0;start=now_ms();evaluate(ctx,prefix,pos);
        result["history_build_ms"]=now_ms()-start;result["history_source"]="fresh exact prefix slice; recurrent state recomputed";
    }else{
    std::string cache_path=cfg.at("history_cache_metadata");auto stored=J::parse(read(cache_path));auto meta=stored.at("metadata");
    require(meta.at("prefix_ids").get<std::vector<llama_token>>()==prefix,"history exact tokens");
    require(meta.at("context").get<int>()==context_size&&meta.at("batch").get<int>()==batch_size,"history context/batch");
    auto identity=model+":"+std::to_string(std::filesystem::file_size(model))+":"+std::to_string(static_cast<long long>(std::filesystem::last_write_time(model).time_since_epoch().count()))+":";
    require(meta.at("identity").get<std::string>().find(identity)==0,"history model identity");
    auto bytes=read(cache_path+".state");uint64_t hash=14695981039346656037ULL;for(unsigned char b:bytes){hash^=b;hash*=1099511628211ULL;}std::ostringstream digest;digest<<std::hex<<hash;
    require(bytes.size()==stored.at("state_bytes").get<size_t>()&&digest.str()==stored.at("state_checksum").get<std::string>(),"history size/checksum");
    SessionSnapshot history;history.state.assign(bytes.begin(),bytes.end());history.next_pos=prefix.size();
    start=now_ms();restore_session(ctx,history);result["history_restore_ms"]=now_ms()-start;pos=prefix.size();
    }
    result["actual_H"]=prefix.size();result["actual_P"]=input.size();
    start=now_ms();evaluate(ctx,input,pos);result["full_prefill_ms"]=now_ms()-start;
    SessionSnapshot frozen;frozen.state=export_seq(ctx);frozen.next_pos=pos;result["state_bytes"]=frozen.state.size();result["state_restore_verified"]=true;
    double keep=cfg.value("keep_ratio",1.0);require(keep>=0&&keep<=1,"invalid FFN keep ratio");
    CompactProbe pack(engine.model);start=now_ms();pack.install(model,keep,htp,true);
    result["ffn_compaction"]={{"keep_ratio",keep},{"packing_ms",now_ms()-start},{"packed_bytes",pack.packed_bytes},{"layers",pack.shapes},
        {"scope","fixed contiguous channels; no importance selection or acceptance claim"}};
    // Prefill使用完整FFN。替换后先用不同于prefill的batch形状重建图，
    // 再恢复原状态；后续单token图可正常复用，避免测到旧dense图。
    if(keep!=1){
        require(input.size()>2&&output.size()>=2,"sparse graph rebuild requires distinct batch shapes");
        pos=frozen.next_pos;evaluate(ctx,{output[0],output[1]},pos);
        restore_session(ctx,frozen);result["ffn_compaction"]["graph_rebuilt_with_batch2_before_single_token"]=true;
    }
    if(cfg.value("weight_inventory",false)){
        J weights=J::array();for(auto &entry:engine.model->tensors_by_name){auto *w=entry.second;
            if(entry.first.find(".ffn_")!=std::string::npos){for(auto &layer:engine.model->layers){
                for(auto *active:{layer.ffn_up,layer.ffn_gate,layer.ffn_down})if(active&&entry.first==active->name)w=active;
            }}
            weights.push_back({{"name",entry.first},{"type",ggml_type_name(w->type)},{"ne",{w->ne[0],w->ne[1],w->ne[2],w->ne[3]}},
                {"logical_bytes",ggml_nbytes(w)},{"buffer",w->buffer?ggml_backend_buffer_name(w->buffer):"none"}});
        }result["weight_inventory"]=weights;
    }
    save_probe(path,result);std::vector<float> reference;
    int formal=0,warm=0;bool warming=true;double warm_ms=0;
    while(warming||formal<cfg.at("rounds").get<int>()){
        int rep=warming?-(++warm):formal++;
        restore_session(ctx,frozen);pos=frozen.next_pos;llama_perf_context_reset(ctx);auto before=resource_sample();
        J observed_before=cfg.value("observe_cpu_state",false)?observe_cpu_state():J(nullptr);
        std::cerr<<"A2O_DECODE_BEGIN rep="<<rep<<"\n";if(instrument)begin(1);
        start=now_ms();std::vector<double> per_token;
        for(auto tok:output){double t=now_ms();evaluate(ctx,{tok},pos);per_token.push_back(now_ms()-t);}
        double elapsed=now_ms()-start;J attribution=nullptr;if(instrument)attribution=J::parse(finish());
        std::cerr<<"A2O_DECODE_END rep="<<rep<<" elapsed_ms="<<elapsed<<"\n";
        auto perf=llama_perf_context(ctx);auto values=last_probe(ctx,nv);if(reference.empty())reference=values;
        result["rows"].push_back({{"repetition",rep},{"warmup",rep<0},{"D",output.size()},{"elapsed_ms",elapsed},{"token_ms",per_token},
            {"backend_timing",attribution},{"llama_eval_ms",perf.t_eval_ms},{"graph_reused",perf.n_reused},
            {"logits_vs_first_repeat",compare_probe(reference,values)},{"resources_before",before},{"resources_after",resource_sample()},
            {"cpu_state_before",observed_before},{"cpu_state_after",cfg.value("observe_cpu_state",false)?observe_cpu_state():J(nullptr)}});
        if(warming){warm_ms+=elapsed;if(warm_ms>=1000.*cfg.value("minimum_warmup_seconds",0.))warming=false;}
        result["warmup_elapsed_ms"]=warm_ms;result["warmup_runs"]=warm;save_probe(path,result);
    }
    if(cfg.value("save_logits",false)){
        restore_session(ctx,frozen);pos=frozen.next_pos;std::ofstream f(path+".logits.f32",std::ios::binary);J ids=J::array();
        for(auto tok:output){evaluate(ctx,{tok},pos);auto values=last_probe(ctx,nv);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));ids.push_back(std::max_element(values.begin(),values.end())-values.begin());}
        require(bool(f),"save diagnostic logits");result["numerical_replay"]={{"argmax_ids",ids},{"vocab",nv},{"tokens",output.size()}};
    }
    if(cfg.value("check_checkpoint_replay",false)){
        require(output.size()>=4,"checkpoint fixture too short");size_t split=output.size()/2;
        restore_session(ctx,frozen);pos=frozen.next_pos;for(auto tok:output)evaluate(ctx,{tok},pos);auto expected=last_probe(ctx,nv);auto expected_state=export_seq(ctx);
        restore_session(ctx,frozen);pos=frozen.next_pos;for(size_t i=0;i<split;++i)evaluate(ctx,{output[i]},pos);
        SessionSnapshot middle;middle.state=export_seq(ctx);middle.next_pos=pos;
        restore_session(ctx,middle);pos=middle.next_pos;for(size_t i=split;i<output.size();++i)evaluate(ctx,{output[i]},pos);
        auto comparison=compare_probe(expected,last_probe(ctx,nv));bool same_state=expected_state==export_seq(ctx);
        result["checkpoint_replay"]={{"accepted_prefix_tokens",split},{"restored_bytes_verified",true},{"logits",comparison},{"final_state_bitwise_equal",same_state}};
        save_probe(path,result);require(same_state&&comparison["argmax_match"].get<bool>(),"checkpoint replay mismatch");
    }
    if(cfg.value("check_json_memory",false)){
        llama_memory_clear(llama_get_memory(ctx),true);pos=0;
        const std::string assistant="<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const std::string first="<|im_start|>user\nRemember the number 314159 for my next question. Reply only with the JSON object {\"code\":314159}, without markdown.<|im_end|>\n"+assistant;
        evaluate(ctx,tokens(vocab,first,true),pos);J turns=J::array();
        for(int turn=0;turn<2;++turn){
            std::string response;bool stopped=false;llama_token end=-1;J ids=J::array();
            for(int i=0;i<64;++i){auto tok=greedy(ctx,nv);if(llama_vocab_is_eog(vocab,tok)){stopped=true;end=tok;break;}response+=piece(vocab,tok);ids.push_back(tok);evaluate(ctx,{tok},pos);}
            J parsed=J::parse(response,nullptr,false);bool passed=stopped&&parsed.is_object()&&parsed.contains("code")&&parsed["code"]==314159;
            turns.push_back({{"turn",turn},{"response",response},{"ids",ids},{"eog",stopped},{"parsed",parsed.is_discarded()?J(nullptr):parsed},{"passed",passed}});
            result["json_memory"]={{"turns",turns},{"same_context_without_restore_between_turns",true}};save_probe(path,result);
            require(passed,"JSON/memory response failed");
            if(turn==0){evaluate(ctx,{end},pos);const std::string next="\n<|im_start|>user\nWhat number did I ask you to remember? Reply only with a JSON object containing the key code and the integer number, without markdown.<|im_end|>\n"+assistant;evaluate(ctx,tokens(vocab,next,false),pos);}
        }
        result["json_memory"]["passed"]=true;
    }
    result["completed"]=true;save_probe(path,result);return 0;
}catch(const std::exception&e){std::cerr<<"ATTRIBUTION_FAILURE "<<e.what()<<'\n';return 1;}}

// 短decode归因：同步后端split、跨后端copy、同步等待与剩余主机工作。
#define A2O_SPARSE_PROBE_NO_MAIN
#include "sparse_feasibility.cpp"
#include <dlfcn.h>
#include "query_head_probe.h"
#include "spec_oracle.h"
#include "kv_reduce_probe.h"

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
    if(cfg.value("spec_oracle",false)){run_spec_oracle(model,cfg,path);return 0;}
    auto begin=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"a2o_attr_begin"));
    auto finish=reinterpret_cast<const char*(*)()>(dlsym(RTLD_DEFAULT,"a2o_attr_end"));
    bool instrument=cfg.value("host_instrumentation",false);require(!instrument||(begin&&finish),"missing private attribution API");
    J result={{"config",cfg},{"backend",backend},{"rows",J::array()}};double start=now_ms();
    PrefillPrior prior;
    prior.collect_ffn=cfg.value("collect_ffn_prior",true);
    prior.collect_kv=cfg.value("collect_kv_prior",false);
    prior.record_n_kv=cfg.value("record_n_kv",true);
    prior.kv_tail_tokens=cfg.value("kv_tail_tokens",0);
    bool use_prior=cfg.value("prefill_prior",false)||prior.collect_kv;
    bool kv_offload=cfg.value("kv_offload",false),flash_attention=cfg.value("flash_attention",false);
    Engine engine(model,htp,cfg.value("cpu_repack",false),kv_offload,htp?cfg.value("htp_threads",4):6,flash_attention,(use_prior||prior.record_n_kv)?PrefillPrior::callback:nullptr,&prior);auto *ctx=engine.ctx;auto *vocab=llama_model_get_vocab(engine.model);int nv=llama_vocab_n_tokens(vocab);
    result["cpu_repack_requested"]=cfg.value("cpu_repack",false);
    result["kv_offload_requested"]=kv_offload;
    result["flash_attention_requested"]=flash_attention;
    result["model_context_init_ms"]=now_ms()-start;
    auto normal=tokens(vocab,cfg.at("normal_prompt"),true);llama_pos pos=0;evaluate(ctx,normal,pos);
    std::string text;bool eog=false;for(int i=0;i<16;++i){auto tok=greedy(ctx,nv);if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}text+=piece(vocab,tok);evaluate(ctx,{tok},pos);}
    result["normal_response"]={{"text",text},{"passed",eog&&text.find("OK")!=std::string::npos}};save_probe(path,result);require(result["normal_response"]["passed"],"normal response failed");
    auto prefix=cfg.at("prefix_ids").get<std::vector<llama_token>>(),input=cfg.at("input_ids").get<std::vector<llama_token>>(),output=cfg.at("generated_ids").get<std::vector<llama_token>>();
    int needle_begin=-1,needle_end=-1;
    if(cfg.contains("history_tokens")){
        int h=cfg.at("history_tokens");require(h>0&&h<=int(prefix.size()),"invalid diagnostic history length");prefix.resize(h);
        if(cfg.contains("kv_needle")){
            const auto &nd=cfg.at("kv_needle");
            if(nd.contains("filler")){
                auto head=tokens(vocab,nd.at("system_prefix").get<std::string>(),true);
                auto pad=tokens(vocab,nd.at("filler").get<std::string>(),false);
                auto tail=tokens(vocab,nd.value("closing",std::string()),false);
                require(!pad.empty()&&int(head.size()+tail.size())<h,"invalid needle filler lengths");
                prefix=std::move(head);
                while(int(prefix.size()+tail.size())<h)prefix.push_back(pad[prefix.size()%pad.size()]);
                prefix.insert(prefix.end(),tail.begin(),tail.end());
            }
            auto fact=tokens(vocab,nd.at("fact").get<std::string>(),false);
            needle_begin=nd.at("pos").get<int>();needle_end=needle_begin+int(fact.size());
            require(needle_begin>=4&&needle_end<h,"needle range out of history bounds");
            for(size_t i=0;i<fact.size();++i)prefix[size_t(needle_begin)+i]=fact[i];
            input=tokens(vocab,nd.at("question").get<std::string>(),false);
        }
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
    prior.enabled=use_prior;start=now_ms();evaluate(ctx,input,pos);prior.enabled=false;result["full_prefill_ms"]=now_ms()-start;
    llama_token prefill_first_tok=cfg.value("kv_eval_quality",false)?greedy(ctx,nv):-1;
    SessionSnapshot frozen;frozen.state=export_seq(ctx);frozen.next_pos=pos;result["state_bytes"]=frozen.state.size();result["state_restore_verified"]=true;
    if(J kv_info=run_kv_reduction(ctx,frozen,cfg,prior,int(input.size()),needle_begin,needle_end);!kv_info.is_null()){
        result["kv_reduction"]=kv_info;result["state_bytes"]=frozen.state.size();
    }
    double keep=cfg.value("keep_ratio",1.0);require(keep>=0&&keep<=1,"invalid FFN keep ratio");
    CompactProbe pack(engine.model);start=now_ms();pack.install(model,keep,htp,true,use_prior?&prior:nullptr);
    result["ffn_compaction"]={{"keep_ratio",keep},{"packing_ms",now_ms()-start},{"packed_bytes",pack.packed_bytes},{"layers",pack.shapes},
        {"scope","fixed contiguous channels; no importance selection or acceptance claim"}};
    // Prefill使用完整FFN。替换后先用不同于prefill的batch形状重建图，
    // 再恢复原状态；后续单token图可正常复用，避免测到旧dense图。
    QueryHeadProbe query(engine.model);double qkeep=cfg.value("query_keep_ratio",1.0);
    start=now_ms();query.install(model,qkeep,htp,use_prior?&prior:nullptr);
    result["query_compaction"]={{"keep_ratio",qkeep},{"packing_ms",now_ms()-start},{"packed_bytes",query.packed_bytes},{"layers",query.shapes},{"kv_unchanged",true}};
    if(keep!=1||qkeep!=1){
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
            {"backend_timing",attribution},{"llama_eval_ms",perf.t_eval_ms},{"graph_reused",perf.n_reused},{"decode_n_kv",prior.last_n_kv},
            {"logits_vs_first_repeat",compare_probe(reference,values)},{"resources_before",before},{"resources_after",resource_sample()},
            {"cpu_state_before",observed_before},{"cpu_state_after",cfg.value("observe_cpu_state",false)?observe_cpu_state():J(nullptr)}});
        if(warming){warm_ms+=elapsed;if(warm_ms>=1000.*cfg.value("minimum_warmup_seconds",0.))warming=false;}
        result["warmup_elapsed_ms"]=warm_ms;result["warmup_runs"]=warm;save_probe(path,result);
    }
    if(cfg.contains("verify_benchmark_k")){
        J checks=J::array();
        for(int k:cfg.at("verify_benchmark_k").get<std::vector<int>>())for(int rep=-1;rep<3;++rep){
            restore_session(ctx,frozen);pos=frozen.next_pos;auto before=resource_sample();double forward=0,diagnostic=0;
            for(size_t offset=0;offset<output.size();offset+=k){
                int count=std::min(size_t(k),output.size()-offset);auto batch=llama_batch_init(count,0,1);batch.n_tokens=count;
                for(int i=0;i<count;++i){batch.token[i]=output[offset+i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1;}
                double t=now_ms();int rc=llama_decode(ctx,batch);llama_synchronize(ctx);forward+=now_ms()-t;
                require(rc==0,"verify batch decode failed");pos+=count;
                t=now_ms();for(int i=0;i<count;++i){auto *values=llama_get_logits_ith(ctx,i);require(values,"verify missing logits");for(int j=0;j<nv;++j)require(std::isfinite(values[j]),"verify finite");}diagnostic+=now_ms()-t;llama_batch_free(batch);
            }
            checks.push_back({{"k",k},{"repetition",rep},{"warmup",rep<0},{"D",output.size()},{"elapsed_ms",forward},{"diagnostic_ms",diagnostic},{"resources_before",before},{"resources_after",resource_sample()}});
            result["verify_benchmark"]=checks;save_probe(path,result);
        }
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
    if(cfg.contains("acceptance_anchors")){
        auto trajectory=cfg.at("acceptance_reference_ids").get<std::vector<llama_token>>();
        int k=cfg.value("acceptance_k",8);J blocks=J::array();
        for(int anchor:cfg.at("acceptance_anchors").get<std::vector<int>>()){
            require(anchor>=0&&anchor<=int(trajectory.size()),"invalid acceptance anchor");
            restore_session(ctx,frozen);pos=frozen.next_pos;
            for(int i=0;i<anchor;++i)evaluate(ctx,{trajectory[i]},pos);
            J block={{"anchor",anchor}};
            if(cfg.contains("verify_blocks")){
                auto supplied=cfg.at("verify_blocks");auto found=std::find_if(supplied.begin(),supplied.end(),[&](const J &v){return v.at("anchor")==anchor;});require(found!=supplied.end(),"verify anchor missing");
                auto proposals=found->at("draft_ids").get<std::vector<llama_token>>();J target=J::array();int accepted=0;bool matching=true;
                for(auto candidate:proposals){auto prediction=greedy(ctx,nv);target.push_back(prediction);if(matching&&candidate==prediction)++accepted;else matching=false;evaluate(ctx,{candidate},pos);}
                block["draft_ids"]=proposals;block["target_argmax_ids"]=target;block["accepted_prefix"]=accepted;block["proposed"]=proposals.size();
            }else{
                J proposals=J::array();double t=now_ms();
                for(int i=0;i<k;++i){auto tok=greedy(ctx,nv);proposals.push_back(tok);evaluate(ctx,{tok},pos);if(llama_vocab_is_eog(vocab,tok))break;}
                block["draft_ids"]=proposals;block["draft_elapsed_ms"]=now_ms()-t;
            }
            blocks.push_back(block);result["acceptance_blocks"]=blocks;save_probe(path,result);
        }
        result["acceptance_scope"]="greedy speculative prefix acceptance at fixed teacher-forced anchors; CPU verifier must be a separate condition; not sampled rejection acceptance or E2E speedup";
    }
    if(cfg.contains("quality_prompts")){
        J samples=J::array();
        for(const auto &prompt:cfg.at("quality_prompts")){
            llama_memory_clear(llama_get_memory(ctx),true);pos=0;
            auto content=prompt.get<std::string>();evaluate(ctx,tokens(vocab,content,true),pos);
            std::string text;J ids=J::array();bool stopped=false;
            for(int i=0;i<cfg.value("quality_max_tokens",64);++i){auto tok=greedy(ctx,nv);ids.push_back(tok);if(llama_vocab_is_eog(vocab,tok)){stopped=true;break;}text+=piece(vocab,tok);evaluate(ctx,{tok},pos);}
            samples.push_back({{"prompt",content},{"text",text},{"ids",ids},{"eog",stopped}});
            result["quality_samples"]=samples;save_probe(path,result);
        }
    }
    if(cfg.value("kv_eval_quality",false)){
        if(output.size()>=2&&!cfg.contains("kv_needle")){
            restore_session(ctx,frozen);pos=frozen.next_pos;
            J step_nll=J::array(),argmax_ids=J::array();double nll_sum=0.0;
            for(size_t i=0;i+1<output.size();++i){
                evaluate(ctx,{output[i]},pos);auto vals=last_probe(ctx,nv);
                float mx=*std::max_element(vals.begin(),vals.end());double z=0.0;for(float v:vals)z+=std::exp(double(v-mx));
                double nll=std::log(z)-double(vals[size_t(output[i+1])]-mx);
                nll_sum+=nll;step_nll.push_back(nll);argmax_ids.push_back(std::max_element(vals.begin(),vals.end())-vals.begin());
            }
            result["kv_nll"]={{"steps",step_nll},{"mean_nll",nll_sum/double(output.size()-1)},{"argmax_ids",argmax_ids}};
        }
        restore_session(ctx,frozen);pos=frozen.next_pos;
        std::string gen_text;J gen_ids=J::array();bool gen_eog=false;int max_gen=cfg.value("kv_generate_max_tokens",64);
        if(prefill_first_tok>=0){
            gen_ids.push_back(prefill_first_tok);
            if(llama_vocab_is_eog(vocab,prefill_first_tok)){gen_eog=true;}
            else{
                gen_text+=piece(vocab,prefill_first_tok);evaluate(ctx,{prefill_first_tok},pos);
                for(int i=1;i<max_gen;++i){auto tok=greedy(ctx,nv);gen_ids.push_back(tok);if(llama_vocab_is_eog(vocab,tok)){gen_eog=true;break;}gen_text+=piece(vocab,tok);evaluate(ctx,{tok},pos);}
            }
        }
        J gen={{"first_tok",prefill_first_tok},{"text",gen_text},{"ids",gen_ids},{"eog",gen_eog}};
        if(cfg.contains("kv_needle")){
            const auto &nd=cfg.at("kv_needle");
            J parsed=J::parse(gen_text,nullptr,false);
            if(parsed.is_discarded()){auto l=gen_text.find('{'),r=gen_text.rfind('}');if(l!=std::string::npos&&r!=std::string::npos&&r>l)parsed=J::parse(gen_text.substr(l,r-l+1),nullptr,false);}
            bool ok=!parsed.is_discarded()&&parsed.is_object()&&parsed.value("code",-1)==nd.at("expected_code").get<int>()&&parsed.value("codename",std::string())==nd.at("expected_codename").get<std::string>();
            gen["parsed"]=parsed.is_discarded()?J(nullptr):parsed;gen["needle_passed"]=ok;
        }
        result["kv_generation"]=gen;save_probe(path,result);
    }
    if(result.contains("acceptance_blocks"))for(auto &block:result["acceptance_blocks"]){std::string text;for(auto id:block["draft_ids"].get<std::vector<llama_token>>())text+=piece(vocab,id);block["draft_text"]=text;}
    result["completed"]=true;save_probe(path,result);return 0;
}catch(const std::exception&e){std::cerr<<"ATTRIBUTION_FAILURE "<<e.what()<<'\n';return 1;}}

#pragma once
#include <thread>
#include <atomic>
static J run_spec_oracle(const std::string &model,const J &cfg,const std::string &path){
    J result={{"config",cfg},{"rows",J::array()},{"oracle_all_accepted",true},{"future_tokens_known_for_overlap",true}};
    double t=now_ms();PrefillPrior prior;
    Engine draft(model,true,false,false,4,false,PrefillPrior::callback,&prior);
    Engine target(model,false,false,false,6);
    result["dual_init_ms"]=now_ms()-t;
    auto *vocab=llama_model_get_vocab(target.model);int nv=llama_vocab_n_tokens(vocab);
    auto normal=tokens(vocab,cfg.at("normal_prompt"),true);J normal_results=J::array();
    for(auto *ctx:{target.ctx,draft.ctx}){llama_pos pos=0;evaluate(ctx,normal,pos);std::string text;bool eog=false;for(int i=0;i<16;++i){auto tok=greedy(ctx,nv);if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}text+=piece(vocab,tok);evaluate(ctx,{tok},pos);}require(eog&&text.find("OK")!=std::string::npos,"oracle normal response");normal_results.push_back(text);}
    result["normal_response"]={{"passed",true},{"texts",normal_results}};
    auto stored=J::parse(read(cfg.at("history_cache_metadata")));auto meta=stored.at("metadata");
    require(meta.at("prefix_ids")==cfg.at("prefix_ids")&&meta.at("context")==context_size&&meta.at("batch")==batch_size,"oracle history identity");
    auto identity=model+":"+std::to_string(std::filesystem::file_size(model))+":"+std::to_string(static_cast<long long>(std::filesystem::last_write_time(model).time_since_epoch().count()))+":";
    require(meta.at("identity").get<std::string>().find(identity)==0,"oracle model identity");
    std::string bytes=read(cfg.at("history_cache_metadata").get<std::string>()+".state");uint64_t hash=14695981039346656037ULL;for(unsigned char b:bytes){hash^=b;hash*=1099511628211ULL;}std::ostringstream hex;hex<<std::hex<<hash;require(hex.str()==stored.at("state_checksum")&&bytes.size()==stored.at("state_bytes"),"oracle history checksum");
    SessionSnapshot history;history.state.assign(bytes.begin(),bytes.end());history.next_pos=cfg.at("prefix_ids").size();
    auto input=cfg.at("input_ids").get<std::vector<llama_token>>(),output=cfg.at("generated_ids").get<std::vector<llama_token>>();
    SessionSnapshot ts,ds;
    t=now_ms();restore_session(target.ctx,history);llama_pos tp=history.next_pos;evaluate(target.ctx,input,tp);ts.state=export_seq(target.ctx);ts.next_pos=tp;
    restore_session(draft.ctx,history);llama_pos dp=history.next_pos;prior.enabled=true;evaluate(draft.ctx,input,dp);prior.enabled=false;ds.state=export_seq(draft.ctx);ds.next_pos=dp;
    result["prefill_restore_ms"]=now_ms()-t;
    CompactProbe pack(draft.model);QueryHeadProbe query(draft.model);t=now_ms();pack.install(model,.125,true,true,&prior);query.install(model,.25,true,&prior);result["packing_ms"]=now_ms()-t;
    evaluate(draft.ctx,{output[0],output[1]},dp);restore_session(draft.ctx,ds);
    result["state_restore_verified"]=true;save_probe(path,result);
    // 全接受时oracle候选等于约定token流。并行使用已知后续块，忽略因果预测限制，是乐观上界。
    for(int k:cfg.at("oracle_k").get<std::vector<int>>()){
        require(output.size()%k==0,"oracle D multiple k");
        for(const std::string mode:cfg.value("oracle_modes",std::vector<std::string>{"cpu_decode","verify_only","draft_only","serial","overlap"})){
            double warm_ms=0;int formal=0,warm=0;
            while(warm_ms<1000.*cfg.value("oracle_warmup_seconds",5.)||formal<cfg.at("rounds").get<int>()){
                bool warming=warm_ms<1000.*cfg.value("oracle_warmup_seconds",5.);int rep=warming?-(++warm):formal++;
                restore_session(target.ctx,ts);restore_session(draft.ctx,ds);tp=ts.next_pos;dp=ds.next_pos;
                auto before=resource_sample();double cpu_ms=0,draft_ms=0;J blocks=J::array();t=now_ms();
                for(size_t offset=0;offset<output.size();offset+=k){std::vector<llama_token> block(output.begin()+offset,output.begin()+offset+k);double ct=0,dt=0;
                    auto cpu_work=[&](){double b=now_ms();if(mode=="cpu_decode")for(auto tok:block)evaluate(target.ctx,{tok},tp);else {auto logits=block_probe(target.ctx,block,tp,nv);tp+=block.size();for(auto &v:logits)for(float x:v)require(std::isfinite(x),"oracle target logits");}ct=now_ms()-b;};
                    auto draft_work=[&](){double b=now_ms();for(auto tok:block){auto prediction=greedy(draft.ctx,nv);(void)prediction;evaluate(draft.ctx,{tok},dp);}dt=now_ms()-b;};
                    if(mode=="overlap"){
                        std::atomic<int> ready{0};std::atomic<bool> go{false};std::exception_ptr ce,de;
                        std::thread a([&](){++ready;while(!go.load())std::this_thread::yield();try{cpu_work();}catch(...){ce=std::current_exception();}});
                        std::thread b([&](){++ready;while(!go.load())std::this_thread::yield();try{draft_work();}catch(...){de=std::current_exception();}});
                        while(ready.load()!=2)std::this_thread::yield();go=true;a.join();b.join();if(ce)std::rethrow_exception(ce);if(de)std::rethrow_exception(de);
                    }else {if(mode!="draft_only")cpu_work();if(mode=="serial"||mode=="draft_only")draft_work();}
                    cpu_ms+=ct;draft_ms+=dt;blocks.push_back({{"offset",offset},{"cpu_ms",ct},{"draft_ms",dt}});
                }
                double elapsed=now_ms()-t;auto after=resource_sample();
                // 验证序列状态写入完整且可恢复；不同batch浮点不要求逐字节相同。
                if(mode!="draft_only"){require(tp==ts.next_pos+int(output.size()),"target oracle position");auto state=export_seq(target.ctx);require(!state.empty(),"oracle target state");}
                if(mode=="serial"||mode=="overlap"||mode=="draft_only")require(dp==ds.next_pos+int(output.size()),"draft oracle position");
                result["rows"].push_back({{"k",k},{"mode",mode},{"warmup",warming},{"repetition",rep},{"D",output.size()},{"elapsed_ms",elapsed},{"cpu_ms",cpu_ms},{"draft_ms",draft_ms},{"blocks",blocks},{"resources_before",before},{"resources_after",after}});
                if(warming)warm_ms+=elapsed;save_probe(path,result);
            }
        }
    }
    result["completed"]=true;save_probe(path,result);return result;
}

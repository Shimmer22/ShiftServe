// 独立 P/D 扫描；包含在 PhoneService 内，复用完整历史缓存。
J stage_sweep(const J &body) {
    require(cpu && htp,"stage sweep requires explicit CPU and HTP");
    auto prefix=body.at("prefix_ids").get<std::vector<llama_token>>();
    auto input=body.at("input_ids").get<std::vector<llama_token>>();
    auto output=body.at("generated_ids").get<std::vector<llama_token>>();
    auto points=body.at("decode_points").get<std::vector<size_t>>();
    auto order=body.at("backend_order").get<std::vector<std::string>>();
    auto sorted=order;std::sort(sorted.begin(),sorted.end());
    require(sorted==std::vector<std::string>({"cpu","htp"}),"invalid backend order");
    require(!prefix.empty() && !input.empty(),"empty sweep input");
    require(prefix.size()+input.size()+output.size()<size_t(context_size),"sweep overflow");
    require(std::is_sorted(points.begin(),points.end()) &&
            std::adjacent_find(points.begin(),points.end())==points.end(),"invalid decode checkpoints");
    require(output.empty()?points.empty():!points.empty() && points.front()>0 && points.back()==output.size(),"decode shape mismatch");
    lane.close();anchor={};anchor_ids.clear();
    J cache;double boot=now_ms();const char *dir=std::getenv("A2O_REPLAY_CACHE_DIR");
    auto frozen=replay_cache->get(cpu->ctx,prefix,dir?dir:"",cache);
    double bootstrap=now_ms()-boot;
    J rows=J::array();std::vector<float> reference;double cosine=1,nrmse=0;
    for(const auto &backend:order) {
        auto *ctx=backend=="cpu"?cpu->ctx:htp->ctx;
        double start=now_ms();restore_session(ctx,frozen);double restore=now_ms()-start;
        llama_pos pos=prefix.size();start=now_ms();evaluate(ctx,input,pos);double prefill=now_ms()-start;
        auto *logits=llama_get_logits_ith(ctx,-1);require(logits,"missing sweep logits");
        std::vector<float> values(logits,logits+llama_vocab_n_tokens(vocab));
        for(auto value:values)require(std::isfinite(value),"nonfinite sweep logits");
        if(reference.empty())reference=values;
        else {
            double dot=0,aa=0,bb=0,err=0;
            for(size_t i=0;i<values.size();++i){dot+=values[i]*reference[i];aa+=reference[i]*reference[i];bb+=values[i]*values[i];err+=std::pow(values[i]-reference[i],2);}
            require(aa>0 && bb>0,"zero sweep logits");cosine=dot/std::sqrt(aa*bb);nrmse=std::sqrt(err/aa);
        }
        J checkpoints=J::array();size_t next=0;start=now_ms();
        for(size_t i=0;i<output.size();++i) {
            evaluate(ctx,{output[i]},pos);
            if(i+1==points[next]) {
                double elapsed=now_ms()-start;
                // 有限值检查放在记录计时之后；单独记录检查点开销。
                auto *last=llama_get_logits_ith(ctx,-1);require(last,"missing decode logits");
                double check=now_ms();
                for(int j=0;j<llama_vocab_n_tokens(vocab);++j)require(std::isfinite(last[j]),"nonfinite decode logits");
                double overhead=now_ms()-check;
                checkpoints.push_back({{"D",i+1},{"decode_ms",elapsed},{"validation_ms",overhead}});
                start+=now_ms()-check; // 排除检查及 JSON 记录开销。
                ++next;
            }
        }
        rows.push_back({{"backend",backend},{"H",prefix.size()},{"P",input.size()},
            {"D",output.size()},{"n_threads",backend=="cpu"?cpu_threads:4},
            {"n_threads_batch",backend=="cpu"?cpu_threads:4},{"restore_ms",restore},{"prefill_ms",prefill},
            {"decode_checkpoints",checkpoints},{"state_restore_verified",true},{"finite_logits",true}});
    }
    return {{"mode","stage_sweep_v1"},{"bootstrap_ms",bootstrap},{"cache",cache},{"rows",rows},
        {"logit_comparison",{{"cosine",cosine},{"nrmse",nrmse},{"passed",cosine>=0.99}}},
        {"scope","fixed-token stage timing; restore/validation excluded; no sampling or task-quality claim"}};
}

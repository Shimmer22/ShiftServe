#pragma once
// 用 prefill 阶段观察到的 kq_soft_max 重要性直接指导 CPU KV 裁剪与紧凑重排。
static std::vector<double> pool_kv_scores(const std::vector<double> &scores,int begin,int end,int window){
    std::vector<double> out(scores.size(),0.0);int rad=std::max(0,window/2);
    for(int p=begin;p<end;++p){
        int lo=std::max(begin,p-rad),hi=std::min(end-1,p+rad);
        double sum=0.0;for(int q=lo;q<=hi;++q)if(size_t(q)<scores.size())sum+=scores[size_t(q)];
        out[size_t(p)]=sum/double(hi-lo+1);
    }
    return out;
}

// 仅计算保留掩码并在上下文中执行 seq_rm 裁剪；供离线/在线与 HC 回迁前复用。
static J evict_kv_by_prior(llama_context *ctx,int total,const J &cfg,const PrefillPrior &prior,
        int input_len,int needle_begin=-1,int needle_end=-1,std::vector<int> *live_pos=nullptr){
    const std::string policy=cfg.value("kv_reduce_policy","none");
    if(policy=="none")return nullptr;
    require(policy=="recent"||policy=="prefill_prior"||policy=="adaptive_cost","unsupported kv_reduce_policy: "+policy);
    const std::string score_policy=(policy=="adaptive_cost")?"prefill_prior":policy;
    const bool use_live=(live_pos!=nullptr && !live_pos->empty());
    const int n_cells=use_live?int(live_pos->size()):total;
    auto pos_at=[&](int idx)->int{return use_live?(*live_pos)[size_t(idx)]:idx;};
    const int keep_target=cfg.at("kv_keep_tokens");
    const int sink=std::min(n_cells,std::max(cfg.value("kv_sink_tokens",4),cfg.value("kv_pinned_tokens",0)));
    const int recent_min=std::min(n_cells-sink,cfg.value("kv_recent_tokens",std::max(input_len,64)));
    const int pool_window=cfg.value("kv_pool_window",1);

    std::vector<uint8_t> keep(size_t(n_cells),0);
    if(n_cells<=keep_target){
        std::fill(keep.begin(),keep.end(),1);
    }else{
        for(int p=0;p<sink;++p)keep[size_t(p)]=1;
        int recent_start=n_cells-recent_min;
        if(score_policy=="recent"){
            int extra=std::max(0,keep_target-sink-recent_min);
            recent_start=std::max(sink,recent_start-extra);
        }
        for(int p=recent_start;p<n_cells;++p)keep[size_t(p)]=1;
        int budget=std::max(0,keep_target-(sink+(n_cells-recent_start)));
        if(score_policy=="prefill_prior"&&budget>0&&recent_start>sink){
            require(prior.kv_scores.size()>=size_t(recent_start),"missing prefill kq_soft_max prior for history");
            auto smoothed=pool_window>1?pool_kv_scores(prior.kv_scores,sink,recent_start,pool_window):prior.kv_scores;
            std::vector<std::pair<double,int>> ranked;ranked.reserve(size_t(recent_start-sink));
            for(int p=sink;p<recent_start;++p)ranked.push_back({smoothed[size_t(p)],p});
            std::stable_sort(ranked.begin(),ranked.end(),[](const auto &a,const auto &b){return a.first>b.first;});
            int take=std::min(budget,int(ranked.size()));
            for(int i=0;i<take;++i)keep[size_t(ranked[size_t(i)].second)]=1;
        }
    }

    int kept=0,needle_kept=0,needle_total=(needle_begin>=0&&needle_end>needle_begin)?(needle_end-needle_begin):0;
    for(int idx=0;idx<n_cells;++idx){
        if(keep[size_t(idx)]){
            ++kept;
            int lp=pos_at(idx);
            if(lp>=needle_begin&&lp<needle_end)++needle_kept;
        }
    }

    auto *mem=llama_get_memory(ctx);
    double t0=now_ms();
    int rm_spans=0,rm_tokens=0;
    for(int p=0;p<n_cells;){
        if(keep[size_t(p)]){++p;continue;}
        int q=p+1;while(q<n_cells&&!keep[size_t(q)])++q;
        int lo=pos_at(p),hi=pos_at(q-1)+1;
        require(llama_memory_seq_rm(mem,0,llama_pos(lo),llama_pos(hi)),"kv seq_rm failed");
        ++rm_spans;rm_tokens+=(q-p);p=q;
    }
    double evict_ms=now_ms()-t0;
    if(live_pos!=nullptr){
        std::vector<int> next_live;next_live.reserve(size_t(kept));
        for(int idx=0;idx<n_cells;++idx)if(keep[size_t(idx)])next_live.push_back(pos_at(idx));
        *live_pos=std::move(next_live);
    }

    J info={{"policy",policy},{"keep_target",keep_target},{"sink_tokens",sink},{"recent_tokens",recent_min},
        {"pool_window",pool_window},{"before_pos_count",n_cells},{"kept_count",kept},
        {"evicted_count",rm_tokens},{"rm_spans",rm_spans},{"evict_ms",evict_ms},
        {"max_pos_preserved",llama_memory_seq_pos_max(mem,0)}};
    if(needle_total>0){
        info["needle_range"]={needle_begin,needle_end};
        info["needle_kept"]=needle_kept;
        info["needle_total"]=needle_total;
        info["needle_recall_ratio"]=double(needle_kept)/double(needle_total);
    }
    return info;
}

static J run_kv_reduction(llama_context *ctx,SessionSnapshot &frozen,const J &cfg,const PrefillPrior &prior,
        int input_len,int needle_begin=-1,int needle_end=-1,std::vector<int> *live_pos=nullptr){
    J info=evict_kv_by_prior(ctx,int(frozen.next_pos),cfg,prior,input_len,needle_begin,needle_end,live_pos);
    if(info.is_null())return nullptr;
    const bool do_compact=cfg.value("kv_compact",true);
    const size_t bytes_before=frozen.state.size();
    auto *mem=llama_get_memory(ctx);

    double t1=now_ms();
    if(do_compact){
        auto compacted=export_seq(ctx);
        llama_memory_clear(mem,false);
        require(llama_state_seq_set_data(ctx,compacted.data(),compacted.size(),0)==compacted.size(),"compact KV restore failed");
        llama_synchronize(ctx);
        frozen.state=std::move(compacted);
    }else{
        frozen.state=export_seq(ctx);
    }
    double compact_ms=now_ms()-t1;

    info["compacted"]=do_compact;
    info["compact_ms"]=compact_ms;
    info["state_bytes_before"]=bytes_before;
    info["state_bytes_after"]=frozen.state.size();
    info["max_pos_preserved"]=llama_memory_seq_pos_max(mem,0);
    return info;
}

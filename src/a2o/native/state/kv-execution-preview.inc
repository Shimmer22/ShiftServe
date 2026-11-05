// 编入私有 llama-kv-cache.cpp；只计算 apply 后的视图宽度，不提交任何 KV 元数据/内容。
uint32_t llama_kv_cache::a2o_preview_n_kv(const slot_info & sinfo) const {
    if(n_swa || n_stream!=1 || get_has_shift() || sinfo.n_stream()!=1)
        throw std::runtime_error("execution preview requires unshifted, non-SWA, single-stream KV");
    const auto &cells=v_cells[sinfo.strm[0]];
    uint32_t high=cells.used_max_p1();
    for(uint32_t index:sinfo.idxs[0]) {
        if(index>=cells.size() || !cells.is_empty(index))
            throw std::runtime_error("execution preview would overwrite a live KV cell");
        high=std::max(high,index+1);
    }
    const uint32_t pad=std::max(n_pad,256u);
    return std::min(cells.size(),std::max(pad,(uint32_t)GGML_PAD(high,pad)));
}
void llama_kv_cache_context::a2o_preview_graph() {
    if(ubatches.size()!=1 || i_cur!=0 || sinfos.size()!=1)
        throw std::runtime_error("execution preview supports exactly one microbatch");
    n_kv=kv->a2o_preview_n_kv(sinfos[0]);
}

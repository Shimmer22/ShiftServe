// 在诊断构建中用同一份外部 KV 快照比较两种 CPU 权重布局。
SessionSnapshot prefill_ab_snapshot;
std::string prefill_ab_checksum;

J prefill_weight_ab(const J &body) {
    const bool run_htp=condition=="htp";
    require(!cpu_repack && !kv_reduce &&
            (run_htp ? bool(htp) : (cpu && (shared_weights ? condition=="route" : condition=="cpu"))),
            "expected a supported prefill diagnostic arm");
    const auto &prefix=body.at("prefix_ids");
    const auto ids=body.at("input_ids").get<std::vector<llama_token>>();
    const auto expected=body.at("snapshot_checksum").get<std::string>();
    require(prefix.size()==4096 && !ids.empty() && prefix.size()+ids.size()<size_t(context_size),
            "invalid frozen prefill shape");
    if(prefill_ab_snapshot.state.empty()) {
        const char *directory=std::getenv("A2O_REPLAY_CACHE_DIR");
        require(directory && *directory,"prefill A/B requires a staged snapshot");
        const auto file=std::filesystem::path(directory)/"prefill-ab.json";
        const auto stored=J::parse(read(file.string()));
        const auto &metadata=stored.at("metadata");
        require(metadata.at("format")==2 && metadata.at("context")==context_size &&
                metadata.at("batch")==batch_size && metadata.at("prefix_ids")==prefix,
                "A/B snapshot metadata mismatch");
        const auto statefile=file.string()+".state";
        const auto size=stored.at("state_bytes").get<size_t>();
        require(size>0 && std::filesystem::file_size(statefile)==size,"A/B snapshot size mismatch");
        SessionSnapshot next;
        next.state.resize(size);
        std::ifstream stream(statefile,std::ios::binary);
        require(bool(stream.read(reinterpret_cast<char *>(next.state.data()),size)),"A/B snapshot read failed");
        uint64_t hash=14695981039346656037ULL;
        for(auto byte:next.state){hash^=byte;hash*=1099511628211ULL;}
        std::ostringstream digest;
        digest<<std::hex<<hash;
        require(digest.str()==expected && stored.at("state_checksum")==expected,"A/B snapshot checksum mismatch");
        next.next_pos=prefix.size();
        next.evaluated_tokens=prefix.size();
        prefill_ab_snapshot=std::move(next);
        prefill_ab_checksum=expected;
        lane.close();
        if(htp && !run_htp)htp->release_ctx();
    }
    require(prefill_ab_checksum==expected,"A/B snapshot changed during a block");
    auto *ctx=(run_htp?htp:cpu)->ensure_ctx();
    require(llama_n_threads(ctx)==4 && llama_n_threads_batch(ctx)==4,"A/B host threads mismatch");
    restore_session(ctx,prefill_ab_snapshot);
    const auto before=resource_sample();
    const auto clocks_before=phone_clock_sample();
    auto pos=prefill_ab_snapshot.next_pos;
    const double start=now_ms();
    evaluate(ctx,ids,pos);
    const double elapsed=now_ms()-start;
    const auto clocks_after=phone_clock_sample();
    const auto after=resource_sample();
    const auto *logits=llama_get_logits_ith(ctx,-1);
    require(logits,"missing A/B logits");
    const auto vocab_size=llama_vocab_n_tokens(vocab);
    require(std::all_of(logits,logits+vocab_size,[](float x){return std::isfinite(x);}),"nonfinite A/B logits");
    return {{"mode","cpu_prefill_ab_v1"},{"backend",run_htp?"htp":"cpu"},{"shared_weights",shared_weights},
            {"H",prefix.size()},{"P",ids.size()},{"n_threads",llama_n_threads(ctx)},
            {"n_threads_batch",llama_n_threads_batch(ctx)},{"snapshot_checksum",prefill_ab_checksum},
            {"prefill_ms",elapsed},{"restore_verified",true},{"top_token",std::max_element(logits,logits+vocab_size)-logits},
            {"resources_before",before},{"resources_after",after},{"clocks_before",clocks_before},{"clocks_after",clocks_after}};
}

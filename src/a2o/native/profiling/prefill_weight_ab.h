// 在诊断构建中用同一份外部 KV 快照比较独立 CPU、共享 CPU、独立 HTP 与 M(H) 交接开销。
std::map<std::string, SessionSnapshot> prefill_ab_snapshots;

static std::string prefill_ab_fnv1a(const std::vector<uint8_t> &bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream digest;
    digest << std::hex << hash;
    return digest.str();
}

J prefill_weight_ab(const J &body) {
    const std::string snapshot_name = body.value("snapshot_name", std::string("prefill-ab.json"));
    const auto &prefix = body.at("prefix_ids");
    const auto prefix_vec = prefix.get<std::vector<llama_token>>();
    require(!prefix_vec.empty() && prefix_vec.size() < size_t(context_size), "invalid prefix length");

    if (body.value("prepare_snapshot", false)) {
        require(!shared_weights && condition == "cpu" && cpu, "prepare_snapshot requires independent cpu service");
        const char *directory = std::getenv("A2O_REPLAY_CACHE_DIR");
        require(directory && *directory, "prepare_snapshot requires replay cache dir");
        std::filesystem::create_directories(directory);
        const auto file = std::filesystem::path(directory) / snapshot_name;
        const auto statefile = file.string() + ".state";
        if (std::filesystem::exists(file) && std::filesystem::exists(statefile)) {
            const auto stored = J::parse(read(file.string()));
            const auto &metadata = stored.at("metadata");
            const size_t size = stored.at("state_bytes").get<size_t>();
            if (metadata.value("format", 0) == 2 && metadata.value("context", 0) == context_size
                    && metadata.value("batch", 0) == batch_size && metadata.at("prefix_ids") == prefix
                    && size > 0 && std::filesystem::file_size(statefile) == size) {
                SessionSnapshot next;
                next.state.resize(size);
                std::ifstream stream(statefile, std::ios::binary);
                require(bool(stream.read(reinterpret_cast<char *>(next.state.data()), size)), "snapshot read failed");
                const auto digest = prefill_ab_fnv1a(next.state);
                require(digest == stored.at("state_checksum").get<std::string>(), "existing snapshot checksum mismatch");
                next.next_pos = prefix_vec.size();
                next.evaluated_tokens = prefix_vec.size();
                prefill_ab_snapshots[digest] = std::move(next);
                return {{"mode", "cpu_prefill_ab_prepare_v1"}, {"reused", true}, {"H", prefix_vec.size()},
                        {"snapshot_name", snapshot_name}, {"snapshot_checksum", digest}, {"state_bytes", size}};
            }
        }
        auto *ctx = cpu->ensure_ctx();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos pos = 0;
        evaluate(ctx, prefix_vec, pos);
        SessionSnapshot next;
        next.state = export_seq(ctx);
        next.next_pos = prefix_vec.size();
        next.evaluated_tokens = prefix_vec.size();
        const auto digest = prefill_ab_fnv1a(next.state);
        const J metadata = {{"format", 2}, {"context", context_size}, {"batch", batch_size}, {"prefix_ids", prefix}};
        const auto tmp_state = statefile + ".tmp";
        {
            std::ofstream out_state(tmp_state, std::ios::binary);
            out_state.write(reinterpret_cast<const char *>(next.state.data()), next.state.size());
            require(bool(out_state), "snapshot state write failed");
        }
        std::filesystem::rename(tmp_state, statefile);
        const auto tmp_meta = file.string() + ".tmp";
        {
            std::ofstream out_meta(tmp_meta);
            out_meta << J{{"metadata", metadata}, {"state_bytes", next.state.size()}, {"state_checksum", digest}}.dump();
            require(bool(out_meta), "snapshot meta write failed");
        }
        std::filesystem::rename(tmp_meta, file);
        const size_t size = next.state.size();
        prefill_ab_snapshots[digest] = std::move(next);
        return {{"mode", "cpu_prefill_ab_prepare_v1"}, {"reused", false}, {"H", prefix_vec.size()},
                {"snapshot_name", snapshot_name}, {"snapshot_checksum", digest}, {"state_bytes", size}};
    }

    const bool measure_handoff = body.value("measure_handoff", false);
    const bool handoff_only_h = body.value("handoff_only_h", false);
    const bool run_htp = measure_handoff || condition == "htp";
    require(!cpu_repack && !kv_reduce, "prefill diagnostic requires cpu_repack=0 and kv_reduce=0");
    if (measure_handoff) {
        require(condition == "route" && shared_weights && cpu && htp, "measure_handoff requires shared route service");
    } else if (run_htp) {
        require(htp && condition == "htp", "independent htp arm requires condition=htp");
    } else {
        require(cpu && (shared_weights ? condition == "route" : condition == "cpu"),
                "expected a supported prefill cpu diagnostic arm");
    }

    const auto ids = body.value("input_ids", std::vector<llama_token>{});
    const auto expected = body.at("snapshot_checksum").get<std::string>();
    require((handoff_only_h ? ids.empty() : !ids.empty()) && prefix_vec.size() + ids.size() < size_t(context_size),
            "invalid frozen prefill shape");

    if (prefill_ab_snapshots.find(expected) == prefill_ab_snapshots.end()) {
        const char *directory = std::getenv("A2O_REPLAY_CACHE_DIR");
        require(directory && *directory, "prefill A/B requires a staged snapshot");
        const auto file = std::filesystem::path(directory) / snapshot_name;
        const auto stored = J::parse(read(file.string()));
        const auto &metadata = stored.at("metadata");
        require(metadata.at("format") == 2 && metadata.at("context") == context_size &&
                metadata.at("batch") == batch_size && metadata.at("prefix_ids") == prefix,
                "A/B snapshot metadata mismatch");
        const auto statefile = file.string() + ".state";
        const auto size = stored.at("state_bytes").get<size_t>();
        require(size > 0 && std::filesystem::file_size(statefile) == size, "A/B snapshot size mismatch");
        SessionSnapshot next;
        next.state.resize(size);
        std::ifstream stream(statefile, std::ios::binary);
        require(bool(stream.read(reinterpret_cast<char *>(next.state.data()), size)), "A/B snapshot read failed");
        const auto digest = prefill_ab_fnv1a(next.state);
        require(digest == expected && stored.at("state_checksum") == expected, "A/B snapshot checksum mismatch");
        next.next_pos = prefix_vec.size();
        next.evaluated_tokens = prefix_vec.size();
        prefill_ab_snapshots[expected] = std::move(next);
    }

    const auto &snap = prefill_ab_snapshots.at(expected);
    require(snap.next_pos == prefix_vec.size(), "cached snapshot H mismatch");
    lane.close();
    if (htp && !run_htp) htp->release_ctx();

    auto *ctx = (run_htp ? htp : cpu)->ensure_ctx();
    const int base_threads = llama_n_threads(ctx);
    const int base_batch_threads = llama_n_threads_batch(ctx);
    const int req_threads = body.value("cpu_prefill_threads", base_batch_threads);
    require(req_threads >= 1 && req_threads <= 16, "invalid cpu_prefill_threads");
    const bool change_threads = !run_htp && (req_threads != base_threads || req_threads != base_batch_threads);
    if (change_threads) llama_set_n_threads(ctx, req_threads, req_threads);
    const int active_threads = llama_n_threads(ctx);
    const int active_batch_threads = llama_n_threads_batch(ctx);

    const double r0 = now_ms();
    restore_session(ctx, snap);
    const double restore_ms = now_ms() - r0;

    const auto before = resource_sample();
    const auto clocks_before = phone_clock_sample();
    auto pos = snap.next_pos;
    double elapsed = 0.0;
    if (!ids.empty()) {
        const double start = now_ms();
        try {
            evaluate(ctx, ids, pos);
        } catch (...) {
            if (change_threads) llama_set_n_threads(ctx, base_threads, base_batch_threads);
            throw;
        }
        elapsed = now_ms() - start;
    }
    if (change_threads) llama_set_n_threads(ctx, base_threads, base_batch_threads);

    const auto clocks_after = phone_clock_sample();
    long top_token = -1;
    if (!ids.empty()) {
        const auto *logits = llama_get_logits_ith(ctx, -1);
        require(logits, "missing A/B logits");
        const auto vocab_size = llama_vocab_n_tokens(vocab);
        require(std::all_of(logits, logits + vocab_size, [](float x) { return std::isfinite(x); }), "nonfinite A/B logits");
        top_token = std::max_element(logits, logits + vocab_size) - logits;
    }

    double handoff_ms = 0.0;
    size_t handoff_bytes = 0;
    if (measure_handoff) {
        auto *cpu_ctx = cpu->ensure_ctx();
        llama_memory_clear(llama_get_memory(cpu_ctx), true);
        const bool verify_hnd = body.value("verify_handoff_bytes", false);
        const double h0 = now_ms();
        handoff_bytes = transfer(ctx, cpu_ctx, verify_hnd ? KvVerification::full_bytes : KvVerification::import_size);
        handoff_ms = now_ms() - h0;
        htp->release_ctx();
    }
    const auto after = resource_sample();

    return {{"mode", "cpu_prefill_ab_v1"}, {"backend", run_htp ? "htp" : "cpu"}, {"shared_weights", shared_weights},
            {"measure_handoff", measure_handoff}, {"H", prefix_vec.size()}, {"P", ids.size()},
            {"n_threads", active_threads}, {"n_threads_batch", active_batch_threads},
            {"snapshot_checksum", expected}, {"restore_ms", restore_ms}, {"prefill_ms", elapsed},
            {"handoff_ms", handoff_ms}, {"handoff_bytes", handoff_bytes}, {"restore_verified", true},
            {"top_token", top_token}, {"resources_before", before}, {"resources_after", after},
            {"clocks_before", clocks_before}, {"clocks_after", clocks_after}};
}

#pragma once
// 完整 sequence 的导出、迁移与可选字节核验。
enum class KvVerification { import_size, full_bytes };

static std::vector<uint8_t> export_seq(llama_context * ctx) {
    llama_synchronize(ctx);
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    require(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size(), "seq export failed");
    return state;
}

template<class Checkpoint>
static size_t import_seq(llama_context * to, const std::vector<uint8_t> & state,
                         KvVerification verification, llama_pos expected_max_pos, Checkpoint checkpoint) {
    require(llama_state_seq_set_data(to, state.data(), state.size(), 0) == state.size(), "seq import failed");
    if (verification == KvVerification::full_bytes) {
        const auto readback = export_seq(to);
        checkpoint(); // 两份序列化缓冲区同时存活。
        require(readback == state, "KV/position metadata changed during transfer");
        std::cerr << "TRANSFER verified_bytes=" << state.size() << '\n';
    } else {
        llama_synchronize(to);
        require(llama_memory_seq_pos_max(llama_get_memory(to),0)==expected_max_pos,
                "KV last position changed during transfer");
        checkpoint();
    }
    return state.size();
}

static size_t transfer(llama_context * from, llama_context * to,
                       KvVerification verification = KvVerification::full_bytes) {
    const auto state = export_seq(from);
    const auto max_pos = verification == KvVerification::import_size
        ? llama_memory_seq_pos_max(llama_get_memory(from),0) : -1;
    return import_seq(to, state, verification, max_pos, [] {});
}

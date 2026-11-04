#pragma once
// 完整 sequence 的导出、迁移与字节核验。
static std::vector<uint8_t> export_seq(llama_context * ctx) {
    llama_synchronize(ctx);
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    require(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size(), "seq export failed");
    return state;
}
static size_t transfer(llama_context * from, llama_context * to) {
    auto state = export_seq(from);
    std::cerr << "TRANSFER exported_bytes=" << state.size() << " importing\n";
    require(llama_state_seq_set_data(to, state.data(), state.size(), 0) == state.size(), "seq import failed");
    std::cerr << "TRANSFER imported; synchronizing and verifying\n";
    llama_synchronize(to);
    require(export_seq(to) == state, "KV/position metadata changed during transfer");
    return state.size();
}

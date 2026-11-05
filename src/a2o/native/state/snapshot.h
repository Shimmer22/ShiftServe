#pragma once
// 快照同时保存序列、模板前缀、位置与版本。
struct SessionSnapshot {
    std::vector<uint8_t> state;
    std::string rendered_prefix;
    llama_pos next_pos=0;
    int evaluated_tokens=0;
    uint64_t version=0;
};

static size_t restore_session(llama_context * ctx, const SessionSnapshot & snapshot,double * work_ms=nullptr) {
    llama_synchronize(ctx);
    const double begin=work_ms?now_ms():0;
    llama_memory_clear(llama_get_memory(ctx),true);
    if(snapshot.state.empty()) {if(work_ms) *work_ms=now_ms()-begin;return 0;}
    require(llama_state_seq_set_data(ctx,snapshot.state.data(),snapshot.state.size(),0)==snapshot.state.size(),"session restore failed");
    if(work_ms) *work_ms=now_ms()-begin; // 不含下面的调试核验。
    require(export_seq(ctx)==snapshot.state,"session restore changed KV/position state");
    return snapshot.state.size();
}

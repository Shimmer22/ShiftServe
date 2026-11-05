#pragma once
#include <stdint.h>

// 私有、版本化的 host ABI。仅准备首个 microbatch，不传入未知 Tool 内容。
#define A2O_EXEC_ABI 1u
enum a2o_exec_action { A2O_EXEC_PREPARE=1, A2O_EXEC_BIND=2, A2O_EXEC_QUERY=3, A2O_EXEC_DISCARD=4 };
enum a2o_exec_status { A2O_EXEC_EMPTY=0, A2O_EXEC_READY=1, A2O_EXEC_BOUND=2, A2O_EXEC_CONSUMED=3, A2O_EXEC_MISS=4, A2O_EXEC_FAILED=5 };
enum a2o_exec_flags { A2O_EXEC_DESCRIPTORS=1, A2O_EXEC_MAP=2, A2O_EXEC_PREFAULT=4 };
struct a2o_execution_request {
    uint32_t abi, size;
    uint64_t version;                 // CPU 权威快照版本，返回后必须一致。
    int64_t next_pos;
    uint32_t n_tokens, flags;          // 首个 microbatch 的精确形状，不是逻辑 P 上限。
    uint64_t descriptor_budget, map_budget, prefault_budget;
};
struct a2o_execution_stats {
    uint32_t abi, size, status, reserved;
    uint64_t version, graph_uid, graph_nodes, htp_nodes, cpu_nodes;
    uint64_t descriptor_batches, descriptor_ops, descriptor_bytes;
    uint64_t mapped_buffers, mapped_bytes, prefault_bytes;
    uint64_t graph_hits, descriptor_hits, descriptor_misses;
    double graph_ms, allocate_ms, backend_ms, bind_ms;
    char reason[96];
};
struct llama_context;
typedef int (*a2o_execution_control_fn)(struct llama_context *,uint32_t,
    const struct a2o_execution_request *,struct a2o_execution_stats *);
// scheduler 通过调用方回调解析 backend 扩展，避免 ggml-base 反向依赖 registry 所在的 libggml。
typedef int (*a2o_backend_prepare_callback)(void *,void *,uint32_t,
    const struct a2o_execution_request *,struct a2o_execution_stats *);

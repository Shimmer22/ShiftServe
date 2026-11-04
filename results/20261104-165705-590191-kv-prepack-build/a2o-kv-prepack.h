// 私有实验协议；只支持本实验 Qwen2.5-VL-3B、F16、单序列、H 为 32 倍数。
// 使用现有 profiler RPC 的保留 mode；不修改 wire ABI，不用于 PMU 配置。
#ifndef A2O_KV_PREPACK_H
#define A2O_KV_PREPACK_H
#define A2O_PREPACK_MODE 0x41324f50u
#define A2O_MAX_KV 80
struct a2o_kv_entry {
    uint32_t fd, offset, capacity, history, kind;
    uint16_t * tiles;
    size_t bytes;
};
struct a2o_kv_cache {
    struct a2o_kv_entry entries[A2O_MAX_KV];
    uint32_t count;
    uint64_t hits[2], bytes[2], packed_bytes;
};
// 私有批处理共享缓冲：[count, validate, descriptors...]；固定 uint32 字段。
struct a2o_kv_descriptor {
    uint32_t fd, size, offset, kind, capacity, history;
};
struct a2o_kv_pending {
    struct a2o_kv_entry *entry;
    __fp16 *dst;
    void *scratch;
    size_t k, ncols, hist, reused;
    unsigned requests;
};
bool a2o_enqueue_kv_tiles(struct htp_context *ctx, const __fp16 *weight,
                         int k, int stride, size_t nc, size_t ncols,
                         __fp16 *dst, void *scratch, struct a2o_kv_pending *pending);
void a2o_finish_kv_tiles(struct htp_context *ctx, struct a2o_kv_pending *pending);
// 返回 true 表示实际读取缓存；形状或地址不匹配返回 false，调用方须释放 HMX 锁。
bool a2o_load_kv_tiles(struct htp_context *ctx, const __fp16 *weight,
                      int k, int stride, size_t nc, size_t ncols,
                      __fp16 *dst, void *scratch);
#endif

// 准备 RPC 使用独立、短生命周期的 VTCM 和 DMA 队列；不借用仍在轮询的算子上下文。
#include "hmx-utils.h"
#define A2O_PACK_BLOCK_BYTES (64u*1024u)
struct a2o_pack_workspace { unsigned rctx; uint8_t *base; dma_queue *dma; };
static void a2o_pack_workspace_close(struct a2o_pack_workspace *w) {
    if(w->dma) {dma_queue_flush(w->dma);dma_queue_delete(w->dma);}
    if(w->rctx) HAP_compute_res_release(w->rctx);
    memset(w,0,sizeof(*w));
}
static int a2o_pack_workspace_open(struct a2o_pack_workspace *w) {
    memset(w,0,sizeof(*w));
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr,0);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr,256*1024,256*1024,256*1024);
    w->rctx=HAP_compute_res_acquire(&attr,1000000u);
    unsigned size=0;void *base=NULL;
    if(!w->rctx) return AEE_ENOMEMORY;
    if(HAP_compute_res_attr_get_vtcm_ptr_v2(&attr,&base,&size)!=0 || !base || size<3*A2O_PACK_BLOCK_BYTES+128) {
        a2o_pack_workspace_close(w);return AEE_ENOMEMORY;
    }
    w->base=(uint8_t*)base;w->dma=dma_queue_create(16);
    if(!w->dma) {a2o_pack_workspace_close(w);return AEE_ENOMEMORY;}
    return AEE_SUCCESS;
}
static void a2o_pack_scatter_sync(void *fence) {
    // scatter_release + 同地址依赖读取，保证 HVX scatter 完成后才让 DMA 读取 VTCM。
    // Qualcomm HVX Programmer's Reference: Memory / scatter synchronization。
    asm volatile("vmem(%0 + #0):scatter_release" : : "r"(fence) : "memory");
    HVX_Vector completed=*(volatile HVX_Vector*)fence;
    asm volatile("" : : "v"(completed) : "memory");
}

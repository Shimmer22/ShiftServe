#include "a2o-kv-prepack.h"

bool a2o_load_kv_tiles(struct htp_context *ctx,const __fp16 *weight,int k,int stride,
                     size_t nc,size_t ncols,__fp16 *dst,void *scratch) {
    struct a2o_kv_cache *c=ctx->a2o_cache;
    if(!c || !ncols || nc%32 || ncols%32) return false;
    for(uint32_t i=0;i<c->count;i++) {
        struct a2o_kv_entry *e=&c->entries[i];
        for(uint32_t mi=0;mi<HTP_MAX_MMAPS;mi++) {
            struct htp_mmap *m=&ctx->mmap[mi];
            if(!m->size || m->fd!=e->fd) continue;
            for(size_t head=0;head<2;head++) {
                size_t head_offset=e->kind?head*128*e->capacity*2:head*128*2;
                if((uintptr_t)weight!=m->base+e->offset+head_offset) continue;
                const __fp16 *tiles=(const __fp16*)e->tiles+head*e->history*128;
                size_t reused=0;
                if(!e->kind) {
                    if(k!=128 || stride!=256 || nc+ncols>e->capacity) return false;
                    size_t hist=nc<e->history?hex_smin(ncols,e->history-nc):0;
                    if(hist) {
                        reused=hist*k*2;
                        dma_queue_push(ctx->dma[0],dma_make_ptr(dst,tiles+nc*k),reused,reused,reused,1);
                        dma_queue_pop(ctx->dma[0]);
                    }
                    if(hist<ncols) {
                        size_t bytes=k*2;
                        dma_queue_push(ctx->dma[0],dma_make_ptr(scratch,weight+(nc+hist)*stride),bytes,stride*2,bytes,ncols-hist);
                        dma_queue_pop(ctx->dma[0]);
                        // hist 为 32 倍数，因此尾部可以直接接在历史 tile 后。
                        hmx_interleave_rows_to_tiles(dst+hist*k,scratch,ncols-hist,k,k,0,ncols-hist);
                    }
                } else {
                    if(stride!=(int)e->capacity || k<(int)e->history || k>(int)e->capacity || k%32 || nc+ncols>128) return false;
                    size_t h=e->history,tail=k-h;
                    const __fp16 *src=tiles+nc*h;
                    // 每个 N tile 的历史 K tiles 连续；目标 K stride 包含新增/填充部分。
                    dma_queue_push(ctx->dma[0],dma_make_ptr(dst,src),k*32*2,h*32*2,h*32*2,ncols/32);
                    dma_queue_pop(ctx->dma[0]);reused=ncols*h*2;
                    if(tail) {
                        dma_queue_push(ctx->dma[0],dma_make_ptr(scratch,weight+nc*stride+h),tail*2,stride*2,tail*2,ncols);
                        dma_queue_pop(ctx->dma[0]);
                        for(size_t row=0;row<ncols;row+=32)
                            hmx_interleave_rows_to_tiles(dst+row*k+h*32,(const __fp16*)scratch+row*tail,32,tail,tail,0,32);
                    }
                }
                c->hits[e->kind]++;c->bytes[e->kind]+=reused;return true;
            }
        }
    }
    return false;
}

#include "a2o-kv-prepack.h"

bool a2o_enqueue_kv_tiles(struct htp_context *ctx,const __fp16 *weight,int k,int stride,
                        size_t nc,size_t ncols,__fp16 *dst,void *scratch,
                        struct a2o_kv_pending *p) {
    memset(p,0,sizeof(*p));
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
                p->entry=e;p->dst=dst;p->scratch=scratch;p->k=k;p->ncols=ncols;
                if(!e->kind) {
                    if(k!=128 || stride!=256 || nc>e->capacity || ncols>e->capacity-nc) return false;
                    p->hist=nc<e->history?hex_smin(ncols,e->history-nc):0;
                    if(p->hist) {
                        p->reused=p->hist*k*2;
                        dma_queue_push(ctx->dma[0],dma_make_ptr(dst,tiles+nc*k),p->reused,p->reused,p->reused,1);
                        p->requests++;
                    }
                    if(p->hist<ncols) {
                        size_t bytes=k*2;
                        dma_queue_push(ctx->dma[0],dma_make_ptr(scratch,weight+(nc+p->hist)*stride),bytes,stride*2,bytes,ncols-p->hist);
                        p->requests++;
                    }
                } else {
                    if(stride!=(int)e->capacity || k<(int)e->history || k>(int)e->capacity || k%32 || nc>128 || ncols>128-nc) return false;
                    p->hist=e->history;
                    size_t tail=k-p->hist;
                    dma_queue_push(ctx->dma[0],dma_make_ptr(dst,tiles+nc*p->hist),k*32*2,p->hist*32*2,p->hist*32*2,ncols/32);
                    p->requests++;p->reused=ncols*p->hist*2;
                    if(tail) {
                        dma_queue_push(ctx->dma[0],dma_make_ptr(scratch,weight+nc*stride+p->hist),tail*2,stride*2,tail*2,ncols);
                        p->requests++;
                    }
                }
                return true;
            }
        }
    }
    return false;
}

void a2o_finish_kv_tiles(struct htp_context *ctx,struct a2o_kv_pending *p) {
    for(unsigned i=0;i<p->requests;i++) dma_queue_pop(ctx->dma[0]);
    if(!p->entry->kind) {
        if(p->hist<p->ncols)
            hmx_interleave_rows_to_tiles(p->dst+p->hist*p->k,p->scratch,p->ncols-p->hist,p->k,p->k,0,p->ncols-p->hist);
    } else {
        size_t tail=p->k-p->hist;
        if(tail) for(size_t row=0;row<p->ncols;row+=32)
            hmx_interleave_rows_to_tiles(p->dst+row*p->k+p->hist*32,(const __fp16*)p->scratch+row*tail,32,tail,tail,0,32);
    }
    ctx->a2o_cache->hits[p->entry->kind]++;
    ctx->a2o_cache->bytes[p->entry->kind]+=p->reused;
    p->requests=0;
}

bool a2o_load_kv_tiles(struct htp_context *ctx,const __fp16 *weight,int k,int stride,
                     size_t nc,size_t ncols,__fp16 *dst,void *scratch) {
    struct a2o_kv_pending p;
    if(!a2o_enqueue_kv_tiles(ctx,weight,k,stride,nc,ncols,dst,scratch,&p)) return false;
    a2o_finish_kv_tiles(ctx,&p);return true;
}

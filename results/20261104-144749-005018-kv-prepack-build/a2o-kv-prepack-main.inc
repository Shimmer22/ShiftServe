#include <stdlib.h>
#include "a2o-kv-prepack.h"

static size_t a2o_tile_index(size_t n, size_t k, size_t nk) {
    return ((n / 32) * nk + k / 32) * 1024 + (k % 32 / 2) * 64 + (n % 32) * 2 + k % 2;
}

static void a2o_cache_clear(struct htp_context *ctx) {
    if (!ctx->a2o_cache) return;
    for (uint32_t i=0;i<ctx->a2o_cache->count;i++) free(ctx->a2o_cache->entries[i].tiles);
    free(ctx->a2o_cache); ctx->a2o_cache=NULL;
}

static int a2o_prepack_rpc(struct htp_context *ctx, const uint32_t *a) {
    if (a[0]==0) { a2o_cache_clear(ctx); return AEE_SUCCESS; }
    if (!ctx->a2o_cache) ctx->a2o_cache=calloc(1,sizeof(struct a2o_kv_cache));
    struct a2o_kv_cache *c=ctx->a2o_cache;
    if (!c) return AEE_ENOMEMORY;
    if (a[0]!=1 && a[0]!=2) return AEE_EBADPARM;
    void *base=HAP_mmap2(NULL,a[2],HAP_PROT_READ|HAP_PROT_WRITE,0,a[1],0);
    if (base==(void*)-1) return AEE_EBADPARM;
    int ret=AEE_SUCCESS;
    if (a[0]==2) {
        uint64_t stats[6]={c->hits[0],c->hits[1],c->bytes[0],c->bytes[1],c->packed_bytes,c->count};
        if(a[2]<sizeof(stats)) ret=AEE_EBADPARM;
        else {memcpy(base,stats,sizeof(stats));hex_l2flush(base,sizeof(stats));}
    } else {
        // a: operation, fd, mapped size, tensor offset, kind K=0/V=1, capacity, H, validate
        const uint32_t cap=a[5],h=a[6],kind=a[4];
        const size_t bytes=(size_t)h*256*2;
        if(kind>1 || !h || h%32 || cap<h || cap>16384 || c->count>=A2O_MAX_KV ||
           (uint64_t)a[3]+(uint64_t)cap*256*2>a[2]) ret=AEE_EBADPARM;
        else {
            struct a2o_kv_entry *e=&c->entries[c->count];
            e->fd=a[1];e->offset=a[3];e->capacity=cap;e->history=h;e->kind=kind;e->bytes=bytes;
            if(posix_memalign((void**)&e->tiles,128,bytes)) ret=AEE_ENOMEMORY;
            else {
                const uint16_t *src=(const uint16_t*)((uint8_t*)base+a[3]);
                // 精确复制 F16 位模式，无量化或浮点运算；每个 KV head 单独存 tiles。
                for(size_t head=0;head<2;head++) {
                    const size_t n=kind?128:h,k=kind?h:128;
                    uint16_t *dst=e->tiles+head*h*128;
                    for(size_t r=0;r<n;r++) for(size_t col=0;col<k;col++) {
                        size_t ix=kind?(head*128+r)*cap+col:r*256+head*128+col;
                        dst[a2o_tile_index(r,col,k/32)]=src[ix];
                    }
                    if(a[7]) for(size_t r=0;r<n;r++) for(size_t col=0;col<k;col++) {
                        size_t ix=kind?(head*128+r)*cap+col:r*256+head*128+col;
                        if(dst[a2o_tile_index(r,col,k/32)]!=src[ix]) ret=AEE_EFAILED;
                    }
                }
                hex_l2flush(e->tiles,bytes);
                if(ret==AEE_SUCCESS){c->count++;c->packed_bytes+=bytes;}
                else {free(e->tiles);e->tiles=NULL;}
            }
        }
    }
    HAP_munmap2(base,a[2]);return ret;
}

#include <stdlib.h>
#include "a2o-kv-prepack.h"
#include "a2o-kv-prepack-workspace.inc"
#include "a2o-kv-prepack-pack.inc"

static size_t a2o_tile_index(size_t n,size_t k,size_t nk) {
    return ((n/32)*nk+k/32)*1024+(k%32/2)*64+(n%32)*2+k%2;
}
static void a2o_cache_clear(struct htp_context *ctx) {
    if(!ctx->a2o_cache) return;
    for(uint32_t i=0;i<ctx->a2o_cache->count;i++) free(ctx->a2o_cache->entries[i].tiles);
    free(ctx->a2o_cache);ctx->a2o_cache=NULL;
}
static int a2o_prepare_entry(struct a2o_kv_cache *c,struct a2o_pack_workspace *w,
                             const struct a2o_kv_descriptor *d,int validate) {
    const uint32_t cap=d->capacity,h=d->history,kind=d->kind;
    if(kind>1 || !h || h%32 || cap<h || cap>16384 || c->count>=A2O_MAX_KV ||
       d->offset%128 || (uint64_t)d->offset+(uint64_t)cap*256*2>d->size) return AEE_EBADPARM;
    // 同一普通 KV 只能存在一份缓存，避免指针匹配到旧项。
    for(uint32_t i=0;i<c->count;i++) if(c->entries[i].fd==d->fd && c->entries[i].offset==d->offset) return AEE_EBADPARM;
    void *base=HAP_mmap2(NULL,d->size,HAP_PROT_READ|HAP_PROT_WRITE,0,d->fd,0);
    if(base==(void*)-1) return AEE_EBADPARM;
    struct a2o_kv_entry *e=&c->entries[c->count];
    e->fd=d->fd;e->offset=d->offset;e->capacity=cap;e->history=h;e->kind=kind;e->bytes=(size_t)h*256*2;
    int ret=AEE_SUCCESS;
    if(posix_memalign((void**)&e->tiles,128,e->bytes)) ret=AEE_ENOMEMORY;
    else {
        const uint16_t *src=(const uint16_t*)((uint8_t*)base+d->offset);
        ret=a2o_pack_tensor(w,e,src);
        if(ret==AEE_SUCCESS && validate) {
            for(size_t head=0;head<2;head++) {
                size_t n=kind?128:h,k=kind?h:128;
                const uint16_t *dst=e->tiles+head*h*128;
                for(size_t r=0;r<n;r++) for(size_t col=0;col<k;col++) {
                    size_t ix=kind?(head*128+r)*cap+col:r*256+head*128+col;
                    if(dst[a2o_tile_index(r,col,k/32)]!=src[ix]) ret=AEE_EFAILED;
                }
            }
        }
        if(ret==AEE_SUCCESS) {c->count++;c->packed_bytes+=e->bytes;}
        else {free(e->tiles);e->tiles=NULL;}
    }
    HAP_munmap2(base,d->size);return ret;
}
static int a2o_prepack_rpc(struct htp_context *ctx,const uint32_t *a) {
    if(a[0]==0) {a2o_cache_clear(ctx);return AEE_SUCCESS;}
    if(a[0]!=1 && a[0]!=2 && a[0]!=3) return AEE_EBADPARM;
    if(!ctx->a2o_cache) ctx->a2o_cache=calloc(1,sizeof(struct a2o_kv_cache));
    struct a2o_kv_cache *c=ctx->a2o_cache;
    if(!c) return AEE_ENOMEMORY;
    if(a[0]==1) {
        struct a2o_kv_descriptor d={a[1],a[2],a[3],a[4],a[5],a[6]};
        struct a2o_pack_workspace w;int ret=a2o_pack_workspace_open(&w);
        if(ret==AEE_SUCCESS) ret=a2o_prepare_entry(c,&w,&d,a[7]);
        a2o_pack_workspace_close(&w);
        if(ret!=AEE_SUCCESS) a2o_cache_clear(ctx);
        return ret;
    }
    void *base=HAP_mmap2(NULL,a[2],HAP_PROT_READ|HAP_PROT_WRITE,0,a[1],0);
    if(base==(void*)-1) return AEE_EBADPARM;
    int ret=AEE_SUCCESS;
    if(a[0]==2) {
        uint64_t stats[6]={c->hits[0],c->hits[1],c->bytes[0],c->bytes[1],c->packed_bytes,c->count};
        if(a[2]<sizeof(stats)) ret=AEE_EBADPARM;
        else {memcpy(base,stats,sizeof(stats));hex_l2flush(base,sizeof(stats));}
    } else {
        uint32_t *header=(uint32_t*)base;
        if(a[2]<8 || !header[0] || header[0]>A2O_MAX_KV || 8+(uint64_t)header[0]*sizeof(struct a2o_kv_descriptor)>a[2] || c->count) ret=AEE_EBADPARM;
        else {
            struct a2o_pack_workspace w;ret=a2o_pack_workspace_open(&w);
            const struct a2o_kv_descriptor *d=(const struct a2o_kv_descriptor*)(header+2);
            for(uint32_t i=0;ret==AEE_SUCCESS && i<header[0];i++) ret=a2o_prepare_entry(c,&w,&d[i],header[1]);
            a2o_pack_workspace_close(&w);
        }
    }
    HAP_munmap2(base,a[2]);
    if(ret!=AEE_SUCCESS) a2o_cache_clear(ctx);
    return ret;
}

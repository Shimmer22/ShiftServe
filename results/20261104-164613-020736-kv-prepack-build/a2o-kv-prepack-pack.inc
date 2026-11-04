// K 每块最多 [256,128]，V 每块最多 [32,1024]；所有块恰好覆盖完整 32x32 tiles。
struct a2o_pack_block {size_t head, rt, kt, rows, cols;};
static struct a2o_pack_block a2o_pack_block_at(const struct a2o_kv_entry *e,size_t index) {
    size_t n=e->kind?128:e->history,k=e->kind?e->history:128;
    size_t rn=e->kind?32:256,ck=e->kind?1024:128;
    size_t nr=(n+rn-1)/rn,nk=(k+ck-1)/ck;
    struct a2o_pack_block b;
    b.head=index/(nr*nk);b.rt=(index/nk%nr)*rn;b.kt=(index%nk)*ck;
    b.rows=hex_smin(rn,n-b.rt);b.cols=hex_smin(ck,k-b.kt);return b;
}
static void a2o_pack_input(struct a2o_pack_workspace *w,const struct a2o_kv_entry *e,
                           const uint16_t *src,struct a2o_pack_block b,void *dst) {
    size_t stride=e->kind?e->capacity:256;
    size_t offset=e->kind?(b.head*128+b.rt)*stride+b.kt:b.rt*stride+b.head*128+b.kt;
    dma_queue_push(w->dma,dma_make_ptr(dst,src+offset),b.cols*2,stride*2,b.cols*2,b.rows);
}
static int a2o_pack_tensor(struct a2o_pack_workspace *w,struct a2o_kv_entry *e,const uint16_t *src) {
    size_t n=e->kind?128:e->history,k=e->kind?e->history:128;
    size_t total=2*((n+(e->kind?31:255))/(e->kind?32:256))*((k+(e->kind?1023:127))/(e->kind?1024:128));
    __fp16 *input[2]={(__fp16*)w->base,(__fp16*)(w->base+A2O_PACK_BLOCK_BYTES)};
    __fp16 *output=(__fp16*)(w->base+2*A2O_PACK_BLOCK_BYTES);
    void *fence=w->base+3*A2O_PACK_BLOCK_BYTES;
    // DMA 绕过 L2；先 clean+invalidate 新分配目标，防止旧脏 cache line 覆盖 DMA 结果。
    hex_l2flush(e->tiles,e->bytes);
    a2o_pack_input(w,e,src,a2o_pack_block_at(e,0),input[0]);
    for(size_t i=0;i<total;i++) {
        dma_queue_pop(w->dma); // 当前输入完成。
        if(i) dma_queue_pop(w->dma); // 上一输出完成，才复用 output。
        struct a2o_pack_block b=a2o_pack_block_at(e,i);
        if(i+1<total) a2o_pack_input(w,e,src,a2o_pack_block_at(e,i+1),input[(i+1)%2]);
        hmx_interleave_rows_to_tiles(output,input[i%2],b.rows,b.cols,b.cols,0,b.rows);
        a2o_pack_scatter_sync(fence);
        uint16_t *dst=e->tiles+b.head*e->history*128+b.rt*k+b.kt*32;
        dma_queue_push(w->dma,dma_make_ptr(dst,output),k*32*2,b.cols*32*2,b.cols*32*2,b.rows/32);
    }
    dma_queue_pop(w->dma);return AEE_SUCCESS;
}

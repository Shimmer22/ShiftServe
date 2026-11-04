// 在 host 上运行实际缓存读取代码；DMA mock 逐行检查地址，并用独立 tile 公式核验输出。
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <deque>

#define HTP_MAX_MMAPS 1
struct htp_mmap { uintptr_t base; size_t size; uint32_t fd; };
struct htp_context { struct a2o_kv_cache *a2o_cache; htp_mmap mmap[1]; struct dma_queue *dma[1]; };
struct Region { uintptr_t start; size_t size; };
static std::vector<Region> regions;
static bool valid(const void *p,size_t n) {
    auto a=(uintptr_t)p;
    for(auto r:regions) if(a>=r.start && n<=r.size && a-r.start<=r.size-n) return true;
    return false;
}
struct dma_ptr { void *dst; const void *src; };
static dma_ptr dma_make_ptr(void *d,const void *s) {return {d,s};}
struct Transfer {dma_ptr ptr;size_t ds,ss,row,n;std::vector<uint8_t> snapshot;};
struct dma_queue {std::deque<Transfer> pending;};
static size_t pushes=0;
static void dma_queue_push(dma_queue *q,dma_ptr p,size_t ds,size_t ss,size_t row,size_t n) {
    Transfer t{p,ds,ss,row,n,{}};
    for(size_t i=0;i<n;++i) {
        auto d=(uint8_t*)p.dst+i*ds;auto s=(const uint8_t*)p.src+i*ss;
        assert(valid(d,row) && valid(s,row));t.snapshot.insert(t.snapshot.end(),s,s+row);
    }
    q->pending.push_back(std::move(t));++pushes;
}
static void dma_queue_pop(dma_queue *q) {
    assert(!q->pending.empty());auto t=q->pending.front();q->pending.pop_front();
    for(size_t i=0;i<t.n;++i) {
        auto s=(const uint8_t*)t.ptr.src+i*t.ss;
        // DMA 尚未完成前不得覆盖 source；延迟到 pop 才复制，防止同步 mock 掩盖错误。
        assert(!memcmp(s,t.snapshot.data()+i*t.row,t.row));
        memcpy((uint8_t*)t.ptr.dst+i*t.ds,s,t.row);
    }
}
static void dma_queue_flush(dma_queue *q) {while(!q->pending.empty()) dma_queue_pop(q);}
static size_t hex_smin(size_t a,size_t b) {return std::min(a,b);}
// 独立参考：按物理 tile、K pair、N 行的顺序遍历。
static void hmx_interleave_rows_to_tiles(__fp16 *dst,const void *src,int n,int k,int stride,int first,int last) {
    auto in=(const uint16_t*)src;auto out=(uint16_t*)dst;
    for(int nt=first/32;nt<(last+31)/32;++nt) for(int kt=0;kt<k/32;++kt)
        for(int pair=0;pair<16;++pair) for(int r=0;r<32;++r) for(int h=0;h<2;++h) {
            size_t at=((nt*(k/32)+kt)*1024)+pair*64+r*2+h;
            assert(valid(out+at,2));out[at]=(nt*32+r<n)?in[(nt*32+r)*stride+kt*32+pair*2+h]:0;
        }
}
#define A2O_PACK_BLOCK_BYTES (64u*1024u)
struct a2o_pack_workspace {unsigned rctx;uint8_t *base;dma_queue *dma;};
static int mock_posix_memalign(void **p,size_t align,size_t bytes) {
    int rc=posix_memalign(p,align,bytes);
    if(!rc) regions.push_back({(uintptr_t)*p,bytes});return rc;
}
#define posix_memalign mock_posix_memalign
static int a2o_pack_workspace_open(a2o_pack_workspace *w) {
    *w={};void *base=nullptr;
    assert(!posix_memalign(&base,128,256*1024));w->base=(uint8_t*)base;w->dma=new dma_queue;return 0;
}
static void a2o_pack_workspace_close(a2o_pack_workspace *w) {
    if(w->dma) {dma_queue_flush(w->dma);delete w->dma;}
    free(w->base);*w={};
}
static void a2o_pack_scatter_sync(void *) {}
static void *mapped,*batch_mapped;
enum { AEE_SUCCESS=0,AEE_ENOMEMORY=1,AEE_EBADPARM=2,AEE_EFAILED=3,HAP_PROT_READ=1,HAP_PROT_WRITE=2 };
static void *HAP_mmap2(void *,size_t,int,int,uint32_t fd,int) {return fd==2?batch_mapped:mapped;}
static void HAP_munmap2(void *,size_t) {}
static void hex_l2flush(void *,size_t) {}
#include "a2o-kv-prepack.h"
static a2o_kv_cache *mock_calloc(size_t n,size_t size) {return (a2o_kv_cache*)std::calloc(n,size);}
#define calloc mock_calloc
#include "state/htp-kv-prepack-main.inc"
#undef calloc
#include "state/htp-kv-prepack-load.inc"

static void run(int kind,int history,int capacity,int k,int total_n,int chunk) {
    std::vector<uint16_t> ordinary(capacity*256);
    for(size_t i=0;i<ordinary.size();++i) ordinary[i]=(uint16_t)((i*271+17) ^ (i>>8)*7919 ^ (i>>16)*31337);
    regions={{(uintptr_t)ordinary.data(),ordinary.size()*2}};
    dma_queue queue;
    htp_context ctx={};ctx.dma[0]=&queue;ctx.mmap[0]={(uintptr_t)ordinary.data(),ordinary.size()*2,1};
    mapped=ordinary.data();
    uint32_t args[]={1,1,(uint32_t)(ordinary.size()*2),0,(uint32_t)kind,(uint32_t)capacity,(uint32_t)history,1};
    assert(a2o_prepack_rpc(&ctx,args)==AEE_SUCCESS);
    auto &entry=ctx.a2o_cache->entries[0];
    auto *packed=entry.tiles;
    for(int head=0;head<2;++head) for(int nc=0;nc<total_n;nc+=chunk) {
        int n=std::min(chunk,total_n-nc);
        std::vector<uint16_t> dst(n*k,0xffff),scratch(n*k,0xffff),expected(n*k,0xffff);
        auto weight=ordinary.data()+(kind?head*128*capacity:head*128);
        int stride=kind?capacity:256;
        regions={{(uintptr_t)ordinary.data(),ordinary.size()*2},{(uintptr_t)packed,entry.bytes},
                 {(uintptr_t)dst.data(),dst.size()*2},{(uintptr_t)scratch.data(),scratch.size()*2},
                 {(uintptr_t)expected.data(),expected.size()*2}};
        assert(a2o_load_kv_tiles(&ctx,(const __fp16*)weight,k,stride,nc,n,(__fp16*)dst.data(),scratch.data()));
        hmx_interleave_rows_to_tiles((__fp16*)expected.data(),weight+nc*stride,n,k,stride,0,n);
        assert(dst==expected);
        assert(queue.pending.empty());
        // 两个 tile buffers + 一份尾部 scratch，镜像生产流水：next 在当前计算期间 pending。
        if(nc+chunk<total_n) {
            int next_n=std::min(chunk,total_n-nc-chunk);
            std::vector<uint16_t> next(next_n*k,0xffff),next_ref(next_n*k,0xffff);
            regions.push_back({(uintptr_t)next.data(),next.size()*2});
            regions.push_back({(uintptr_t)next_ref.data(),next_ref.size()*2});
            a2o_kv_pending pending;
            assert(a2o_enqueue_kv_tiles(&ctx,(const __fp16*)weight,k,stride,nc+chunk,next_n,(__fp16*)next.data(),scratch.data(),&pending));
            assert(!queue.pending.empty());assert(dst==expected);
            a2o_finish_kv_tiles(&ctx,&pending);
            hmx_interleave_rows_to_tiles((__fp16*)next_ref.data(),weight+(nc+chunk)*stride,next_n,k,stride,0,next_n);
            assert(next==next_ref && queue.pending.empty());
        }
        // 非法范围必须在提交 DMA 前拒绝。
        assert(!a2o_load_kv_tiles(&ctx,(const __fp16*)weight,k,stride,nc+1,n,(__fp16*)dst.data(),scratch.data()));
        assert(!a2o_load_kv_tiles(&ctx,(const __fp16*)weight,kind?capacity+32:127,stride,nc,n,(__fp16*)dst.data(),scratch.data()));
    }
    // 完整消费流水，复用两份 tile buffer 和单独尾部 scratch；两个 GQA group 重置指针。
    for(int group=0;group<2;++group) {
        std::vector<uint16_t> first(chunk*k,0xbeef),second(chunk*k,0xbeef),tail(chunk*k,0xbeef);
        regions={{(uintptr_t)ordinary.data(),ordinary.size()*2},{(uintptr_t)packed,entry.bytes},
                 {(uintptr_t)first.data(),first.size()*2},{(uintptr_t)second.data(),second.size()*2},
                 {(uintptr_t)tail.data(),tail.size()*2}};
        auto weight=ordinary.data()+(kind?group*128*capacity:group*128);
        int stride=kind?capacity:256;
        auto current=(__fp16*)first.data();void *other=second.data();
        a2o_kv_pending pending;
        assert(a2o_enqueue_kv_tiles(&ctx,(const __fp16*)weight,k,stride,0,std::min(chunk,total_n),current,tail.data(),&pending));
        for(int nc=0;nc<total_n;nc+=chunk) {
            if(nc) {void *old=current;current=(__fp16*)other;other=old;}
            a2o_finish_kv_tiles(&ctx,&pending);
            int n=std::min(chunk,total_n-nc),next=nc+chunk;
            if(next<total_n) assert(a2o_enqueue_kv_tiles(&ctx,(const __fp16*)weight,k,stride,next,std::min(chunk,total_n-next),(__fp16*)other,tail.data(),&pending));
            std::vector<uint16_t> expected(n*k);
            regions.push_back({(uintptr_t)expected.data(),expected.size()*2});
            hmx_interleave_rows_to_tiles((__fp16*)expected.data(),weight+nc*stride,n,k,stride,0,n);
            assert(!memcmp(current,expected.data(),expected.size()*2));
        }
        assert(queue.pending.empty());
    }
    a2o_cache_clear(&ctx);
    assert(ctx.a2o_cache==nullptr);
    uint16_t dummy[32*128]={};
    assert(!a2o_load_kv_tiles(&ctx,(const __fp16*)ordinary.data(),128,256,0,32,(__fp16*)dummy,dummy));
}
static void batch_rpc_test() {
    const int cap=2048,h=1056;
    std::vector<uint16_t> ordinary(cap*256*2);
    for(size_t i=0;i<ordinary.size();i++) ordinary[i]=(uint16_t)((i*271+17)^(i>>8)*7919);
    regions={{(uintptr_t)ordinary.data(),ordinary.size()*2}};
    mapped=ordinary.data();htp_context ctx={};
    struct {uint32_t count,validate;a2o_kv_descriptor d[2];} batch={2,1,{
        {1,(uint32_t)(ordinary.size()*2),0,0,cap,h},
        {1,(uint32_t)(ordinary.size()*2),cap*256*2,1,cap,h}}};
    batch_mapped=&batch;
    uint32_t args[]={3,2,sizeof(batch),0,0,0,0,0};
    assert(a2o_prepack_rpc(&ctx,args)==AEE_SUCCESS);
    assert(ctx.a2o_cache->count==2 && ctx.a2o_cache->packed_bytes==(size_t)h*256*2*2);
    a2o_cache_clear(&ctx);
    batch.d[1].capacity=h-32;
    assert(a2o_prepack_rpc(&ctx,args)==AEE_EBADPARM && ctx.a2o_cache==nullptr);
    batch.count=100;
    assert(a2o_prepack_rpc(&ctx,args)==AEE_EBADPARM && ctx.a2o_cache==nullptr);
    args[2]=4;
    assert(a2o_prepack_rpc(&ctx,args)==AEE_EBADPARM && ctx.a2o_cache==nullptr);
}
int main() {
    batch_rpc_test();
    for(int h:{64,96,1056,2048,4096,4128}) for(int tail:{32,64,256}) {
        int cap=h+512;
        for(int chunk:{32,64,128,512}) {
            run(0,h,cap,128,h+tail,chunk);
            run(1,h,cap,h+tail,128,chunk);
        }
    }
}

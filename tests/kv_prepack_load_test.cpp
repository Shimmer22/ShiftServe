// 在 host 上运行实际缓存读取代码；DMA mock 逐行检查地址，并用独立 tile 公式核验输出。
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define HTP_MAX_MMAPS 1
struct htp_mmap { uintptr_t base; size_t size; uint32_t fd; };
struct htp_context { struct a2o_kv_cache *a2o_cache; htp_mmap mmap[1]; void *dma[1]; };
struct Region { uintptr_t start; size_t size; };
static std::vector<Region> regions;
static bool valid(const void *p,size_t n) {
    auto a=(uintptr_t)p;
    for(auto r:regions) if(a>=r.start && n<=r.size && a-r.start<=r.size-n) return true;
    return false;
}
struct dma_ptr { void *dst; const void *src; };
static dma_ptr dma_make_ptr(void *d,const void *s) {return {d,s};}
static void dma_queue_push(void *,dma_ptr p,size_t ds,size_t ss,size_t row,size_t n) {
    for(size_t i=0;i<n;++i) {
        auto d=(uint8_t*)p.dst+i*ds;auto s=(const uint8_t*)p.src+i*ss;
        assert(valid(d,row) && valid(s,row));memcpy(d,s,row);
    }
}
static void dma_queue_pop(void *) {}
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
static void *mapped;
enum { AEE_SUCCESS=0,AEE_ENOMEMORY=1,AEE_EBADPARM=2,AEE_EFAILED=3,HAP_PROT_READ=1,HAP_PROT_WRITE=2 };
static void *HAP_mmap2(void *,size_t,int,int,uint32_t,int) {return mapped;}
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
    for(size_t i=0;i<ordinary.size();++i) ordinary[i]=(uint16_t)(i*271+17);
    htp_context ctx={};ctx.mmap[0]={(uintptr_t)ordinary.data(),ordinary.size()*2,1};
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
        // 非法范围必须在提交 DMA 前拒绝。
        assert(!a2o_load_kv_tiles(&ctx,(const __fp16*)weight,k,stride,nc+1,n,(__fp16*)dst.data(),scratch.data()));
        assert(!a2o_load_kv_tiles(&ctx,(const __fp16*)weight,kind?capacity+32:127,stride,nc,n,(__fp16*)dst.data(),scratch.data()));
    }
    a2o_cache_clear(&ctx);
    assert(ctx.a2o_cache==nullptr);
    uint16_t dummy[32*128]={};
    assert(!a2o_load_kv_tiles(&ctx,(const __fp16*)ordinary.data(),128,256,0,32,(__fp16*)dummy,dummy));
}
int main() {
    for(int h:{64,2048,4096}) for(int tail:{32,64,256}) {
        int cap=h+512;
        for(int chunk:{32,64,128,512}) {
            run(0,h,cap,128,h+tail,chunk);
            run(1,h,cap,h+tail,128,chunk);
        }
    }
}

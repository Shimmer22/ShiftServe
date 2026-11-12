// 设备内补齐验证：外部31行、尾部哨兵、与主机32行参考逐元素比较。
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using J=nlohmann::ordered_json;
static double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
static ggml_context *context(){auto *c=ggml_init({2*1024*1024,nullptr,true});check(c,"context init");return c;}
struct Graph {
    ggml_context *ctx;ggml_tensor *x,*y,*guard;ggml_cgraph *graph;ggml_backend_buffer_t buffer;
    Graph(ggml_tensor *w,int rows,ggml_backend_t backend):ctx(context()){
        x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w->ne[0],rows);ggml_set_name(x,"fixed_input");ggml_set_input(x);
        y=ggml_mul_mat(ctx,w,x);ggml_set_name(y,"retained_projection");ggml_set_output(y);
        graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,y);
        // 将哨兵紧接在有效输出之后；多写补行会立刻破坏哨兵。
        guard=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,128);
        auto bt=ggml_backend_get_default_buffer_type(backend);size_t align=ggml_backend_buft_get_alignment(bt);
        auto round=[&](size_t size){return (size+align-1)/align*align;};
        size_t xb=round(ggml_backend_buft_get_alloc_size(bt,x)),yb=round(ggml_backend_buft_get_alloc_size(bt,y));
        check(yb==ggml_nbytes(y),"output guard must immediately follow output");
        buffer=ggml_backend_alloc_buffer(backend,xb+yb+round(ggml_backend_buft_get_alloc_size(bt,guard)));check(buffer,"graph allocation");
        auto *base=static_cast<char*>(ggml_backend_buffer_get_base(buffer));
        check(ggml_backend_tensor_alloc(buffer,x,base)==GGML_STATUS_SUCCESS,"input alloc");
        check(ggml_backend_tensor_alloc(buffer,y,base+xb)==GGML_STATUS_SUCCESS,"output alloc");
        check(ggml_backend_tensor_alloc(buffer,guard,base+xb+yb)==GGML_STATUS_SUCCESS,"guard alloc");
        std::vector<float> sentinel(128,12345.5f);ggml_backend_tensor_set(guard,sentinel.data(),0,sentinel.size()*sizeof(float));
    }
    ~Graph(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
};
static J compare(const std::vector<float> &a,const std::vector<float> &b){
    check(a.size()==b.size(),"comparison size");double aa=0,bb=0,ab=0,err=0,maxerr=0;size_t bits=0,close=0;
    for(size_t i=0;i<a.size();++i){check(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite output");
        double d=double(b[i])-a[i];aa+=double(a[i])*a[i];bb+=double(b[i])*b[i];ab+=double(a[i])*b[i];err+=d*d;maxerr=std::max(maxerr,std::abs(d));
        bits+=std::memcmp(&a[i],&b[i],sizeof(float))==0;close+=std::abs(d)<=1e-5+1e-5*std::abs(a[i]);}
    return {{"elements",a.size()},{"bitwise_equal",bits==a.size()},{"bitwise_equal_elements",bits},
        {"allclose_atol_1e_5_rtol_1e_5",close==a.size()},{"allclose_elements",close},
        {"max_abs",maxerr},{"nrmse",std::sqrt(err/std::max(aa,1e-30))},{"cosine",ab/std::sqrt(std::max(aa*bb,1e-30))}};
}
int main(int argc,char **argv){try{
    check(argc==3,"usage: htp_small_m MODEL OUTPUT_JSON");ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_name("HTP0");check(dev,"HTP0 unavailable");
    auto htp=ggml_backend_dev_init(dev,nullptr);auto cpu=ggml_backend_cpu_init();check(htp&&cpu,"explicit backend init");ggml_backend_cpu_set_n_threads(cpu,6);
    auto extra=reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_backend_dev_get_extra_bufts"));
    check(extra,"missing HTP repack buffer hook");auto bufts=extra(dev);check(bufts&&bufts[0],"HTP weight buffer unavailable");
    ggml_context *meta=nullptr;auto *gguf=gguf_init_from_file(argv[1],{true,&meta});check(gguf&&meta,"GGUF metadata");
    J results=J::array();std::mt19937 rng(20261119);std::uniform_real_distribution<float> dist(-1,1);
    for(const char *name:{"blk.0.attn_qkv.weight","blk.0.ffn_up.weight","blk.0.ffn_down.weight"}){
        auto *source=ggml_get_tensor(meta,name);check(source&&source->type==GGML_TYPE_Q4_0,"expected real Q4_0 tensor");
        const int64_t k=source->ne[0],n=source->ne[1];auto id=gguf_find_tensor(gguf,name);
        std::vector<uint8_t> quant(ggml_nbytes(source));std::ifstream input(argv[1],std::ios::binary);
        input.seekg(gguf_get_data_offset(gguf)+gguf_get_tensor_offset(gguf,id));input.read(reinterpret_cast<char*>(quant.data()),quant.size());check(bool(input),"GGUF weight read");
        auto *wc=context();auto *w=ggml_new_tensor_2d(wc,GGML_TYPE_Q4_0,k,n);ggml_set_name(w,name);
        auto wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,bufts[0]);check(wb,"HTP weight allocation");ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_tensor_set(w,quant.data(),0,quant.size());
        std::vector<float> x31(k*31),x32(k*32,0);for(auto &v:x31)v=dist(rng);std::copy(x31.begin(),x31.end(),x32.begin());
        Graph plain(w,31,htp),padded(w,32,htp);check(ggml_backend_dev_supports_op(dev,plain.y)&&ggml_backend_dev_supports_op(dev,padded.y),"HTP op unsupported; no fallback");
        // FP32参考使用同一Q4权重的精确解量化值，单独CPU计算，不计入HTP时间。
        std::vector<float> wf(k*n);ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(quant.data(),wf.data(),k*n);
        auto *rc=context();auto *rw=ggml_new_tensor_2d(rc,GGML_TYPE_F32,k,n);auto rb=ggml_backend_alloc_ctx_tensors(rc,cpu);check(rb,"reference weight allocation");ggml_backend_tensor_set(rw,wf.data(),0,wf.size()*sizeof(float));
        Graph ref(rw,31,cpu);ggml_backend_tensor_set(ref.x,x31.data(),0,x31.size()*sizeof(float));check(ggml_backend_graph_compute(cpu,ref.graph)==GGML_STATUS_SUCCESS,"CPU reference compute");
        std::vector<float> reference(n*31);ggml_backend_tensor_get(ref.y,reference.data(),0,reference.size()*sizeof(float));
        std::vector<float> a(n*31),b(n*32),sentinel(128);J trials=J::array();
        auto run=[&](bool pad){Graph &g=pad?padded:plain;double total=now_ms();
            if(pad){std::copy(x31.begin(),x31.end(),x32.begin());std::fill(x32.begin()+k*31,x32.end(),0);}
            auto &x=pad?x32:x31;ggml_backend_tensor_set(g.x,x.data(),0,x.size()*sizeof(float));double setup=now_ms()-total,start=now_ms();
            check(ggml_backend_graph_compute(htp,g.graph)==GGML_STATUS_SUCCESS,"HTP compute failed");ggml_backend_synchronize(htp);double compute=now_ms()-start;
            start=now_ms();auto &output=pad?b:a;ggml_backend_tensor_get(g.y,output.data(),0,output.size()*sizeof(float));
            double read=now_ms()-start,elapsed=now_ms()-total;
            ggml_backend_tensor_get(g.guard,sentinel.data(),0,sentinel.size()*sizeof(float));
            check(std::all_of(sentinel.begin(),sentinel.end(),[](float v){return v==12345.5f;}),"output guard overwritten");
            return J{{"condition",pad?"pad32_discard_last":"plain31"},{"setup_upload_ms",setup},{"compute_ms",compute},{"read_crop_ms",read},{"total_ms",elapsed}};
        };
        run(false);run(true); // 两条件独立预热，不进入正式十次。
        auto retained=[&](){return std::vector<float>(b.begin(),b.begin()+n*31);};
        J numerical=compare(a,retained()),plain_ref=compare(reference,a),padded_ref=compare(reference,retained());
        std::vector<int> orders={0,1,0,1,0,1,0,1,0,1};std::shuffle(orders.begin(),orders.end(),rng);
        for(int rep=0;rep<10;++rep){J pair=J::array();bool first_pad=orders[rep];pair.push_back(run(first_pad));pair.push_back(run(!first_pad));
            trials.push_back({{"repetition",rep},{"results",pair},{"retained_output_comparison",compare(a,retained())}});}
        for(auto entry:{std::pair<const char*,std::vector<float>*>("plain31",&a),{"pad32",&b}}){
            std::string path=std::string(argv[2])+"."+name+"."+entry.first+".f32";
            std::ofstream dump(path,std::ios::binary);dump.write(reinterpret_cast<const char*>(entry.second->data()),n*31*sizeof(float));check(bool(dump),"output dump failed");
        }
        results.push_back({{"tensor",name},{"output_guard_passed",true},{"input_bytes_plain31",k*31*sizeof(float)},{"input_bytes_pad32",k*32*sizeof(float)},
            {"output_bytes_plain31",n*31*sizeof(float)},{"output_bytes_pad32",n*32*sizeof(float)},{"K",k},{"N",n},{"real_rows",31},{"padded_rows",32},{"trials",trials},
            {"retained_output_comparison",numerical},{"plain31_vs_fp32_reference",plain_ref},{"pad32_vs_fp32_reference",padded_ref}});
        {std::ofstream output(argv[2]);output<<J{{"scope","device-side small-M microbenchmark; reused host buffers, guarded31 vs host32; not full model prefill/state equivalence"},{"cases",results}}.dump(2);}
        ggml_backend_buffer_free(rb);ggml_free(rc);ggml_backend_buffer_free(wb);ggml_free(wc);
    }
    gguf_free(gguf);ggml_free(meta);ggml_backend_free(cpu);ggml_backend_free(htp);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

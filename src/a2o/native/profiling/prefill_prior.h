#pragma once
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
// 只在计时前的观察prefill收集逐通道平方和或注意力分数，不接触decode未来token。
struct PrefillPrior {
    bool enabled=false,collect_ffn=true,collect_kv=false,record_n_kv=false;
    int kv_tail_tokens=0;
    int kv_exclude_tail_tokens=0;
    int kv_layer_stride=4;
    double kv_callback_ms=0.0;
    int64_t last_n_kv=-1;
    std::map<int,std::vector<double>> ffn,query;
    std::vector<double> kv_scores;
    std::vector<float> kv_layer_acc,kv_read_scratch;
    static bool callback(ggml_tensor *t,bool ask,void *data){
        auto &self=*static_cast<PrefillPrior*>(data);
        const char *cname=t->name;
        if(ask&&self.record_n_kv&&std::strcmp(cname,"kq_soft_max-0")==0)self.last_n_kv=t->ne[0];
        if(!self.enabled)return !ask;
        bool is_ffn=self.collect_ffn&&std::strncmp(cname,"ffn_swiglu-",11)==0;
        bool is_query=self.collect_ffn&&std::strncmp(cname,"kqv_out-",8)==0;
        bool is_kq=self.collect_kv&&std::strncmp(cname,"kq_soft_max-",12)==0;
        if(is_kq&&self.kv_layer_stride>1){
            int layer=std::atoi(cname+12);
            if((layer%self.kv_layer_stride)!=0)is_kq=false;
        }
        if(ask)return is_ffn||is_query||is_kq;
        if(!is_ffn&&!is_query&&!is_kq)return true;
        require(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"prefill prior requires contiguous F32");
        if(is_kq){
            const double start=now_ms();
            const int64_t n_kv=t->ne[0],n_tok=t->ne[1],n_head=t->ne[2];
            require(t->ne[3]==1,"kq_soft_max prior expects single stream");
            const int64_t q_end=(self.kv_exclude_tail_tokens>0&&n_tok>self.kv_exclude_tail_tokens)?(n_tok-self.kv_exclude_tail_tokens):n_tok;
            const int64_t q0=(self.kv_tail_tokens>0&&q_end>self.kv_tail_tokens)?(q_end-self.kv_tail_tokens):0;
            const int64_t q_len=q_end-q0;
            if(self.kv_scores.size()<size_t(n_kv))self.kv_scores.resize(size_t(n_kv),0.0);
            self.kv_layer_acc.assign(size_t(n_kv),0.0f);
            float * __restrict__ acc=self.kv_layer_acc.data();
            if(t->buffer&&ggml_backend_buffer_is_host(t->buffer)&&t->data){
                const float *ptr=reinterpret_cast<const float*>(t->data);
                for(int64_t h=0;h<n_head;++h){
                    const int64_t h_off=h*n_tok*n_kv;
                    for(int64_t q=q0;q<q_end;++q){
                        const float * __restrict__ row=ptr+h_off+q*n_kv;
                        for(int64_t k=0;k<n_kv;++k)acc[k]+=row[k];
                    }
                }
            }else{
                // 仅按 head 拷贝末尾 q_len 行，避免把整块 [n_kv, n_tok, n_head] 全部读回 host。
                const size_t head_tail_elems=size_t(q_len)*size_t(n_kv);
                self.kv_read_scratch.resize(head_tail_elems);
                const size_t head_tail_bytes=head_tail_elems*sizeof(float);
                for(int64_t h=0;h<n_head;++h){
                    const size_t byte_off=size_t((h*n_tok+q0)*n_kv)*sizeof(float);
                    ggml_backend_tensor_get(t,self.kv_read_scratch.data(),byte_off,head_tail_bytes);
                    for(int64_t dq=0;dq<q_len;++dq){
                        const float * __restrict__ row=self.kv_read_scratch.data()+size_t(dq)*size_t(n_kv);
                        for(int64_t k=0;k<n_kv;++k)acc[k]+=row[k];
                    }
                }
            }
            for(int64_t k=0;k<n_kv;++k){
                float v=acc[k];require(std::isfinite(v),"kq prior nonfinite");
                self.kv_scores[size_t(k)]+=double(v);
            }
            self.kv_callback_ms+=now_ms()-start;
            return true;
        }
        std::string name=cname;
        int layer=std::stoi(name.substr(name.find_last_of('-')+1));
        std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
        auto &energy=(is_ffn?self.ffn:self.query)[layer];if(energy.empty())energy.resize(t->ne[0],0);
        require(energy.size()==size_t(t->ne[0]),"prior width changed");
        for(size_t i=0;i<values.size();++i){require(std::isfinite(values[i]),"prior nonfinite");energy[i%energy.size()]+=double(values[i])*values[i];}
        return true;
    }
    std::vector<int> select_ffn(int layer,int width,int keep) const {
        auto it=ffn.find(layer);require(it!=ffn.end()&&it->second.size()==size_t(width),"missing prefill FFN prior");
        std::vector<std::pair<double,int>> rank;
        for(int i=0;i<width;i+=256){double e=0;for(int j=0;j<256;++j)e+=it->second[i+j];rank.push_back({e,i});}
        std::stable_sort(rank.begin(),rank.end(),[](auto a,auto b){return a.first>b.first;});
        std::vector<int> ids;for(int i=0;i<keep/256;++i)for(int j=0;j<256;++j)ids.push_back(rank[i].second+j);
        std::sort(ids.begin(),ids.end());return ids;
    }
    std::vector<int> select_query(int layer,int nh,int nkv,int hd,int kg) const {
        auto it=query.find(layer);require(it!=query.end()&&it->second.size()==size_t(nh*hd),"missing prefill query prior");
        std::vector<int> ids;
        for(int g=0;g<nkv;++g){std::vector<std::pair<double,int>> rank;
            for(int h=g*(nh/nkv);h<(g+1)*(nh/nkv);++h){double e=0;for(int d=0;d<hd;++d)e+=it->second[h*hd+d];rank.push_back({e,h});}
            std::stable_sort(rank.begin(),rank.end(),[](auto a,auto b){return a.first>b.first;});
            for(int j=0;j<kg;++j)for(int d=0;d<hd;++d)ids.push_back(rank[j].second*hd+d);
        }
        std::sort(ids.begin(),ids.end());return ids;
    }
};

// 仅在最后一个 batch chunk 启用 kq_soft_max 采集（回调内部再按 kv_tail_tokens 切片），
// 既不增加额外 forward 次数，也避免拆出 <32 的小 batch 掉进 HTP 性能低谷。
static void evaluate_with_prior_tail(llama_context *ctx,const std::vector<llama_token> &ts,
        llama_pos &pos,PrefillPrior &prior,bool collect,int tail_tokens,int exclude_tail_tokens=0){
    prior.kv_scores.clear();
    if(!collect||ts.empty()){
        prior.enabled=false;
        prior.kv_exclude_tail_tokens=0;
        evaluate(ctx,ts,pos);
        return;
    }
    const size_t bs=size_t(batch_size);
    const size_t n_tail=(tail_tokens>0)?std::min(ts.size(),size_t(tail_tokens)):ts.size();
    const size_t n_excl=size_t(std::max(0,exclude_tail_tokens));
    const size_t n_need=std::min(ts.size(),n_tail+n_excl);
    size_t split=0;
    if(ts.size()>bs){
        const size_t rem=ts.size()%bs;
        if(rem==0){
            split=ts.size()-bs;
        }else if(rem>=n_need){
            split=ts.size()-rem;
        }else{
            // 保持总 chunk 数不变，同时把最后一块凑到 >=32（且 <=bs）以避开 HTP 小 batch 低谷。
            const size_t tail_chunk=std::min(bs,std::max(n_need,size_t(32)));
            split=(ts.size()>tail_chunk)?(ts.size()-tail_chunk):0;
        }
    }
    if(split>0){
        prior.enabled=false;
        evaluate(ctx,std::vector<llama_token>(ts.begin(),ts.begin()+split),pos);
    }
    using Setter = void (*)(llama_context *, ggml_backend_sched_eval_callback, void *);
    static const auto set_callback = reinterpret_cast<Setter>(dlsym(RTLD_DEFAULT, "llama_a2o_set_eval_callback"));
    require(set_callback, "runtime does not support scoped prefill callbacks");
    struct ScopedPrior {
        Setter set;
        llama_context * ctx;
        PrefillPrior & prior;
        ~ScopedPrior() {
            set(ctx,nullptr,nullptr);
            prior.enabled=false;
            prior.kv_exclude_tail_tokens=0;
        }
    } scope{set_callback,ctx,prior};
    prior.kv_exclude_tail_tokens=exclude_tail_tokens;
    prior.enabled=true;
    set_callback(ctx,&PrefillPrior::callback,&prior);
    evaluate(ctx,std::vector<llama_token>(ts.begin()+split,ts.end()),pos);
}

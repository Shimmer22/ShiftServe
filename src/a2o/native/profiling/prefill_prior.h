#pragma once
#include <cstring>
// 只在计时前的观察prefill收集逐通道平方和或注意力分数，不接触decode未来token。
struct PrefillPrior {
    bool enabled=false,collect_ffn=true,collect_kv=false,record_n_kv=false;
    int kv_tail_tokens=0;
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
        if(ask)return is_ffn||is_query||is_kq;
        if(!is_ffn&&!is_query&&!is_kq)return true;
        require(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"prefill prior requires contiguous F32");
        if(is_kq){
            const int64_t n_kv=t->ne[0],n_tok=t->ne[1],n_head=t->ne[2];
            require(t->ne[3]==1,"kq_soft_max prior expects single stream");
            const float *ptr=nullptr;
            if(t->buffer&&ggml_backend_buffer_is_host(t->buffer)&&t->data){
                ptr=reinterpret_cast<const float*>(t->data);
            }else{
                self.kv_read_scratch.resize(size_t(ggml_nelements(t)));
                ggml_backend_tensor_get(t,self.kv_read_scratch.data(),0,ggml_nbytes(t));
                ptr=self.kv_read_scratch.data();
            }
            if(self.kv_scores.size()<size_t(n_kv))self.kv_scores.resize(size_t(n_kv),0.0);
            self.kv_layer_acc.assign(size_t(n_kv),0.0f);
            float * __restrict__ acc=self.kv_layer_acc.data();
            const int64_t q0=(self.kv_tail_tokens>0&&n_tok>self.kv_tail_tokens)?(n_tok-self.kv_tail_tokens):0;
            for(int64_t h=0;h<n_head;++h){
                const int64_t h_off=h*n_tok*n_kv;
                for(int64_t q=q0;q<n_tok;++q){
                    const float * __restrict__ row=ptr+h_off+q*n_kv;
                    for(int64_t k=0;k<n_kv;++k)acc[k]+=row[k];
                }
            }
            for(int64_t k=0;k<n_kv;++k){
                float v=acc[k];require(std::isfinite(v),"kq prior nonfinite");
                self.kv_scores[size_t(k)]+=double(v);
            }
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

// 仅在末尾 tail_tokens 个 token 窗口启用 kq_soft_max 采集，避免前段 chunk 重复触发图切分与同步。
static void evaluate_with_prior_tail(llama_context *ctx,const std::vector<llama_token> &ts,
        llama_pos &pos,PrefillPrior &prior,bool collect,int tail_tokens){
    prior.kv_scores.clear();
    if(!collect||ts.empty()){
        prior.enabled=false;
        evaluate(ctx,ts,pos);
        return;
    }
    int n_tail=tail_tokens>0?std::min(int(ts.size()),tail_tokens):int(ts.size());
    size_t split=ts.size()-size_t(n_tail);
    if(split>0){
        prior.enabled=false;
        evaluate(ctx,std::vector<llama_token>(ts.begin(),ts.begin()+split),pos);
    }
    prior.enabled=true;
    evaluate(ctx,std::vector<llama_token>(ts.begin()+split,ts.end()),pos);
    prior.enabled=false;
}

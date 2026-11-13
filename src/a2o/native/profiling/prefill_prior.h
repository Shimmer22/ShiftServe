#pragma once
// 只在计时前的观察prefill收集逐通道平方和，不接触decode未来token。
struct PrefillPrior {
    bool enabled=false;
    std::map<int,std::vector<double>> ffn,query;
    static bool callback(ggml_tensor *t,bool ask,void *data){
        auto &self=*static_cast<PrefillPrior*>(data);if(!self.enabled)return !ask;
        std::string name=t->name;bool is_ffn=name.rfind("ffn_swiglu-",0)==0,is_query=name.rfind("kqv_out-",0)==0;
        if(ask)return is_ffn||is_query;
        if(!is_ffn&&!is_query)return true;
        require(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"prefill prior requires contiguous F32");
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

#pragma once
// 仅缩小Query heads；每个KV组保留相同数量Query，KV形状和状态保持不变。
struct QueryHeadProbe {
    llama_model *model;
    std::vector<std::pair<ggml_tensor*,ggml_tensor*>> original;
    std::array<uint32_t, LLAMA_MAX_LAYERS> heads;
    ggml_context *meta=nullptr;
    J shapes=J::array();
    size_t packed_bytes=0;
    explicit QueryHeadProbe(llama_model *m):model(m),heads(m->hparams.n_head_arr){
        for(auto &l:m->layers)original.push_back({l.wq,l.wo});
    }
    ~QueryHeadProbe(){
        for(size_t i=0;i<original.size();++i){model->layers[i].wq=original[i].first;model->layers[i].wo=original[i].second;}
        model->hparams.n_head_arr=heads;if(meta)ggml_free(meta);
    }
    void install(const std::string &path,double ratio,bool htp,const PrefillPrior *prior=nullptr){
        if(ratio==1)return;
        require(ratio>0&&ratio<1,"invalid query head keep ratio");
        auto *uf=gguf_init_from_file(path.c_str(),{true,nullptr});require(uf,"query GGUF metadata");
        std::ifstream file(path,std::ios::binary);meta=ggml_init({1024*1024,nullptr,true});require(meta,"query metadata");
        for(size_t i=0;i<original.size();++i){
            const int nh=heads[i],nkv=model->hparams.n_head_kv(i),hd=model->hparams.n_embd_head_k(i);
            const int group=nh/nkv,kg=std::lround(group*ratio),kept=kg*nkv;
            require(nh%nkv==0&&kg>=1&&kg<group,"query keep must retain integral heads per KV group");
            std::vector<int> channels;
            for(int g=0;g<nkv;++g)for(int j=0;j<kg;++j)for(int d=0;d<hd;++d)channels.push_back((g*group+j)*hd+d);
            if(prior)channels=prior->select_query(i,nh,nkv,hd,kg);
            auto q=original[i].first,o=original[i].second;
            require(q&&o&&!model->layers[i].wqkv,"separate query/output weights required");
            for(int which=0;which<2;++which){
                auto *src=which?o:q;require(src->type==GGML_TYPE_Q4_0,"query compaction requires Q4_0");
                auto *dst=ggml_new_tensor_2d(meta,src->type,which?int(channels.size()):src->ne[0],which?src->ne[1]:int(channels.size()));
                ggml_set_name(dst,src->name);dst->buffer=src->buffer;dst->data=src->data;
                require(ggml_backend_buffer_init_tensor(dst->buffer,dst)==GGML_STATUS_SUCCESS,"query tensor init");
                if(htp){auto *c=ggml_init({65536,nullptr,true});auto *x=ggml_new_tensor_2d(c,GGML_TYPE_F32,dst->ne[0],1);require(ggml_backend_dev_supports_op(ggml_backend_dev_by_name("HTP0"),ggml_mul_mat(c,dst,x)),"query matrix unsupported HTP");ggml_free(c);}
                auto id=gguf_find_tensor(uf,src->name);require(id>=0,"query weight missing");
                std::vector<uint8_t> raw(ggml_nbytes(src)),packed(ggml_nbytes(dst));file.seekg(gguf_get_data_offset(uf)+gguf_get_tensor_offset(uf,id));file.read(reinterpret_cast<char*>(raw.data()),raw.size());require(bool(file),"query weight read");
                if(!which){for(size_t r=0;r<channels.size();++r)std::memcpy(packed.data()+r*dst->nb[1],raw.data()+channels[r]*src->nb[1],dst->nb[1]);}
                else { // 128维完整head，对齐Q4_0的32元素量化块，不重新量化。
                    const size_t block=ggml_type_size(src->type),bs=ggml_blck_size(src->type);
                    for(int64_t r=0;r<src->ne[1];++r)for(size_t c=0;c<channels.size();c+=bs)
                        std::memcpy(packed.data()+r*dst->nb[1]+c/bs*block,raw.data()+r*src->nb[1]+channels[c]/bs*block,block);
                }
                ggml_backend_tensor_set(dst,packed.data(),0,packed.size());packed_bytes+=packed.size();
                if(which)model->layers[i].wo=dst;else model->layers[i].wq=dst;
            }
            model->hparams.n_head_arr[i]=kept;
            shapes.push_back({{"layer",i},{"query_heads",kept},{"original_query_heads",nh},{"kv_heads",nkv},{"head_dim",hd},{"selected_channels",channels}});
        }
        gguf_free(uf);
    }
};

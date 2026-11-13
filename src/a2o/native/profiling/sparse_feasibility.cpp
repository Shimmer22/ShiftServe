// 固定FFN宽度和批量验证成本探针；只测性能，不验证重要性迁移与生成质量。
#define main a2o_legacy_main
#define json a2o_json_string
#include "../agent_runtime.cpp"
#undef json
#undef main
#include "llama-model.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <array>
#include <cstring>
#include <filesystem>
#include <random>
using J=nlohmann::ordered_json;

static void save_probe(const std::string &path,const J &value){
    std::ofstream stream(path+".tmp");stream<<value.dump(2)<<'\n';
    require(bool(stream),"write probe JSON");stream.close();
    require(std::rename((path+".tmp").c_str(),path.c_str())==0,"commit probe JSON");
}
static J compare_probe(const std::vector<float>&a,const std::vector<float>&b){
    require(a.size()==b.size()&&!a.empty(),"logits shape mismatch");
    double aa=0,bb=0,ab=0,err=0,maxerr=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite logits");
        double d=double(a[i])-b[i];aa+=double(a[i])*a[i];bb+=double(b[i])*b[i];ab+=double(a[i])*b[i];err+=d*d;maxerr=std::max(maxerr,std::abs(d));}
    return {{"cosine",ab/std::sqrt(std::max(aa*bb,1e-30))},{"nrmse",std::sqrt(err/std::max(aa,1e-30))},
        {"max_abs",maxerr},{"argmax_match",std::max_element(a.begin(),a.end())-a.begin()==std::max_element(b.begin(),b.end())-b.begin()}};
}
static std::vector<float> last_probe(llama_context *ctx,int nv){
    auto *p=llama_get_logits_ith(ctx,-1);require(p,"missing logits");
    std::vector<float> values(p,p+nv);for(float v:values)require(std::isfinite(v),"nonfinite logits");return values;
}
static std::vector<std::vector<float>> block_probe(llama_context *ctx,const std::vector<llama_token>&ids,llama_pos pos,int nv){
    auto b=llama_batch_init(ids.size(),0,1);b.n_tokens=ids.size();
    for(size_t i=0;i<ids.size();++i){b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=1;}
    int rc=llama_decode(ctx,b);llama_batch_free(b);require(rc==0,"block decode failed: "+std::to_string(rc));llama_synchronize(ctx);
    std::vector<std::vector<float>> logits;
    for(size_t i=0;i<ids.size();++i){auto *p=llama_get_logits_ith(ctx,i);require(p,"missing per-position verify logits");logits.emplace_back(p,p+nv);}
    return logits;
}
#include "prefill_prior.h"
struct CompactProbe {
    struct Layer {ggml_tensor *up,*gate,*down;};
    llama_model *model;
    std::vector<Layer> originals;
    std::vector<ggml_context*> contexts;
    std::vector<ggml_backend_buffer_t> buffers;
    size_t packed_bytes=0;
    J shapes=J::array();
    explicit CompactProbe(llama_model *m):model(m){for(auto &l:m->layers)originals.push_back({l.ffn_up,l.ffn_gate,l.ffn_down});}
    void reset(){for(size_t i=0;i<originals.size();++i){auto &l=model->layers[i];l.ffn_up=originals[i].up;l.ffn_gate=originals[i].gate;l.ffn_down=originals[i].down;}}
    ~CompactProbe(){reset();for(auto b:buffers)ggml_backend_buffer_free(b);for(auto c:contexts)ggml_free(c);}
    void restore_weights(const std::string &path){
        auto *uf=gguf_init_from_file(path.c_str(),{true,nullptr});require(uf,"restore weight metadata");std::ifstream file(path,std::ios::binary);
        for(auto old:originals)for(auto *w:{old.up,old.gate,old.down}){
            auto id=gguf_find_tensor(uf,w->name);require(id>=0,"restore FFN tensor missing");std::vector<uint8_t> raw(ggml_nbytes(w));
            file.seekg(gguf_get_data_offset(uf)+gguf_get_tensor_offset(uf,id));file.read(reinterpret_cast<char*>(raw.data()),raw.size());require(bool(file),"restore FFN weight read");
            ggml_backend_tensor_set(w,raw.data(),0,raw.size());
        }gguf_free(uf);
    }
    void install(const std::string &path,double ratio,bool htp,bool inplace,const PrefillPrior *prior=nullptr){
        if(ratio==1)return;
        if(ratio==0){for(auto &l:model->layers){l.ffn_up=nullptr;l.ffn_gate=nullptr;l.ffn_down=nullptr;}return;}
        auto *uf=gguf_init_from_file(path.c_str(),{true,nullptr});require(uf,"GGUF metadata open");
        std::ifstream file(path,std::ios::binary);require(bool(file),"GGUF weights open");
        // FastRPC映射数量有限；所有层共用一个后端权重buffer。
        auto *c=ggml_init({1024*1024,nullptr,true});require(c,"compact metadata allocation");contexts.push_back(c);
        std::vector<std::array<ggml_tensor*,3>> targets;
        for(size_t i=0;i<originals.size();++i){
            auto &l=model->layers[i];auto old=originals[i];
            require(old.up&&old.gate&&old.down,"dense FFN required");
            require(!l.ffn_up_s&&!l.ffn_gate_s&&!l.ffn_down_s,"scaled FFN unsupported");
            int64_t keep=int64_t(old.up->ne[1]*ratio)/256*256;require(keep>0,"empty compact FFN");
            std::array<ggml_tensor*,3> source={old.up,old.gate,old.down},target;
            for(int j=0;j<3;++j){require(source[j]->type==GGML_TYPE_Q4_0,"compact probe requires Q4_0");
                target[j]=ggml_new_tensor_2d(c,GGML_TYPE_Q4_0,j==2?keep:source[j]->ne[0],j==2?source[j]->ne[1]:keep);
                ggml_set_name(target[j],source[j]->name);}
            if(inplace)for(int j=0;j<3;++j){target[j]->data=source[j]->data;target[j]->buffer=source[j]->buffer;
                require(ggml_backend_buffer_init_tensor(target[j]->buffer,target[j])==GGML_STATUS_SUCCESS,"inplace compact tensor init");}
            targets.push_back(target);
        }
        if(!inplace){auto buft=ggml_backend_buffer_get_type(originals[0].up->buffer);
            auto b=ggml_backend_alloc_ctx_tensors_from_buft(c,buft);require(b,"compact backend allocation");buffers.push_back(b);
            ggml_backend_buffer_set_usage(b,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);}
        std::string buffer_name=ggml_backend_buffer_name(targets[0][0]->buffer);
        require(!htp||buffer_name.find("HTP")!=std::string::npos,"compact HTP weights silently placed elsewhere");
        for(size_t i=0;i<originals.size();++i){
            auto &l=model->layers[i];auto old=originals[i];auto target=targets[i];
            std::array<ggml_tensor*,3> source={old.up,old.gate,old.down};int64_t keep=target[0]->ne[1];
            if(htp){
                auto *dev=ggml_backend_dev_by_name("HTP0");auto *test=ggml_init({65536,nullptr,true});require(dev&&test,"HTP compact capability check");
                for(auto *w:target){auto *x=ggml_new_tensor_2d(test,GGML_TYPE_F32,w->ne[0],1);
                    require(ggml_backend_dev_supports_op(dev,ggml_mul_mat(test,w,x)),"compact matrix unsupported on HTP");}
                ggml_free(test);
            }
            std::vector<int> channels;
            if(prior)channels=prior->select_ffn(i,old.up->ne[1],keep);else for(int j=0;j<keep;++j)channels.push_back(j);
            for(int j=0;j<3;++j){
                int64_t id=gguf_find_tensor(uf,source[j]->name);require(id>=0,"missing FFN GGUF weight");
                std::vector<uint8_t> raw(ggml_nbytes(source[j]));
                file.seekg(gguf_get_data_offset(uf)+gguf_get_tensor_offset(uf,id));file.read(reinterpret_cast<char*>(raw.data()),raw.size());require(bool(file),"FFN weight read");
                std::vector<uint8_t> compact(ggml_nbytes(target[j]));
                if(j<2){for(size_t r=0;r<channels.size();++r)std::memcpy(compact.data()+r*target[j]->nb[1],raw.data()+channels[r]*source[j]->nb[1],target[j]->nb[1]);}
                else {size_t bs=ggml_blck_size(source[j]->type),bytes=ggml_type_size(source[j]->type);
                    for(int64_t row=0;row<target[j]->ne[1];++row)for(size_t c=0;c<channels.size();c+=bs)
                        std::memcpy(compact.data()+row*target[j]->nb[1]+c/bs*bytes,raw.data()+row*source[j]->nb[1]+channels[c]/bs*bytes,bytes);
                }
                ggml_backend_tensor_set(target[j],compact.data(),0,compact.size());packed_bytes+=compact.size();
            }
            l.ffn_up=target[0];l.ffn_gate=target[1];l.ffn_down=target[2];
            shapes.push_back({{"layer",i},{"original_ffn",old.up->ne[1]},{"kept_ffn",keep},{"selected_channels",channels},{"buffer",buffer_name}});
        }
        gguf_free(uf);
    }
};

#ifndef A2O_SPARSE_PROBE_NO_MAIN
int main(int argc,char **argv){try{
    require(argc==5,"usage: sparse_feasibility MODEL CONFIG BACKEND OUTPUT");
    const std::string model_path=argv[1],backend=argv[3],output=argv[4];
    require(backend=="cpu"||backend=="htp","explicit backend required");bool htp=backend=="htp";
    J cfg=J::parse(read(argv[2]));batch_size=128;context_size=cfg.at("context_size");
    // 所有条件一致关闭图复用，确保更换张量后不会继续执行旧图。
    setenv("LLAMA_GRAPH_REUSE_DISABLE","1",1);ggml_backend_load_all();
    J report={{"backend",backend},{"config",cfg},{"scope","fixed contiguous FFN channels; performance only; no importance or quality claim"},
        {"graph_reuse_disabled",true},{"rows",J::array()},{"verification",J::array()}};
    double start=now_ms();Engine engine(model_path,htp,false,!htp,htp?4:6);
    report["model_context_init_ms"]=now_ms()-start;report["resources_after_init"]=resource_sample();save_probe(output,report);
    auto *ctx=engine.ctx;auto *vocab=llama_model_get_vocab(engine.model);int nv=llama_vocab_n_tokens(vocab);
    auto normal=tokens(vocab,cfg.at("normal_prompt"),true);llama_pos pos=0;start=now_ms();evaluate(ctx,normal,pos);
    std::string response;std::vector<llama_token> response_ids;bool eog=false;
    for(int i=0;i<16;++i){auto tok=greedy(ctx,nv);if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}response_ids.push_back(tok);response+=piece(vocab,tok);evaluate(ctx,{tok},pos);}
    report["normal_response"]={{"text",response},{"ids",response_ids},{"eog",eog},{"passed",eog&&response.find("OK")!=std::string::npos},{"elapsed_ms",now_ms()-start}};
    save_probe(output,report);require(report["normal_response"]["passed"],"normal response failed");
    llama_memory_clear(llama_get_memory(ctx),true);pos=0;
    auto prefix=cfg.at("prefix_ids").get<std::vector<llama_token>>(),input=cfg.at("input_ids").get<std::vector<llama_token>>(),generated=cfg.at("generated_ids").get<std::vector<llama_token>>();
    start=now_ms();
    if(cfg.contains("history_cache_metadata")){
        std::string path=cfg.at("history_cache_metadata");auto stored=J::parse(read(path));auto meta=stored.at("metadata");
        require(meta.at("prefix_ids").get<std::vector<llama_token>>()==prefix,"cached history token mismatch");
        require(meta.at("context").get<int>()==context_size&&meta.at("batch").get<int>()==batch_size,"cached history context mismatch");
        const std::string identity=model_path+":"+std::to_string(std::filesystem::file_size(model_path))+":"+std::to_string(static_cast<long long>(std::filesystem::last_write_time(model_path).time_since_epoch().count()))+":";
        require(meta.at("identity").get<std::string>().find(identity)==0,"cached model identity mismatch");
        auto bytes=read(path+".state");require(bytes.size()==stored.at("state_bytes").get<size_t>(),"cached history size mismatch");
        uint64_t hash=14695981039346656037ULL;for(unsigned char v:bytes){hash^=v;hash*=1099511628211ULL;}
        std::ostringstream digest;digest<<std::hex<<hash;require(digest.str()==stored.at("state_checksum").get<std::string>(),"cached history checksum mismatch");
        SessionSnapshot history;history.state.assign(bytes.begin(),bytes.end());history.next_pos=prefix.size();restore_session(ctx,history);pos=prefix.size();
        report["history_source"]={{"metadata",path},{"model_tokens_context_checksum_and_restore_verified",true},{"runtime_identity_checked_by_driver",true}};
    }else evaluate(ctx,prefix,pos);
    report["history_ms"]=now_ms()-start;
    start=now_ms();evaluate(ctx,input,pos);report["full_prefill_ms"]=now_ms()-start;
    SessionSnapshot frozen;frozen.state=export_seq(ctx);frozen.next_pos=pos;frozen.evaluated_tokens=pos;
    report["state_bytes"]=frozen.state.size();report["H"]=prefix.size();report["P"]=input.size();save_probe(output,report);
    auto ks=cfg.at("verify_k").get<std::vector<int>>();
    std::mt19937 rng(cfg.at("seed").get<int>());
    for(int k:ks){
        std::vector<llama_token> block(generated.begin(),generated.begin()+k);
        restore_session(ctx,frozen);pos=frozen.next_pos;std::vector<std::vector<float>> sequential;
        for(auto token:block){evaluate(ctx,{token},pos);sequential.push_back(last_probe(ctx,nv));}
        auto seq_state=export_seq(ctx);auto continuation=generated[k];evaluate(ctx,{continuation},pos);auto seq_cont=last_probe(ctx,nv);
        restore_session(ctx,frozen);auto batched=block_probe(ctx,block,frozen.next_pos,nv);auto batch_state=export_seq(ctx);
        pos=frozen.next_pos+k;evaluate(ctx,{continuation},pos);auto block_cont=last_probe(ctx,nv);
        J comparisons=J::array();for(int i=0;i<k;++i)comparisons.push_back(compare_probe(sequential[i],batched[i]));
        report["verification"].push_back({{"k",k},{"per_position_logits",comparisons},{"state_bitwise_equal",seq_state==batch_state},
            {"continuation",compare_probe(seq_cont,block_cont)}});save_probe(output,report);
        for(int rep=0;rep<cfg.at("rounds").get<int>();++rep){
            std::array<bool,2> order=rep%2?std::array<bool,2>{true,false}:std::array<bool,2>{false,true};
            for(bool batch:order){start=now_ms();restore_session(ctx,frozen);double restore=now_ms()-start;pos=frozen.next_pos;
                start=now_ms();
                if(batch){auto b=llama_batch_init(k,0,1);b.n_tokens=k;for(int i=0;i<k;++i){b.token[i]=block[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=1;}
                    int rc=llama_decode(ctx,b);llama_batch_free(b);require(rc==0,"timed verifier failed");llama_synchronize(ctx);
                }else for(auto token:block)evaluate(ctx,{token},pos);
                double elapsed=now_ms()-start;last_probe(ctx,nv);
                report["rows"].push_back({{"stage","verify"},{"k",k},{"batch",batch},{"repetition",rep},{"elapsed_ms",elapsed},{"restore_ms",restore},{"finite_logits",true}});save_probe(output,report);
            }
        }
    }
    // 验证器状态回滚：先确认API能否直接删除recurrent尾部，再用完整检查点恢复。
    if(cfg.value("measure_rollback",true)){
    restore_session(ctx,frozen);block_probe(ctx,std::vector<llama_token>(generated.begin(),generated.begin()+8),frozen.next_pos,nv);
    bool removed=llama_memory_seq_rm(llama_get_memory(ctx),0,frozen.next_pos+4,-1);
    start=now_ms();restore_session(ctx,frozen);double rollback=now_ms()-start;pos=frozen.next_pos;
    start=now_ms();evaluate(ctx,std::vector<llama_token>(generated.begin(),generated.begin()+4),pos);double replay=now_ms()-start;
    report["rollback"]={{"partial_seq_rm_returned",removed},{"checkpoint_restore_verified",true},{"restore_ms",rollback},{"replay_accepted_4_ms",replay}};save_probe(output,report);
    }
    auto ratios=cfg.at("keep_ratios").get<std::vector<double>>();
    for(double ratio:ratios){
        // 先恢复dense post-prefill状态，再替换FFN权重，避免把其他稀疏条件的状态作为输入。
        bool inplace=cfg.value("inplace_compaction",false);
        restore_session(ctx,frozen);CompactProbe pack(engine.model);start=now_ms();pack.install(model_path,ratio,htp,inplace);double setup=now_ms()-start;
        J condition={{"keep_ratio",ratio},{"inplace_compaction",inplace},{"packing_ms",setup},{"packed_bytes",pack.packed_bytes},{"layers",pack.shapes},{"resources",resource_sample()}};
        report["conditions"].push_back(condition);int condition_index=report["conditions"].size()-1;save_probe(output,report);
        std::cerr<<"SPARSE_CONDITION "<<backend<<" ratio="<<ratio<<" pack_ms="<<setup<<'\n';
        int d=cfg.at("sparse_D");
        for(int rep=-1;rep<cfg.at("rounds").get<int>();++rep){
            start=now_ms();restore_session(ctx,frozen);double restore=now_ms()-start;pos=frozen.next_pos;
            auto before=resource_sample();start=now_ms();for(int i=0;i<d;++i)evaluate(ctx,{generated[i]},pos);
            double elapsed=now_ms()-start;auto values=last_probe(ctx,nv);auto after=resource_sample();
            report["rows"].push_back({{"stage","sparse_decode"},{"condition_index",condition_index},{"keep_ratio",ratio},{"repetition",rep},{"warmup",rep<0},
                {"D",d},{"elapsed_ms",elapsed},{"restore_ms",restore},{"finite_logits",true},{"resources_before",before},{"resources_after",after}});
            save_probe(output,report);
        }
        if(inplace&&ratio>0&&ratio<1){start=now_ms();pack.restore_weights(model_path);report["conditions"][condition_index]["restore_original_weights_ms"]=now_ms()-start;save_probe(output,report);}
        pack.reset();
    }
    report["completed"]=true;save_probe(output,report);return 0;
}catch(const std::exception&e){std::cerr<<"SPARSE_PROBE_FAILURE "<<e.what()<<'\n';return 1;}}
#endif

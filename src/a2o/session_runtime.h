// 多轮基线：完整 sequence snapshot + 精确的模板前缀；GPU 视觉仅编码新增帧。
// Prefix/snapshot design reference: dastin359/SplitLM, engine.cpp (see MULTITURN.md).
#include "async_preparation.h"
struct SessionSnapshot {
    std::vector<uint8_t> state;
    std::string rendered_prefix;
    llama_pos next_pos=0;
    int evaluated_tokens=0;
    uint64_t version=0;
};

struct PreparedChunk {
    std::vector<llama_token> text;
    std::vector<float> image;
    std::vector<mtmd_decoder_pos> positions;
    int n_pos=0;
};

static void evaluate_prepared_image(llama_context * ctx, PreparedChunk & chunk, int dim) {
    // Same [t,y,x,z] layout as mtmd_helper_decode_image_chunk, using public positions.
    for(size_t offset=0;offset<chunk.positions.size();offset+=batch_size) {
        int n=std::min(size_t(batch_size),chunk.positions.size()-offset);
        std::vector<llama_pos> pos(4*n);
        std::vector<int32_t> counts(n,1); llama_seq_id sequence=0;
        std::vector<llama_seq_id*> sequences(n,&sequence);
        std::vector<int8_t> logits(n,0);
        for(int j=0;j<n;++j) {
            const auto & p=chunk.positions[offset+j];
            pos[j]=p.t; pos[n+j]=p.y; pos[2*n+j]=p.x; pos[3*n+j]=p.z;
        }
        llama_batch batch{n,nullptr,chunk.image.data()+offset*dim,pos.data(),counts.data(),sequences.data(),logits.data()};
        require(llama_decode(ctx,batch)==0,"prepared image prefill failed");
    }
    llama_synchronize(ctx);
}

static size_t restore_session(llama_context * ctx, const SessionSnapshot & snapshot) {
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx),true);
    if(snapshot.state.empty()) return 0;
    require(llama_state_seq_set_data(ctx,snapshot.state.data(),snapshot.state.size(),0)==snapshot.state.size(),"session restore failed");
    require(export_seq(ctx)==snapshot.state,"session restore changed KV/position state");
    return snapshot.state.size();
}

static int run_session(const std::map<std::string,std::string> & args) {
    auto get=[&](const std::string & key,const std::string & fallback="") {
        auto it=args.find(key); return it==args.end()?fallback:it->second;
    };
    require(get("--policy","none")=="none" && get("--pre-move","0")=="0","session baseline disables compression and premove");
    require(get("--vision-backend")=="opencl" && get("--vision-flash")=="off","session baseline uses OpenCL vision with FA off");
    double start=now_ms();
    Engine cpu(get("--model"),false,false), htp(get("--model"),true,false);
    double model_ms=now_ms()-start;
    auto vocab=llama_model_get_vocab(cpu.model); int nv=llama_vocab_n_tokens(vocab);
    SessionSnapshot snapshot;
    std::vector<std::pair<std::string,std::string>> conversation={{"system",read(get("--system"))}};
    std::ifstream plan(get("--session")); require(bool(plan),"session plan unavailable");
    std::ofstream out(get("--out")); require(bool(out),"session result unavailable");
    out << "{\"session\":true,\"mode\":\"hybrid\",\"vision_backend\":\"OpenCL\","
        << "\"state_model\":\"Qwen2.5-VL full attention; no DeltaNet\","
        << "\"model_context_init_ms\":" << model_ms << ",\"turns\":[";
    bool compare=get("--compare-prefill","1")=="1";
    bool async_smoke=get("--async-smoke","0")=="1";
    require(!async_smoke || !compare,"async mechanism smoke cannot run paired prefill");
    int round=0, total_encodes=0; std::string line;
    while(std::getline(plan,line)) {
        std::vector<std::string> fields; std::istringstream row(line); std::string field;
        while(std::getline(row,field,'\t')) fields.push_back(field);
        require(fields.size()==4 || fields.size()==5 || (async_smoke && fields.size()==6),"invalid session plan row");
        std::string route=fields[0]; bool use_htp=route=="htp";
        require(use_htp || route=="cpu","session route must be cpu or htp");
        std::string decode_route=fields.size()>=5?fields[4]:"cpu";
        require(decode_route=="cpu" || decode_route=="htp","invalid decode backend");
        auto decode_ctx=decode_route=="htp"?htp.ctx:cpu.ctx;
        std::string preparation_case=fields.size()==6?fields[5]:"none";
        require(preparation_case=="none" || preparation_case=="ready" || preparation_case=="early" || preparation_case=="cancel",
                "unknown async smoke case");
        require(!async_smoke || decode_route=="cpu","async smoke retains CPU authority after decode");
        std::unique_ptr<AsyncPreparation> preparation;
        AsyncPreparation::Decision binding{"cpu","disabled"};
        bool authority_checked=false, stale_rejected=false;
        if(async_smoke && preparation_case!="none") {
            require(!snapshot.state.empty(),"async preparation requires committed history");
            auto immutable=std::make_shared<const SessionSnapshot>(snapshot);
            // Worker never reads/writes CPU context. Snapshot includes prefix and M-RoPE metadata.
            preparation=std::make_unique<AsyncPreparation>(snapshot.version,
                [&,immutable](const AsyncPreparation::Checkpoint & valid) {
                    if(!valid()) return false;
                    llama_synchronize(htp.ctx);
                    if(!valid()) return false;
                    llama_memory_clear(llama_get_memory(htp.ctx),true);
                    if(!valid()) return false;
                    require(llama_state_seq_set_data(htp.ctx,immutable->state.data(),immutable->state.size(),0)==immutable->state.size(),
                            "async HTP import failed");
                    if(!valid()) return false;
                    require(export_seq(htp.ctx)==immutable->state,"async HTP state/position verification failed");
                    return valid();
                },preparation_case!="ready");
            // Deterministic fixture barriers, not a model of real Tool latency.
            if(preparation_case=="ready") preparation->wait();
            if(preparation_case=="cancel") { preparation->cancel(); preparation->wait(); }
            preparation->mark("observation_arrived");
            binding=preparation->bind(snapshot.version);
            route=binding.backend; use_htp=route=="htp";
            require(export_seq(cpu.ctx)==snapshot.state,"background preparation mutated CPU authority");
            authority_checked=true;
        }
        std::string image_path=fields[1]=="-"?"":fields[1];
        std::string user=read(fields[2]); int max_tokens=std::stoi(fields[3]);
        if(!image_path.empty()) user=std::string(mtmd_default_marker())+"\n"+user;
        conversation.push_back({"user",user});
        std::vector<llama_chat_message> messages;
        for(const auto & m:conversation) messages.push_back({m.first.c_str(),m.second.c_str()});
        auto full=format(cpu.model,messages,true);
        require(full.compare(0,snapshot.rendered_prefix.size(),snapshot.rendered_prefix)==0,"session prefix mismatch");
        auto suffix=full.substr(snapshot.rendered_prefix.size());
        std::ofstream(get("--out")+".turn-"+std::to_string(round)+".prompt.txt") << full;
        std::cerr << "SESSION turn=" << round << " route=" << route << " cached_tokens=" << snapshot.evaluated_tokens
                  << " next_pos=" << snapshot.next_pos << '\n';
        double vision_init_ms=0, vision_ms=0, preprocess_ms=0, vision_release_ms=0;
        int encodes=0, new_tokens=0;
        mtmd_context * mm=nullptr; mtmd_input_chunks * chunks=nullptr;
        std::vector<PreparedChunk> prepared;
        std::vector<llama_token> plain;
        if(!image_path.empty()) {
            start=now_ms(); auto mp=mtmd_context_params_default(); mp.use_gpu=true;
            mp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; mp.n_threads=4; mp.warmup=false;
            actual_vision_backend.clear(); mtmd_helper_log_set(vision_log,nullptr);
            mm=mtmd_init_from_file(get("--mmproj").c_str(),cpu.model,mp);
            require(mm && actual_vision_backend=="OpenCL","session GPU vision initialization failed");
            vision_init_ms=now_ms()-start;
            start=now_ms(); auto bitmap=mtmd_helper_bitmap_init_from_file(mm,image_path.c_str());
            require(bitmap,"session image load failed");
            const mtmd_bitmap * bitmaps[]={bitmap}; chunks=mtmd_input_chunks_init();
            mtmd_input_text text={suffix.c_str(),round==0,true};
            require(mtmd_tokenize(mm,chunks,&text,bitmaps,1)==0,"session image tokenize failed");
            mtmd_bitmap_free(bitmap); preprocess_ms=now_ms()-start;
            new_tokens=mtmd_helper_get_n_tokens(chunks);
            llama_pos prepare_pos=snapshot.next_pos;
            for(size_t i=0;i<mtmd_input_chunks_size(chunks);++i) {
                auto chunk=mtmd_input_chunks_get(chunks,i);
                PreparedChunk item; item.n_pos=mtmd_input_chunk_get_n_pos(chunk);
                if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    start=now_ms(); require(mtmd_encode_chunk(mm,chunk)==0,"session GPU encoding failed");
                    size_t n=mtmd_input_chunk_get_n_tokens(chunk);
                    auto data=mtmd_get_output_embd(mm);
                    item.image.assign(data,data+n*llama_model_n_embd_inp(cpu.model));
                    auto image_tokens=mtmd_input_chunk_get_tokens_image(chunk);
                    for(size_t j=0;j<n;++j) item.positions.push_back(mtmd_image_tokens_get_decoder_pos(image_tokens,prepare_pos,j));
                    vision_ms+=now_ms()-start; ++encodes; ++total_encodes;
                } else {
                    size_t n; auto data=mtmd_input_chunk_get_tokens_text(chunk,&n); item.text.assign(data,data+n);
                }
                prepare_pos+=item.n_pos; prepared.push_back(std::move(item));
            }
            start=now_ms(); mtmd_input_chunks_free(chunks); mtmd_free(mm);
            chunks=nullptr; mm=nullptr; vision_release_ms=now_ms()-start;
        } else { plain=tokens(vocab,suffix,round==0); new_tokens=plain.size(); }
        if(preparation) preparation->mark("embedding_or_text_ready");
        auto prefill=[&](llama_context * ctx) {
            llama_pos pos=snapshot.next_pos;
            if(prepared.empty()) { evaluate(ctx,plain,pos); return pos; }
            for(auto & chunk:prepared) {
                if(!chunk.image.empty()) {
                    evaluate_prepared_image(ctx,chunk,llama_model_n_embd_inp(cpu.model)); pos+=chunk.n_pos;
                } else { evaluate(ctx,chunk.text,pos); }
            }
            llama_synchronize(ctx); return pos;
        };
        double cpu_ms=0, htp_ms=0, restore_ms=0, handoff_ms=0;
        llama_pos cpu_pos=0, htp_pos=0;
        std::vector<float> cpu_logits, htp_logits;
        if(compare || !use_htp) {
            start=now_ms(); restore_session(cpu.ctx,snapshot); restore_ms+=now_ms()-start;
            if(preparation) {
                preparation->mark("cpu_prefill_started");
                // Early return binds CPU before releasing the worker; no join on response path.
                if(preparation_case=="early") preparation->release();
            }
            std::cerr<<"SESSION cpu_prefill_begin turn="<<round<<'\n';
            start=now_ms(); cpu_pos=prefill(cpu.ctx); cpu_ms=now_ms()-start;
            std::cerr<<"SESSION cpu_prefill_done turn="<<round<<'\n';
            greedy(cpu.ctx,nv); auto logits=llama_get_logits_ith(cpu.ctx,-1);
            cpu_logits.assign(logits,logits+nv);
        }
        if(compare || use_htp) {
            if(preparation) {
                require(binding.reason=="ready","unready HTP candidate cannot be committed");
                preparation->join(); preparation->mark("htp_prefill_started");
            } else { start=now_ms(); restore_session(htp.ctx,snapshot); restore_ms+=now_ms()-start; }
            std::cerr<<"SESSION htp_prefill_begin turn="<<round<<'\n';
            start=now_ms(); htp_pos=prefill(htp.ctx); htp_ms=now_ms()-start;
            std::cerr<<"SESSION htp_prefill_done turn="<<round<<'\n';
            greedy(htp.ctx,nv); auto logits=llama_get_logits_ith(htp.ctx,-1);
            htp_logits.assign(logits,logits+nv);
        }
        if(compare) require(cpu_pos==htp_pos,"prefill M-RoPE position mismatch");
        if(compare) {
            for(const auto & item:std::vector<std::pair<std::string,const std::vector<float>*>>{{"cpu",&cpu_logits},{"htp",&htp_logits}}) {
                std::ofstream file(get("--out")+".turn-"+std::to_string(round)+"."+item.first+".logits.f32",std::ios::binary);
                file.write(reinterpret_cast<const char*>(item.second->data()),item.second->size()*sizeof(float));
            }
        }
        double cosine=0, nrmse=0, max_abs=0; bool top1=true;
        if(compare) {
            double aa=0,bb=0,ab=0,dd=0;
            for(int i=0;i<nv;++i) {
                double a=cpu_logits[i],b=htp_logits[i],d=a-b;
                aa+=a*a;bb+=b*b;ab+=a*b;dd+=d*d;max_abs=std::max(max_abs,std::abs(d));
            }
            cosine=ab/std::sqrt(aa*bb); nrmse=std::sqrt(dd/aa);
            top1=std::max_element(cpu_logits.begin(),cpu_logits.end())-cpu_logits.begin()==
                 std::max_element(htp_logits.begin(),htp_logits.end())-htp_logits.begin();
        }
        llama_pos pos=use_htp?htp_pos:cpu_pos;
        const auto & selected=use_htp?htp_logits:cpu_logits;
        auto first=llama_token(std::max_element(selected.begin(),selected.end())-selected.begin());
        if(mm) { mtmd_input_chunks_free(chunks); mtmd_free(mm); }
        size_t down_bytes=0;
        if(route!=decode_route) {
            start=now_ms(); SessionSnapshot handoff; handoff.state=export_seq(use_htp?htp.ctx:cpu.ctx);
            down_bytes=restore_session(decode_ctx,handoff); handoff_ms=now_ms()-start;
        }
        std::vector<llama_token> generated; std::string output,closing;
        std::cerr<<"SESSION decode_begin turn="<<round<<" backend="<<decode_route<<'\n';
        bool eog=false; start=now_ms();
        for(int i=0;i<max_tokens;++i) {
            auto tok=i==0?first:greedy(decode_ctx,nv);
            if(preparation && llama_vocab_is_eog(vocab,tok)) preparation->mark("action_complete");
            // Commit the actual generated action and closing token before the next snapshot.
            evaluate(decode_ctx,{tok},pos);
            if(llama_vocab_is_eog(vocab,tok)) { closing=piece(vocab,tok); eog=true; break; }
            generated.push_back(tok); output+=piece(vocab,tok);
        }
        double decode_ms=now_ms()-start;
        auto before_tokens=snapshot.evaluated_tokens; auto before_bytes=snapshot.state.size();
        auto before_pos=snapshot.next_pos;
        snapshot.next_pos=pos;
        snapshot.evaluated_tokens+=new_tokens+generated.size()+(eog?1:0);
        snapshot.rendered_prefix=full+output+closing;
        start=now_ms(); snapshot.state=export_seq(decode_ctx); double snapshot_ms=now_ms()-start;
        ++snapshot.version;
        if(preparation) {
            preparation->mark("cpu_state_committed");
            if(preparation_case=="early") {
                // Even if import finished in the meantime, it belongs to the previous version.
                stale_rejected=preparation->bind(snapshot.version).reason=="stale";
                require(stale_rejected,"obsolete HTP preparation was accepted");
            }
            preparation->mark("drain_started_after_response");
            preparation->join(); // Safe reuse at next round, errors remain fatal.
            preparation->mark("drain_finished");
        }
        conversation.push_back({"assistant",output});
        std::ofstream(get("--out")+".turn-"+std::to_string(round)+".prefix.txt") << snapshot.rendered_prefix;
        out << (round?",":"") << "{\"turn\":" << round << ",\"prefill_backend\":" << json(route)
            << ",\"decode_backend\":" << json(decode_route) << ",\"cached_tokens\":" << before_tokens
            << ",\"cached_state_bytes\":" << before_bytes << ",\"state_bytes\":" << snapshot.state.size()
            << ",\"position_before\":" << before_pos << ",\"next_position\":" << pos
            << ",\"evaluated_tokens\":" << snapshot.evaluated_tokens << ",\"new_tokens\":" << new_tokens
            << ",\"image_encode_calls\":" << encodes << ",\"prefix_match\":true,\"restore_checked\":true"
            << ",\"handoff_bytes\":" << down_bytes << ",\"download_bytes\":" << (use_htp && decode_route=="cpu"?down_bytes:0)
            << ",\"eog_committed\":" << (eog?"true":"false")
            << ",\"compared\":" << (compare?"true":"false") << ",\"logits_cosine\":" << cosine
            << ",\"logits_nrmse\":" << nrmse << ",\"logits_max_abs\":" << max_abs
            << ",\"logits_top1_equal\":" << (top1?"true":"false")
            << ",\"output\":" << json(output) << ",\"generated_ids\":[";
        for(size_t i=0;i<generated.size();++i) out << (i?",":"") << generated[i];
        out << "],\"timing_ms\":{\"vision_init_ms\":" << vision_init_ms << ",\"image_preprocess_ms\":" << preprocess_ms
            << ",\"vision_release_ms\":" << vision_release_ms
            << ",\"vision_encode_ms\":" << vision_ms << ",\"cpu_prefill_ms\":" << cpu_ms
            << ",\"htp_prefill_ms\":" << htp_ms << ",\"restore_checked_ms\":" << restore_ms
            << ",\"handoff_checked_ms\":" << handoff_ms << ",\"decode_ms\":" << decode_ms
            << ",\"snapshot_ms\":" << snapshot_ms << "}";
        if(async_smoke) {
            out << ",\"async_preparation\":{\"case\":" << json(preparation_case)
                << ",\"reason\":" << json(binding.reason) << ",\"snapshot_version\":" << (snapshot.version-1)
                << ",\"committed_version\":" << snapshot.version
                << ",\"authority_checked\":" << (authority_checked?"true":"false")
                << ",\"stale_rejected\":" << (stale_rejected?"true":"false")
                << ",\"status\":" << json(preparation?preparation->state():"disabled") << ",\"events\":[";
            if(preparation) {
                auto events=preparation->events();
                for(size_t i=0;i<events.size();++i) out << (i?",":"") << "{\"event\":" << json(events[i].first)
                    << ",\"ms\":" << events[i].second << "}";
            }
            out << "]}";
        }
        out << "}"; out.flush();
        std::cerr << "SESSION completed turn=" << round << " output=" << output << '\n'; ++round;
    }
    out << "],\"image_encode_calls\":" << total_encodes << ",\"history_source\":\"previous selected decode backend snapshot\""
        << ",\"timing_valid_for_performance\":"
        << (async_smoke || get("--timing-valid-for-performance","1")=="0"?"false":"true") << "}\n";
    return 0;
}

// 四路径短动作校准：同一历史、相同新输入，分别记录每个生成 token 的累计计算耗时。
struct PathResult {
    std::string output;
    std::vector<llama_token> ids;
    std::vector<double> decode_prefix_ms;
    std::vector<float> prefill_logits,final_logits;
    std::vector<std::vector<float>> decision_logits;
    double entry_export=0,entry_clear=0,entry_import=0;
    double switch_export=0,switch_clear=0,switch_import=0,prefill=0,verify=0;
    size_t entry_bytes=0,switch_bytes=0;
    bool eog=false;
};

static std::pair<double,double> path_difference(const std::vector<float> & a,const std::vector<float> & b) {
    double aa=0,bb=0,ab=0,dd=0;
    for(size_t i=0;i<a.size();++i) {double d=a[i]-b[i];aa+=double(a[i])*a[i];bb+=double(b[i])*b[i];ab+=double(a[i])*b[i];dd+=d*d;}
    return {ab/std::sqrt(aa*bb),std::sqrt(dd/aa)};
}

static int run_paths(const std::map<std::string,std::string> & args) {
    auto get=[&](std::string k,std::string d="") {auto i=args.find(k);return i==args.end()?d:i->second;};
    int history=std::stoi(get("--profile-history")),repeats=std::stoi(get("--profile-repeats","3"));
    std::vector<int> lengths;std::istringstream input(get("--profile-inputs","16,64"));std::string part;
    while(std::getline(input,part,',')) lengths.push_back(std::stoi(part));
    double t=now_ms();Engine cpu(get("--model"),false,false),htp(get("--model"),true,false);double init=now_ms()-t;
    auto vocab=llama_model_get_vocab(cpu.model);int nv=llama_vocab_n_tokens(vocab);
    auto prefix=tokens(vocab,"<|im_start|>system\nYou are a GUI agent. For every new request return only {\"action\":\"wait\"}. Do not repeat observation text. No markdown.<|im_end|>\n<|im_start|>user\nPast screen observations:\n",true);
    auto closing=tokens(vocab,"\nThe tool completed successfully. Continue waiting for the next request.<|im_end|>\n<|im_start|>assistant\n{\"action\":\"wait\"}<|im_end|>",false);
    std::string records;
    const char * screens[]={"Settings shows battery level and network status.","Developer options shows logging, graphics and debugging controls.","The application displays a list of saved documents.","A confirmation dialog has been dismissed and the toolbar is visible."};
    for(int i=0;i<history;++i) records+="Observation "+std::to_string(i)+": "+screens[i%4]+" Tool result: success.\n";
    auto body=tokens(vocab,records,false);
    require(history>=int(prefix.size()+closing.size()),"history too small for structured fixture");
    body.resize(history-prefix.size()-closing.size());prefix.insert(prefix.end(),body.begin(),body.end());prefix.insert(prefix.end(),closing.begin(),closing.end());
    llama_pos pos=0;
    {std::ofstream f(get("--out")+".prefix.txt");for(auto tok:prefix)f<<piece(vocab,tok);}
    std::cerr<<"PATH build_history H="<<history<<'\n';t=now_ms();evaluate(cpu.ctx,prefix,pos);double history_ms=now_ms()-t;auto base=export_seq(cpu.ctx);
    auto capture=[&](llama_context * ctx) {greedy(ctx,nv);auto l=llama_get_logits_ith(ctx,-1);return std::vector<float>(l,l+nv);};
    auto execute=[&](std::string path,std::string home,const std::vector<llama_token> & suffix) {
        PathResult r;auto hc=home=="cpu"?cpu.ctx:htp.ctx;
        auto pre=path[0]=='C'?cpu.ctx:htp.ctx,dec=path[1]=='C'?cpu.ctx:htp.ctx;
        llama_memory_clear(llama_get_memory(hc),false);import_profile_state(hc,base);
        auto move=[&](llama_context * from,llama_context * to,double & ex,double & cl,double & im,size_t & bytes) {
            double b=now_ms();auto blob=export_seq(from);ex=now_ms()-b;bytes=blob.size();
            b=now_ms();llama_memory_clear(llama_get_memory(to),false);cl=now_ms()-b;
            b=now_ms();import_profile_state(to,blob);im=now_ms()-b;
            b=now_ms();require(export_seq(to)==blob,"path state/position transfer mismatch");r.verify+=now_ms()-b;
        };
        if(hc!=pre) move(hc,pre,r.entry_export,r.entry_clear,r.entry_import,r.entry_bytes);
        llama_pos next=history;double b=now_ms();evaluate(pre,suffix,next);r.prefill=now_ms()-b;
        r.prefill_logits=capture(pre);auto first=greedy(pre,nv);
        std::cerr<<"PATH prefill_done path="<<path<<" first_token="<<first<<'\n';
        if(pre!=dec) move(pre,dec,r.switch_export,r.switch_clear,r.switch_import,r.switch_bytes);
        std::cerr<<"PATH decode_begin path="<<path<<'\n';
        b=now_ms();double audit_ms=0;
        for(int i=0;i<48;++i) {
            double audit_begin=now_ms();
            r.decision_logits.push_back(i?capture(dec):r.prefill_logits);
            audit_ms+=now_ms()-audit_begin;
            auto tok=i?greedy(dec,nv):first;evaluate(dec,{tok},next);
            r.ids.push_back(tok);r.decode_prefix_ms.push_back(now_ms()-b-audit_ms);
            if(llama_vocab_is_eog(vocab,tok)) {r.eog=true;break;}r.output+=piece(vocab,tok);
        }
        r.final_logits=capture(dec);return r;
    };
    std::ofstream out(get("--out"));require(bool(out),"path output unavailable");
    for(int p:lengths) {
        require(history+p+48<=context_size,"path context capacity exceeded");
        auto suffix=tokens(vocab,"\n<|im_start|>user\n",false);
        auto tail=tokens(vocab,"Wait.<|im_end|>\n<|im_start|>assistant\n",false);
        require(p>=int(suffix.size()+tail.size()),"P too small for complete user turn");
        auto pad=tokens(vocab,"The current screen remains unchanged. No action is needed. ",false);
        size_t n=p-suffix.size()-tail.size();for(size_t i=0;i<n;++i)suffix.push_back(pad[i%pad.size()]);suffix.insert(suffix.end(),tail.begin(),tail.end());
        {std::ofstream f(get("--out")+".P"+std::to_string(p)+".suffix.txt");for(auto tok:suffix)f<<piece(vocab,tok);}
        std::cerr<<"PATH reference H="<<history<<" P="<<p<<'\n';auto ref=execute("CC","cpu",suffix);
        require(ref.eog && ref.output.find("\"action\"")!=std::string::npos && ref.output.find("\"wait\"")!=std::string::npos,
                "CPU reference failed the short-action fixture; stopping path calibration");
        for(int rep=0;rep<repeats;++rep) for(std::string home:{"cpu","htp"}) {
            std::vector<std::string> routes={"CC","HH","HC","CH"};std::rotate(routes.begin(),routes.begin()+rep%4,routes.end());
            for(auto path:routes) {
                std::cerr<<"PATH H="<<history<<" P="<<p<<" rep="<<rep<<" home="<<home<<" path="<<path<<'\n';
                auto r=execute(path,home,suffix);auto pre=path_difference(ref.prefill_logits,r.prefill_logits),end=path_difference(ref.final_logits,r.final_logits);
                bool same=r.ids==ref.ids;double min_cos=pre.first,max_err=pre.second;
                if(same)for(size_t i=0;i<r.decision_logits.size();++i) {
                    auto diff=path_difference(ref.decision_logits[i],r.decision_logits[i]);
                    min_cos=std::min(min_cos,diff.first);max_err=std::max(max_err,diff.second);
                }
                bool numeric=same && min_cos>=.999 && max_err<=.05;
                out<<"{\"history_tokens\":"<<history<<",\"new_tokens\":"<<p<<",\"repetition\":"<<rep<<",\"home\":"<<json(home)<<",\"path\":"<<json(path)
                   <<",\"model_context_init_ms\":"<<init<<",\"history_build_ms\":"<<history_ms<<",\"state_bytes\":"<<base.size()
                   <<",\"entry_export_ms\":"<<r.entry_export<<",\"entry_clear_ms\":"<<r.entry_clear<<",\"entry_import_ms\":"<<r.entry_import
                   <<",\"entry_bytes\":"<<r.entry_bytes<<",\"prefill_ms\":"<<r.prefill
                   <<",\"switch_export_ms\":"<<r.switch_export<<",\"switch_clear_ms\":"<<r.switch_clear<<",\"switch_import_ms\":"<<r.switch_import
                   <<",\"switch_bytes\":"<<r.switch_bytes<<",\"verification_ms\":"<<r.verify<<",\"tiling_ms\":null"
                   <<",\"output\":"<<json(r.output)<<",\"eog\":"<<(r.eog?"true":"false")
                   <<",\"generated_ids_equal\":"<<(same?"true":"false")<<",\"decode_evaluations\":"<<r.ids.size()
                   <<",\"prefill_cosine\":"<<pre.first<<",\"prefill_nrmse\":"<<pre.second
                   <<",\"final_cosine\":"<<end.first<<",\"final_nrmse\":"<<end.second
                   <<",\"decision_min_cosine\":"<<min_cos<<",\"decision_max_nrmse\":"<<max_err
                   <<",\"numerical_passed\":"<<(numeric?"true":"false")<<",\"decode_prefix_ms\":[";
                for(size_t i=0;i<r.decode_prefix_ms.size();++i)out<<(i?",":"")<<r.decode_prefix_ms[i];
                out<<"],\"generated_ids\":[";for(size_t i=0;i<r.ids.size();++i)out<<(i?",":"")<<r.ids[i];out<<"]}\n";out.flush();
                std::cerr<<"PATH completed prefill_ms="<<r.prefill<<" numeric="<<numeric<<" output="<<r.output<<'\n';
            }
        }
    }
    return 0;
}

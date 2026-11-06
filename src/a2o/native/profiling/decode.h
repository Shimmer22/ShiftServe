#pragma once
#include <random>

// 相同CPU prefill快照上的decode隔离实验；候选模型逐一装载，不争抢设备。
struct DecodeFixture {
    int h=0,p=0;
    SessionSnapshot snapshot;
    llama_token first=0;
    double history_ms=0,prefill_ms=0,tokenize_ms=0;
    std::vector<llama_token> ids;
    std::string output;
    std::vector<std::vector<float>> natural_logits,replay_logits;
};

static int run_decode_profile(const std::map<std::string,std::string> & args) {
    auto get=[&](const std::string &key){return args.at(key);};
    require(get("--mode")=="hybrid" && get("--policy")=="none" && get("--pre-move")=="0","decode requires explicit hybrid/none/0");
    int repeats=std::stoi(get("--decode-repeats")), horizon=std::stoi(get("--decode-horizon"));
    require(repeats>=3 && horizon>=8 && horizon<=256,"invalid decode repetitions/horizon");
    std::vector<int> hs,ps;
    auto parse=[&](std::string value,std::vector<int> & values){std::istringstream in(value);std::string item;while(std::getline(in,item,','))values.push_back(std::stoi(item));};
    parse(get("--decode-histories"),hs);parse(get("--decode-inputs"),ps);
    std::ofstream out(get("--out"));require(bool(out),"decode output unavailable");
    ToolEvents events(get("--out")+".events.jsonl");
    double start=now_ms();Engine cpu(get("--model"),false,false);double cpu_init=now_ms()-start;
    auto vocab=llama_model_get_vocab(cpu.model);int nv=llama_vocab_n_tokens(vocab);
    auto capture=[&](llama_context *ctx){greedy(ctx,nv);auto l=llama_get_logits_ith(ctx,-1);return std::vector<float>(l,l+nv);};
    std::vector<DecodeFixture> fixtures;
    int trial_id=0;
    auto array=[](const auto &values){std::ostringstream s;s<<'[';for(size_t i=0;i<values.size();++i)s<<(i?",":"")<<values[i];s<<']';return s.str();};
    auto execute=[&](DecodeFixture &f,llama_context *ctx,const std::string &backend,const std::string &phase,int repetition,double init_ms,bool reference) {
        double restore_work=0,t=now_ms();restore_session(ctx,f.snapshot,&restore_work);double restore_checked=now_ms()-t;
        llama_pos pos=f.h+f.p;
        std::vector<llama_token> ids,replay_ids;std::string text;bool eog=false;
        std::vector<double> natural_ms,natural_active_ms,replay_ms,replay_active_ms;
        double cosine=1,nrmse=0,audit_ms=0,natural_cosine=1,natural_nrmse=0,replay_cosine=1,replay_nrmse=0;
        std::vector<double> natural_cosines,replay_cosines;
        auto audit=[&](const std::vector<float> &a,const std::vector<float> &b,bool replay){auto d=path_difference(a,b);cosine=std::min(cosine,d.first);nrmse=std::max(nrmse,d.second);
            if(replay){replay_cosine=std::min(replay_cosine,d.first);replay_nrmse=std::max(replay_nrmse,d.second);replay_cosines.push_back(d.first);}
            else{natural_cosine=std::min(natural_cosine,d.first);natural_nrmse=std::max(natural_nrmse,d.second);natural_cosines.push_back(d.first);}};
        double wall_start=now_ms();
        for(int i=0;i<64;++i) {
            double a=active_ms(),b=now_ms();auto tok=i?greedy(ctx,nv):f.first;evaluate(ctx,{tok},pos);
            natural_ms.push_back(now_ms()-b);natural_active_ms.push_back(active_ms()-a);
            ids.push_back(tok);
            b=now_ms();auto logits=capture(ctx);
            if(reference){f.natural_logits.push_back(std::move(logits));natural_cosines.push_back(1);}
            else if(size_t(i)<f.natural_logits.size())audit(f.natural_logits[i],logits,false);
            audit_ms+=now_ms()-b;
            if(llama_vocab_is_eog(vocab,tok)){eog=true;break;}text+=piece(vocab,tok);
        }
        double natural_wall=now_ms()-wall_start;
        bool same=reference || (ids==f.ids && text==f.output);
        if(reference){f.ids=ids;f.output=text;}
        require(eog && !ids.empty(),"decode natural response did not reach EOG");
        double return_natural=0;
        if(ctx!=cpu.ctx){t=now_ms();transfer(ctx,cpu.ctx);return_natural=now_ms()-t;}
        // 固定回放不解释成自然响应：重复CPU参考JSON的非EOG token，逐token输入。
        t=now_ms();restore_session(ctx,f.snapshot);double replay_restore=now_ms()-t;
        pos=f.h+f.p;require(f.ids.size()>1,"empty replay fixture");
        wall_start=now_ms();
        for(int i=0;i<horizon;++i) {
            auto tok=f.ids[size_t(i)%(f.ids.size()-1)];replay_ids.push_back(tok);
            double a=active_ms(),b=now_ms();evaluate(ctx,{tok},pos);
            replay_ms.push_back(now_ms()-b);replay_active_ms.push_back(active_ms()-a);
            b=now_ms();auto logits=capture(ctx);
            if(reference){f.replay_logits.push_back(std::move(logits));replay_cosines.push_back(1);}else audit(f.replay_logits[i],logits,true);
            audit_ms+=now_ms()-b;
        }
        double replay_wall=now_ms()-wall_start,return_replay=0;
        if(ctx!=cpu.ctx){t=now_ms();transfer(ctx,cpu.ctx);return_replay=now_ms()-t;}
        bool natural_passed=same && eog && natural_cosine>=.99,replay_passed=replay_cosine>=.99;
        bool passed=natural_passed && replay_passed;
        out<<std::fixed<<"{\"id\":"<<trial_id<<",\"H\":"<<f.h<<",\"P\":"<<f.p<<",\"backend\":"<<json(backend)
           <<",\"phase\":"<<json(phase)<<",\"repetition\":"<<repetition<<",\"horizon\":"<<horizon
           <<",\"model_context_init_ms\":"<<init_ms<<",\"vision_init_ms\":0,\"history_build_ms\":"<<f.history_ms
           <<",\"prefill_ms\":"<<f.prefill_ms<<",\"fixture_tokenize_ms\":"<<f.tokenize_ms
           <<",\"restore_work_ms\":"<<restore_work<<",\"restore_checked_ms\":"<<restore_checked<<",\"replay_restore_checked_ms\":"<<replay_restore
           <<",\"natural_return_checked_ms\":"<<return_natural<<",\"replay_return_checked_ms\":"<<return_replay
           <<",\"state_bytes\":"<<f.snapshot.state.size()<<",\"state_transfer_passed\":true"
           <<",\"natural_step_ms\":"<<array(natural_ms)<<",\"natural_active_step_ms\":"<<array(natural_active_ms)
           <<",\"replay_input_ids\":"<<array(replay_ids)<<",\"replay_step_ms\":"<<array(replay_ms)<<",\"replay_active_step_ms\":"<<array(replay_active_ms)
           <<",\"natural_wall_with_audit_ms\":"<<natural_wall<<",\"replay_wall_with_audit_ms\":"<<replay_wall<<",\"audit_ms\":"<<audit_ms
           <<",\"output\":"<<json(text)<<",\"generated_ids\":"<<array(ids)<<",\"eog\":"<<(eog?"true":"false")
           <<",\"generated_ids_equal\":"<<(same?"true":"false")<<",\"min_cosine\":"<<cosine<<",\"max_nrmse\":"<<nrmse
           <<",\"natural_min_cosine\":"<<natural_cosine<<",\"natural_max_nrmse\":"<<natural_nrmse
           <<",\"replay_min_cosine\":"<<replay_cosine<<",\"replay_max_nrmse\":"<<replay_nrmse
           <<",\"natural_step_cosine\":"<<array(natural_cosines)<<",\"replay_step_cosine\":"<<array(replay_cosines)
           <<",\"natural_passed\":"<<(natural_passed?"true":"false")<<",\"replay_passed\":"<<(replay_passed?"true":"false")
           <<",\"passed\":"<<(passed?"true":"false")<<"}\n";out.flush();
        events.emit("decode_trial_completed",trial_id,"\"H\":"+std::to_string(f.h)+",\"P\":"+std::to_string(f.p)+",\"backend\":"+json(backend)+",\"phase\":"+json(phase)+",\"passed\":"+(passed?"true":"false")+",\"natural_passed\":"+(natural_passed?"true":"false")+",\"replay_passed\":"+(replay_passed?"true":"false"));
        ++trial_id;require(natural_passed,"decode natural response/numerical validation failed");
    };
    for(int h:hs) {
        require(h>0,"invalid H");start=now_ms();auto prefix=observation_history(vocab,h);double tokenize=now_ms()-start;
        require(prefix.size()==size_t(h),"decode history fixture token mismatch");
        std::string rendered;for(auto tok:prefix)rendered+=piece(vocab,tok);
        require(rendered==read(get("--decode-fixtures")+"/H"+std::to_string(h)+".txt"),"decode history bytes differ from original");
        llama_memory_clear(llama_get_memory(cpu.ctx),true);llama_pos pos=0;start=now_ms();evaluate(cpu.ctx,prefix,pos);double history_ms=now_ms()-start;
        SessionSnapshot base;base.state=export_seq(cpu.ctx);base.next_pos=h;
        for(int p:ps) {
            require(h+p+horizon+64<=context_size,"decode context capacity exceeded");
            DecodeFixture f;f.h=h;f.p=p;f.history_ms=history_ms;
            start=now_ms();auto suffix=tokens(vocab,read(get("--decode-fixtures")+"/P"+std::to_string(p)+".txt"),false);f.tokenize_ms=tokenize+now_ms()-start;
            require(suffix.size()==size_t(p),"decode input fixture token mismatch");restore_session(cpu.ctx,base);pos=h;
            start=now_ms();evaluate(cpu.ctx,suffix,pos);f.prefill_ms=now_ms()-start;f.first=greedy(cpu.ctx,nv);
            f.snapshot.state=export_seq(cpu.ctx);f.snapshot.next_pos=pos;
            execute(f,cpu.ctx,"cpu","reference",-1,cpu_init,true);fixtures.push_back(std::move(f));
        }
    }
    events.emit("decode_fixtures_ready",-1,"\"conditions\":"+std::to_string(fixtures.size()));
    std::mt19937 rng(20261106);
    for(const std::string backend:{"cpu","htp","opencl"}) {
        std::unique_ptr<Engine> candidate;
        double init=cpu_init;
        if(backend!="cpu") {start=now_ms();candidate=std::make_unique<Engine>(get("--model"),true,false,backend=="htp"?"HTP0":"GPUOpenCL");init=now_ms()-start;}
        auto ctx=candidate?candidate->ctx:cpu.ctx;
        for(auto &f:fixtures)execute(f,ctx,backend,"warmup",-1,init,false);
        for(int rep=0;rep<repeats;++rep) {
            std::vector<size_t> order(fixtures.size());std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
            for(auto i:order)execute(fixtures[i],ctx,backend,"measurement",rep,init,false);
        }
    }
    events.emit("decode_session_completed",-1,"\"trials\":"+std::to_string(trial_id));
    return 0;
}

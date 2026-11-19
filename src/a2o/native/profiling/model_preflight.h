// 新模型准入：真实自由生成、JSON记忆、四路径及检查点恢复。
J model_preflight(const J &body) {
    require((cpu && htp)||single_resident,"preflight requires explicit CPU/HTP contexts");
    lane.close();anchor={};anchor_ids.clear();
    const std::string path=body.at("path");
    require(path=="cc"||path=="hh"||path=="hc"||path=="ch","invalid preflight path");
    if(single_resident){require(path=="cc"||path=="hh","cross-backend preflight requires dual residency");select_stage_backend(path=="cc"?"cpu":"htp",tokens(vocab,"warmup",true));}
    auto *pre=(path[0]=='c'?cpu:htp)->ensure_ctx();
    auto *dec=(path[1]=='c'?cpu:htp)->ensure_ctx();
    llama_memory_clear(llama_get_memory(pre),true);
    if(pre!=dec)llama_memory_clear(llama_get_memory(dec),true);
    auto chat=render(J{{"messages",J::array({{{"role","user"},{"content","Remember the number 314159 for my next question. Reply only with the JSON object {\"code\":314159}, without markdown."}}})}});
    auto input=tokens(vocab,chat.prompt,true);llama_pos pos=0;
    J turns=J::array();
    for(int turn=0;turn<2;++turn){
        double begin=now_ms();evaluate(pre,input,pos);double prefill=now_ms()-begin;
        auto *lp=llama_get_logits_ith(pre,-1);require(lp,"missing preflight logits");
        const int n_vocab=llama_vocab_n_tokens(vocab);
        for(int i=0;i<n_vocab;++i)require(std::isfinite(lp[i]),"nonfinite preflight logits");
        begin=now_ms();size_t bytes=0;
        if(pre!=dec)bytes=transfer(pre,dec);
        double handoff=now_ms()-begin;
        std::string text;J ids=J::array();bool eog=false;
        begin=now_ms();
        for(int i=0;i<96;++i){
            auto tok=greedy(i==0?pre:dec,n_vocab);
            if(llama_vocab_is_eog(vocab,tok)){evaluate(dec,{tok},pos);eog=true;break;}
            text+=piece(vocab,tok);ids.push_back(tok);evaluate(dec,{tok},pos);
        }
        double decode=now_ms()-begin;J parsed=J::parse(text,nullptr,false);
        bool passed=eog && parsed.is_object() && parsed.contains("code") && parsed["code"]==314159;
        turns.push_back({{"turn",turn},{"output",text},{"generated_ids",ids},{"eog_committed",eog},{"passed",passed},
            {"prefill_ms",prefill},{"handoff_ms",handoff},{"handoff_bytes",bytes},{"decode_ms",decode}});
        if(!passed)return {{"passed",false},{"path",path},{"turns",turns}};
        if(turn==0){
            // 同一上下文保留首轮全部状态；跨后端时完整回迁后再追加第二轮。
            if(pre!=dec)transfer(dec,pre);
            input=tokens(vocab,"\n<|im_start|>user\nWhat number did I ask you to remember? Reply only with the JSON object containing code and that integer, without markdown.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",false);
        }
    }
    SessionSnapshot snapshot;snapshot.state=export_seq(dec);snapshot.next_pos=pos;
    auto probe=tokens(vocab,"checkpoint",false);require(!probe.empty(),"empty checkpoint probe");
    evaluate(dec,probe,pos);auto expected=export_seq(dec);
    auto *lp=llama_get_logits_ith(dec,-1);std::vector<float> expected_logits(lp,lp+llama_vocab_n_tokens(vocab));
    restore_session(dec,snapshot);pos=snapshot.next_pos;evaluate(dec,probe,pos);
    bool same=expected==export_seq(dec);auto *actual=llama_get_logits_ith(dec,-1);
    bool same_logits=std::equal(expected_logits.begin(),expected_logits.end(),actual);
    return {{"passed",same&&same_logits},{"path",path},{"turns",turns},{"checkpoint_state_equal",same},{"checkpoint_logits_equal",same_logits},
        {"scope","text JSON and in-context memory; no image or task-quality equivalence claim"}};
}

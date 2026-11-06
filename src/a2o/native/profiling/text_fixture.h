#pragma once
// 原Observation历史的分段token边界；导出文本重新tokenize不保证相同边界。
static std::vector<llama_token> observation_history(const llama_vocab *vocab,int h) {
    auto prefix=tokens(vocab,"<|im_start|>system\nYou are a GUI agent. For every new request return only {\"action\":\"wait\"}. Do not repeat observation text. No markdown.<|im_end|>\n<|im_start|>user\nPast screen observations:\n",true);
    auto closing=tokens(vocab,"\nThe tool completed successfully. Continue waiting for the next request.<|im_end|>\n<|im_start|>assistant\n{\"action\":\"wait\"}<|im_end|>",false);
    // 复用四路径已验收历史夹具，固定内容以便比较 backend 数值准入。
    std::string records;
    const char * screens[]={"Settings shows battery level and network status.","Developer options shows logging, graphics and debugging controls.","The application displays a list of saved documents.","A confirmation dialog has been dismissed and the toolbar is visible."};
    require(size_t(h)>=prefix.size()+closing.size(),"H too small for structured history");
    std::vector<llama_token> body_tokens;size_t needed=h-prefix.size()-closing.size();
    for(int i=0;i<h;++i) {
        records+="Observation "+std::to_string(i)+": "+screens[i%4]+" Tool result: success.\n";
        if((i+1)%64==0 || i+1==h) {
            body_tokens=tokens(vocab,records,false);
            if(body_tokens.size()>=needed+64)break;
        }
    }
    require(body_tokens.size()>=needed,"history fixture too short");body_tokens.resize(needed);
    prefix.insert(prefix.end(),body_tokens.begin(),body_tokens.end());prefix.insert(prefix.end(),closing.begin(),closing.end());
    require(prefix.size()==h,"H fixture mismatch");
    return prefix;
}

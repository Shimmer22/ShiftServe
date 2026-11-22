#pragma once
#include <limits>
#include <string_view>

// Tokenizer、模板、增量评估与采样。
static std::vector<llama_token> tokens(const llama_vocab * v, std::string_view s, bool first) {
    require(s.size()<=size_t(std::numeric_limits<int32_t>::max()),"tokenizer input too large");
    // llama_tokenize runs the tokenizer even when called only to size its output.
    std::vector<llama_token> out(std::min<size_t>(s.size()+8,65536));
    int32_t n=llama_tokenize(v,s.data(),s.size(),out.data(),out.size(),first,true);
    if(n<0){
        require(n!=std::numeric_limits<int32_t>::min(),"tokenizer output too large");
        out.resize(-n);
        n=llama_tokenize(v,s.data(),s.size(),out.data(),out.size(),first,true);
    }
    require(n>=0,"tokenization failed");
    out.resize(n);
    return out;
}
static void evaluate(llama_context * ctx, const std::vector<llama_token> & ts, llama_pos & pos) {
    for (size_t i = 0; i < ts.size(); i += batch_size) {
        int n = std::min(size_t(batch_size), ts.size() - i);
        auto b = llama_batch_init(n, 0, 1);
        b.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            b.token[j] = ts[i+j]; b.pos[j] = pos+j;
            b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = j == n-1;
        }
        int rc = llama_decode(ctx, b); llama_batch_free(b);
        if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
        pos += n;
    }
    llama_synchronize(ctx);
}
static std::string format(llama_model * model, const std::vector<llama_chat_message> & messages, bool assistant) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    require(tmpl, "model has no chat template");
    int n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), assistant, nullptr, 0);
    require(n > 0, "unsupported model chat template");
    std::string out(n, '\0');
    require(llama_chat_apply_template(tmpl, messages.data(), messages.size(), assistant, out.data(), n) == n,
            "chat formatting failed");
    return out;
}
static llama_token greedy(llama_context * ctx, int n_vocab) {
    const float * p = llama_get_logits_ith(ctx, -1); require(p, "missing logits");
    for (int i = 0; i < n_vocab; ++i) require(std::isfinite(p[i]), "non-finite logits");
    return std::max_element(p, p+n_vocab) - p;
}
static std::string piece(const llama_vocab * v, llama_token tok) {
    std::vector<char> b(256);
    int n = llama_token_to_piece(v, tok, b.data(), b.size(), 0, true);
    if (n < 0) { b.resize(-n); n = llama_token_to_piece(v, tok, b.data(), b.size(), 0, true); }
    require(n >= 0, "detokenization failed"); return {b.data(), size_t(n)};
}

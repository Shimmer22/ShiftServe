#pragma once
// 新图 embedding 及公开 M-RoPE 位置的进程内表示。
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

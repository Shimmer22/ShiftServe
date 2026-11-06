#pragma once
// 显式 CPU/HTP 模型及上下文生命周期。
struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    Engine(const std::string & path, bool npu, bool cpu_repack=true, const char * offload_device="HTP0") {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = npu ? 99 : 0;
        mp.use_mmap = !npu;
        mp.use_extra_bufts = cpu_repack;
        ggml_backend_dev_t devices[2] = {nullptr, nullptr};
        if (npu) {
            devices[0] = ggml_backend_dev_by_name(offload_device);
            require(devices[0], std::string(offload_device)+" is unavailable");
            mp.devices = devices;
        }
        model = llama_model_load_from_file(path.c_str(), mp);
        require(model, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = context_size; cp.n_batch = batch_size; cp.n_ubatch = batch_size;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.offload_kqv = npu;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx = llama_init_from_model(model, cp);
        require(ctx, "context creation failed");
    }
    ~Engine() { if (ctx) llama_free(ctx); if (model) llama_model_free(model); }
};

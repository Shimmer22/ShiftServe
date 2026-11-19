#pragma once
#include <dlfcn.h>
// 显式 CPU/HTP 模型及上下文生命周期。
struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_context_params cp {};
    using ContextFactory = llama_context *(*)(llama_model *, llama_context_params);
private:
    std::shared_ptr<llama_model> model_owner;
    ContextFactory init_context = llama_init_from_model;

    void configure(bool kv_offload, int threads, bool flash_attention,
                   ggml_backend_sched_eval_callback eval_callback, void *eval_user_data) {
        require(threads > 0 && threads <= 16, "invalid engine thread count");
        cp = llama_context_default_params();
        cp.n_ctx = context_size; cp.n_batch = batch_size; cp.n_ubatch = batch_size;
        cp.n_threads = threads; cp.n_threads_batch = threads;
        cp.offload_kqv = kv_offload;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = eval_callback;
        cp.cb_eval_user_data = eval_user_data;
        ensure_ctx();
    }

public:
    Engine(const std::string & path, bool npu, bool cpu_repack=true, bool kv_offload=true, int threads=4, bool flash_attention=false, ggml_backend_sched_eval_callback eval_callback=nullptr, void *eval_user_data=nullptr) {
        require(threads>0 && threads<=16,"invalid engine thread count");
        auto mp = llama_model_default_params();
        int host_gpu_layers = 0;
        if (!npu) {
            if (const char *v = std::getenv("A2O_HOST_GPU_LAYERS")) host_gpu_layers = std::stoi(v);
        }
        mp.n_gpu_layers = npu ? 99 : host_gpu_layers;
        mp.use_mmap = !npu;
        mp.use_extra_bufts = cpu_repack;
        ggml_backend_dev_t devices[2] = {nullptr, nullptr};
        if (npu) {
            devices[0] = ggml_backend_dev_by_name("HTP0");
            require(devices[0], "HTP0 is unavailable");
            mp.devices = devices;
        }
        model_owner = std::shared_ptr<llama_model>(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
        model = model_owner.get();
        require(model, "model load failed");
        configure((npu || host_gpu_layers > 0) && kv_offload, threads, flash_attention, eval_callback, eval_user_data);
    }

    Engine(const Engine &owner, int threads, bool flash_attention,
           ggml_backend_sched_eval_callback eval_callback, void *eval_user_data)
        : model(owner.model), model_owner(owner.model_owner) {
        init_context = reinterpret_cast<ContextFactory>(dlsym(RTLD_DEFAULT, "llama_a2o_init_shared_cpu"));
        require(init_context, "runtime does not provide a shared-weight CPU context");
        configure(false, threads, flash_attention, eval_callback, eval_user_data);
    }

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    void release_ctx() {
        if (ctx) { llama_free(ctx); ctx = nullptr; }
    }
    llama_context * ensure_ctx() {
        if (!ctx) {
            ctx = init_context(model, cp);
            require(ctx, "context creation failed");
        }
        return ctx;
    }
    ~Engine() { release_ctx(); }
};

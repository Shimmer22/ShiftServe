// Minimal Android tool: dump / apply-mask / prefill using only libllama C API.
// No common/arg dependency — easier to link against hexagon-pkg libs.
#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void die(const char * msg) {
    fprintf(stderr, "fatal: %s\n", msg);
    exit(1);
}

static bool write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(data, 1, n, f);
    fclose(f);
    return w == n;
}

static bool write_text(const std::string & path, const std::string & s) {
    return write_file(path, s.data(), s.size());
}

static std::string read_text(const std::string & path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Seg { const char * id; const char * kind; const char * text; };

static const Seg SEGS[] = {
    {"task", "task", "Task: Open Settings and enable Wi-Fi. Stay on the current app context.\n"},
    {"history_0", "history", "Step0: Home screen visible. Icons: Settings, Camera, Browser.\n"},
    {"history_1", "history", "Step1: Tapped Settings icon at bottom left. Transitioning.\n"},
    {"history_2", "history", "Step2: Settings list shows Wi-Fi, Bluetooth, Display.\n"},
    {"history_3", "history", "Step3: Scrolled down past Network section once.\n"},
    {"history_4", "history", "Step4: Saw unrelated About phone entry; ignore for task.\n"},
    {"history_5", "history", "Step5: Returned focus toward Wireless and networks.\n"},
    {"action_prev", "action_prev", "Previous action: tap(x=540,y=820) on Wi-Fi row. Waiting for toggle UI.\n"},
};
static const int N_SEGS = (int)(sizeof(SEGS) / sizeof(SEGS[0]));

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int32_t n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (n <= 0) return {};
    std::vector<llama_token> out(n);
    n = llama_tokenize(vocab, text.c_str(), text.size(), out.data(), (int32_t)out.size(), add_special, true);
    if (n < 0) return {};
    out.resize(n);
    return out;
}

static bool decode_all(llama_context * ctx, const std::vector<llama_token> & toks, int & n_past, int n_batch) {
    for (int i = 0; i < (int)toks.size(); i += n_batch) {
        int n_eval = std::min(n_batch, (int)toks.size() - i);
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int j = 0; j < n_eval; ++j) {
            batch.token[j] = toks[i + j];
            batch.pos[j] = n_past + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (j == n_eval - 1);
        }
        batch.n_tokens = n_eval;
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
        llama_batch_free(batch);
        n_past += n_eval;
    }
    return true;
}

static bool dump_seq(llama_context * ctx, const std::string & path, size_t & nbytes) {
    size_t sz = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> buf(sz);
    size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
    if (n != sz) return false;
    nbytes = n;
    return write_file(path, buf.data(), n);
}

static int apply_mask(llama_context * ctx, const std::vector<int> & keep) {
    llama_memory_t mem = llama_get_memory(ctx);
    int n_pos = (int)keep.size();
    int dropped = 0;
    int i = 0;
    while (i < n_pos) {
        if (keep[i]) { ++i; continue; }
        int j = i;
        while (j < n_pos && !keep[j]) ++j;
        if (!llama_memory_seq_rm(mem, /*seq*/0, /*p0*/i, /*p1*/j)) {
            fprintf(stderr, "seq_rm failed [%d,%d)\n", i, j);
            return -1;
        }
        dropped += j - i;
        i = j;
    }
    return dropped;
}

static void usage() {
    fprintf(stderr,
        "kv_gui_e2e_min --phase dump|apply|prefill -m MODEL [--device HTP0] [-ngl N] [-c CTX]\n"
        "  --dump-dir DIR --keep-mask FILE --next-prompt TEXT\n");
}

int main(int argc, char ** argv) {
    std::string phase = "dump";
    std::string model_path;
    std::string dump_dir = "/data/local/tmp/exp009_kv";
    std::string keep_mask_path;
    std::string next_prompt = "Next observation: Wi-Fi toggle is visible. What is the next GUI action?";
    std::string device = "HTP0";
    int ngl = 99;
    int n_ctx = 2048;
    int n_batch = 512;
    int n_predict = 16;
    int n_threads = 4;
    bool no_mmap = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); exit(1); }
            return argv[++i];
        };
        if (a == "--phase") phase = need("--phase");
        else if (a == "-m") model_path = need("-m");
        else if (a == "--dump-dir") dump_dir = need("--dump-dir");
        else if (a == "--keep-mask") keep_mask_path = need("--keep-mask");
        else if (a == "--next-prompt") next_prompt = need("--next-prompt");
        else if (a == "--device") device = need("--device");
        else if (a == "-ngl") ngl = atoi(need("-ngl"));
        else if (a == "-c") n_ctx = atoi(need("-c"));
        else if (a == "-n") n_predict = atoi(need("-n"));
        else if (a == "-t") n_threads = atoi(need("-t"));
        else if (a == "-b") n_batch = atoi(need("-b"));
        else if (a == "--no-mmap") no_mmap = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 1; }
    }
    if (model_path.empty()) die("-m model required");

    mkdir(dump_dir.c_str(), 0755);
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    mparams.use_mmap = !no_mmap;
    // Select backend device by name (e.g. HTP0), matching --device in CLI tools.
    static ggml_backend_dev_t dev_list[2] = { nullptr, nullptr };
    if (!device.empty() && device != "CPU" && device != "none") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
        if (!dev) {
            fprintf(stderr, "warn: device '%s' not found; available:\n", device.c_str());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                fprintf(stderr, "  - %s\n", ggml_backend_dev_name(d));
            }
        } else {
            dev_list[0] = dev;
            mparams.devices = dev_list;
            fprintf(stderr, "using device %s\n", device.c_str());
        }
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("model load failed");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) die("ctx init failed");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string state_path = dump_dir + "/kv_state.bin";
    const std::string state_c_path = dump_dir + "/kv_state_compressed.bin";
    const std::string seq_path = dump_dir + "/kv_seq0.bin";
    const std::string meta_path = dump_dir + "/segments.json";

    std::vector<llama_token> all_toks;
    int n_past = 0;

    if (phase == "dump") {
        double t0 = now_ms();
        std::ostringstream meta;
        meta << "{\n  \"device_request\":\"" << device << "\",\n  \"segments\": [\n";
        for (int si = 0; si < N_SEGS; ++si) {
            std::string piece = std::string("[SEG ") + SEGS[si].id + "/" + SEGS[si].kind + "]\n" + SEGS[si].text;
            auto toks = tokenize(vocab, piece, /*add_special=*/si == 0);
            if (toks.empty()) die("tokenize failed");
            int pos0 = n_past;
            if (!decode_all(ctx, toks, n_past, n_batch)) die("decode failed");
            int pos1 = n_past;
            all_toks.insert(all_toks.end(), toks.begin(), toks.end());
            meta << "    {\"seg_id\":\"" << SEGS[si].id << "\",\"kind\":\"" << SEGS[si].kind
                 << "\",\"pos0\":" << pos0 << ",\"pos1\":" << pos1
                 << ",\"n_tokens\":" << (pos1 - pos0) << "}"
                 << (si + 1 < N_SEGS ? "," : "") << "\n";
            fprintf(stderr, "seg %s tokens=%d pos=[%d,%d)\n", SEGS[si].id, pos1 - pos0, pos0, pos1);
        }
        double t1 = now_ms();
        if (!llama_state_save_file(ctx, state_path.c_str(), all_toks.data(), all_toks.size()))
            die("state_save_file failed");
        size_t seq_bytes = 0;
        if (!dump_seq(ctx, seq_path, seq_bytes)) die("seq dump failed");
        meta << "  ],\n  \"n_tokens_total\":" << n_past
             << ",\n  \"seq_bytes\":" << seq_bytes
             << ",\n  \"prefill_history_ms\":" << (t1 - t0)
             << ",\n  \"n_pos\":" << n_past << "\n}\n";
        write_text(meta_path, meta.str());
        fprintf(stderr, "DUMP_OK n_pos=%d seq_bytes=%zu prefill_ms=%.3f\n", n_past, seq_bytes, t1 - t0);
        llama_free(ctx); llama_model_free(model);
        return 0;
    }

    // load state for apply/prefill
    {
        std::vector<llama_token> buf(1 << 16);
        size_t n_out = 0;
        const char * load_path = state_path.c_str();
        if (phase == "prefill") {
            // prefer compressed if present
            FILE * t = fopen(state_c_path.c_str(), "rb");
            if (t) { fclose(t); load_path = state_c_path.c_str(); }
        }
        if (!llama_state_load_file(ctx, load_path, buf.data(), buf.size(), &n_out))
            die("state_load_file failed");
        all_toks.assign(buf.begin(), buf.begin() + n_out);
        n_past = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        fprintf(stderr, "loaded %s n_tok=%zu n_past_est=%d\n", load_path, n_out, n_past);
    }

    if (phase == "apply") {
        if (keep_mask_path.empty()) die("--keep-mask required");
        std::ifstream in(keep_mask_path);
        std::vector<int> keep;
        int v;
        while (in >> v) keep.push_back(v);
        if (keep.empty()) die("empty mask");
        double t0 = now_ms();
        int dropped = apply_mask(ctx, keep);
        double t1 = now_ms();
        if (dropped < 0) die("apply mask failed");
        std::vector<llama_token> kept;
        for (int i = 0; i < (int)keep.size() && i < (int)all_toks.size(); ++i)
            if (keep[i]) kept.push_back(all_toks[i]);
        if (!llama_state_save_file(ctx, state_c_path.c_str(), kept.data(), kept.size()))
            die("save compressed failed");
        size_t seq_bytes = 0;
        dump_seq(ctx, dump_dir + "/kv_seq0_compressed.bin", seq_bytes);
        std::ostringstream js;
        js << "{\"apply_ms\":" << (t1 - t0) << ",\"dropped\":" << dropped
           << ",\"kept\":" << ((int)keep.size() - dropped)
           << ",\"seq_bytes_compressed\":" << seq_bytes << "}\n";
        write_text(dump_dir + "/apply_stats.json", js.str());
        fprintf(stderr, "APPLY_OK dropped=%d kept=%d apply_ms=%.3f seq_bytes=%zu\n",
                dropped, (int)keep.size() - dropped, t1 - t0, seq_bytes);
        llama_free(ctx); llama_model_free(model);
        return 0;
    }

    if (phase == "prefill") {
        auto toks = tokenize(vocab, next_prompt, false);
        double t0 = now_ms();
        // After seq_rm, positions may be sparse; append at max_pos+1
        int pos_max = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        n_past = pos_max + 1;
        if (!decode_all(ctx, toks, n_past, n_batch)) die("next prefill failed");
        double t1 = now_ms();

        // greedy decode a few tokens
        std::string gen;
        double t2 = now_ms();
        for (int i = 0; i < n_predict; ++i) {
            llama_token tok = 0;
            // greedy from logits
            float * logits = llama_get_logits_ith(ctx, -1);
            int n_vocab = llama_vocab_n_tokens(vocab);
            float best = -1e30f; int best_id = 0;
            for (int t = 0; t < n_vocab; ++t) {
                if (logits[t] > best) { best = logits[t]; best_id = t; }
            }
            tok = best_id;
            char buf[256];
            int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            if (n > 0) gen.append(buf, buf + n);
            if (llama_vocab_is_eog(vocab, tok)) break;
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0] = tok;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            batch.n_tokens = 1;
            if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); break; }
            llama_batch_free(batch);
            n_past += 1;
        }
        double t3 = now_ms();
        std::ostringstream js;
        js << "{\"next_prompt_tokens\":" << toks.size()
           << ",\"prefill_next_ms\":" << (t1 - t0)
           << ",\"decode_ms\":" << (t3 - t2)
           << ",\"gen\":\"";
        for (char c : gen) {
            if (c == '"' || c == '\\') js << '\\' << c;
            else if (c == '\n') js << "\\n";
            else js << c;
        }
        js << "\"}\n";
        write_text(dump_dir + "/prefill_stats.json", js.str());
        write_text(dump_dir + "/generation.txt", gen);
        fprintf(stderr, "PREFILL_OK prefill_ms=%.3f decode_ms=%.3f gen=%s\n", t1 - t0, t3 - t2, gen.c_str());
        llama_free(ctx); llama_model_free(model);
        return 0;
    }

    die("unknown phase");
    return 1;
}

#pragma once

#include <cerrno>
#include <filesystem>
#include <set>
#include <unistd.h>

namespace a2o::memory {

inline J read_kib(const std::string &path, const std::set<std::string> &keys = {}) {
    std::ifstream input(path);
    if (!input) return {{"error", std::strerror(errno)}};
    J values = J::object();
    for (std::string line; std::getline(input, line);) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = line.substr(0, colon);
        if (!keys.empty() && !keys.count(key)) continue;
        std::istringstream fields(line.substr(colon + 1));
        uint64_t value;
        std::string unit;
        if (fields >> value >> unit && unit == "kB") values[key + "_kib"] = value;
    }
    if (input.bad()) values["error"] = "read failed";
    return values;
}

inline J system_dmabuf() {
    const std::string path = "/sys/kernel/dmabuf/buffers";
    std::error_code error;
    std::filesystem::directory_iterator it(path, error), end;
    if (error) return {{"source", path}, {"error", error.message()}};
    uint64_t bytes = 0;
    size_t count = 0, unreadable = 0;
    for (; it != end && !error; it.increment(error)) {
        uint64_t size;
        if (std::ifstream(it->path() / "size") >> size) {
            bytes += size;
            ++count;
        } else {
            ++unreadable;
        }
    }
    J result = {{"source", path}, {"buffer_count", count}, {"unreadable_count", unreadable}};
    if (error || unreadable) {
        result["error"] = error ? error.message() : "incomplete buffer list";
        result["partial_bytes"] = bytes;
    } else {
        result["bytes"] = bytes;
    }
    return result;
}

inline J dma_mappings(const std::set<uint64_t> &dma_inodes) {
    std::ifstream input("/proc/self/smaps");
    if (!input) return {{"error", std::strerror(errno)}};
    uint64_t identified_pss = 0, identified_rss = 0;
    uint64_t all_size = 0;
    size_t mappings = 0;
    bool dma = false, identified = false;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.rfind("Pss:", 0) == 0 || line.rfind("Rss:", 0) == 0) {
            if (!identified) continue;
            uint64_t value;
            if (!(std::istringstream(line.substr(4)) >> value))
                return {{"error", "invalid RSS/PSS in smaps"}};
            if (line.front() == 'P') identified_pss += value;
            else identified_rss += value;
        } else if (dma && line.rfind("Size:", 0) == 0) {
            uint64_t value;
            if (!(std::istringstream(line.substr(5)) >> value))
                return {{"error", "invalid DMA-BUF size in smaps"}};
            all_size += value;
        } else if (line.find('-') < line.find(' ') && line.find(':') > line.find(' ')) {
            std::istringstream header(line);
            std::string address, protection, offset, device, name;
            uint64_t inode;
            dma = identified = false;
            if (!(header >> address >> protection >> offset >> device >> inode >> name)) continue;
            dma = name.rfind("/dmabuf", 0) == 0 || name.rfind("/dma-buf", 0) == 0;
            identified = dma && dma_inodes.count(inode);
            if (dma) ++mappings;
        }
    }
    if (input.bad()) return {{"error", "read failed"}};
    return {{"source", "/proc/self/smaps"}, {"mapping_count", mappings}, {"mapped_size_kib", all_size},
            {"identified_pss_kib", identified_pss}, {"identified_rss_kib", identified_rss}};
}

inline J process_driver_buffers() {
    const std::string path = "/proc/self/fd";
    std::error_code error;
    std::filesystem::directory_iterator it(path, error), end;
    if (error) return {{"source", path}, {"error", error.message()}};
    std::set<uint64_t> dma_inodes;
    uint64_t dma_bytes = 0;
    size_t dma_unsized = 0;
    J fastrpc = J::array();
    for (; it != end && !error; it.increment(error)) {
        std::error_code link_error;
        const auto target = std::filesystem::read_symlink(it->path(), link_error).string();
        if (link_error) continue;
        if (target.find("fastrpc") != std::string::npos || target.find("adsprpc") != std::string::npos)
            fastrpc.push_back(target);
        // DMA-BUF fdinfo exposes size (bytes), ino and exp_name on kernels with buffer statistics.
        std::ifstream info("/proc/self/fdinfo/" + it->path().filename().string());
        if (!info) continue;
        uint64_t size = 0, ino = 0;
        bool exported = false, sized = false;
        for (std::string line; std::getline(info, line);) {
            std::istringstream fields(line);
            std::string key;
            fields >> key;
            if (key == "size:") sized = bool(fields >> size);
            if (key == "ino:") fields >> ino;
            if (key == "exp_name:") exported = true;
        }
        if (exported && sized && ino) {
            if (dma_inodes.insert(ino).second) dma_bytes += size;
        } else if (exported || target.find("dmabuf") != std::string::npos ||
                   target.find("dma-buf") != std::string::npos) {
            ++dma_unsized;
        }
    }
    J result = {{"source", "/proc/self/fdinfo"}, {"dma_buf_sized_count", dma_inodes.size()},
                {"dma_buf_unsized_fds", dma_unsized}, {"fastrpc_device_fds", fastrpc}};
    if (error) result["error"] = error.message();
    // Positively identified exported buffers only. Closed fds and driver-owned buffers are outside this scope.
    if (!error) result["dma_buf_identified_bytes"] = dma_bytes;
    result["dma_mappings"] = dma_mappings(dma_inodes);
    return result;
}

inline J fastrpc_stats() {
    for (const char *root : {"/sys/kernel/debug/fastrpc", "/sys/kernel/debug/adsprpc", "/d/fastrpc"}) {
        std::error_code error;
        std::filesystem::directory_iterator it(root, error), end;
        if (error) continue;
        for (; it != end && !error; it.increment(error)) {
            const auto name = it->path().filename().string();
            const auto pid = std::to_string(getpid());
            if (name != pid && name.rfind(pid + "-", 0) != 0) continue;
            std::ifstream input(it->path());
            if (!input) continue;
            std::ostringstream raw;
            raw << input.rdbuf();
            if (!raw.str().empty()) return {{"source", it->path().string()}, {"raw", raw.str()}};
        }
    }
    return {{"error", "no readable per-process FastRPC debugfs entry"}};
}

class Ledger {
    bool enabled_;
    bool trim_;
    J samples_ = J::array();
    double overhead_ms_ = 0;

public:
    Ledger(bool enabled, bool trim): enabled_(enabled), trim_(trim) {}

    void sample(const char *stage) {
        if (!enabled_) return;
        const double begin = now_ms();
        J process = resource_sample();
        const auto rollup = read_kib("/proc/self/smaps_rollup");
        if (rollup.contains("Pss_kib")) process["Pss_kib"] = rollup["Pss_kib"];
        J entry = {{"stage", stage}, {"monotonic_ms", begin}, {"process", process},
                   {"smaps_rollup", rollup},
                   {"meminfo", read_kib("/proc/meminfo",
                       {"MemAvailable", "MemFree", "Cached", "Shmem", "Slab", "SReclaimable",
                        "SUnreclaim", "SwapTotal", "SwapFree", "SwapCached"})},
                   {"system_dma_buf", system_dmabuf()}, {"driver_buffers", process_driver_buffers()},
                   {"fastrpc", fastrpc_stats()}};
        const auto &driver = entry.at("driver_buffers");
        if (driver.contains("dma_buf_identified_bytes") && driver.at("dma_mappings").contains("identified_pss_kib") &&
            process.contains("Pss_kib")) {
            // Add the identified DMA-BUF backing once, removing any contribution already in PSS.
            entry["pss_plus_identified_dma_kib"] = process.at("Pss_kib").get<double>() +
                driver.at("dma_buf_identified_bytes").get<double>() / 1024.0 -
                driver.at("dma_mappings").at("identified_pss_kib").get<double>();
        }
        if (!samples_.empty()) {
            const auto &baseline = samples_.front().at("meminfo");
            const auto &current = entry.at("meminfo");
            if (baseline.contains("MemAvailable_kib") && current.contains("MemAvailable_kib"))
                entry["memavailable_drop_kib"] = baseline.at("MemAvailable_kib").get<int64_t>() -
                    current.at("MemAvailable_kib").get<int64_t>();
        }
        entry["collection_ms"] = now_ms() - begin;
        // Preserve checkpoints even if a later inference stage fails.
        std::cerr << ("A2O_MEMORY " + J{{"kv_hwm_trim", trim_}, {"sample", entry}}.dump() + "\n");
        overhead_ms_ += now_ms() - begin;
        samples_.push_back(std::move(entry));
    }

    double overhead_ms() const { return overhead_ms_; }

    J checkpoint_process() const {
        return enabled_ ? samples_.back().at("process") : J(resource_sample());
    }

    J result() {
        return {{"version", 1}, {"kv_hwm_trim", trim_}, {"collection_ms", overhead_ms_},
                {"samples", std::move(samples_)}};
    }
};

} // namespace a2o::memory

#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// 单个候选拥有独立目的上下文；调用者必须保证 join 前不复用/销毁它。
// 取消仅在 work 的安全检查点生效，不中断正在执行的 backend API。
class AsyncPreparation {
public:
    using Checkpoint = std::function<bool()>;
    using Work = std::function<bool(const Checkpoint &)>;
    struct Decision { std::string backend, reason; };
    const uint64_t version;

    AsyncPreparation(uint64_t v, Work work, bool held = false)
        : version(v), released_(!held), origin_(Clock::now()) {
        mark("tool_started");
        worker_ = std::thread([this, work = std::move(work)] {
            try {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return released_ || cancelled_; });
                    if (cancelled_) { finish_locked("cancelled"); return; }
                    state_ = "running";
                    event_locked("prepare_started");
                }
                bool completed = work([this] { return checkpoint(); });
                std::lock_guard<std::mutex> lock(mutex_);
                finish_locked(completed && !cancelled_ ? "ready" : "cancelled");
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = std::current_exception();
                finish_locked("failed");
            }
        });
    }
    AsyncPreparation(const AsyncPreparation &) = delete;
    AsyncPreparation & operator=(const AsyncPreparation &) = delete;
    ~AsyncPreparation() { cancel(); if (worker_.joinable()) worker_.join(); }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true; cv_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cancelled_) {
            cancelled_ = true; event_locked("cancel_requested");
            if (state_ == "ready") state_ = "cancelled";
        }
        cv_.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return done_; });
        if (error_) std::rethrow_exception(error_);
    }
    void join() {
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
    }
    Decision bind(uint64_t current_version) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_); // 失败不伪装为策略回退。
        std::string reason = cancelled_ || state_ == "cancelled" ? "cancelled" :
            version != current_version ? "stale" : state_ == "ready" ? "ready" : "pending";
        event_locked("bind_" + reason);
        return {reason == "ready" ? "htp" : "cpu", reason};
    }
    void mark(const std::string & event) {
        std::lock_guard<std::mutex> lock(mutex_); event_locked(event);
    }
    std::string state() const {
        std::lock_guard<std::mutex> lock(mutex_); return state_;
    }
    std::vector<std::pair<std::string, double>> events() const {
        std::lock_guard<std::mutex> lock(mutex_); return events_;
    }
private:
    using Clock = std::chrono::steady_clock;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool released_, cancelled_ = false, done_ = false;
    std::string state_ = "pending";
    std::exception_ptr error_;
    Clock::time_point origin_;
    std::vector<std::pair<std::string, double>> events_;
    std::thread worker_;
    bool checkpoint() const {
        std::lock_guard<std::mutex> lock(mutex_); return !cancelled_;
    }
    void event_locked(const std::string & event) {
        events_.push_back({event, std::chrono::duration<double, std::milli>(Clock::now() - origin_).count()});
    }
    void finish_locked(const std::string & state) {
        state_ = state; done_ = true; event_locked("prepare_" + state); cv_.notify_all();
    }
};

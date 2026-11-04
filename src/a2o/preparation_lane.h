#pragma once
#include "async_preparation.h"
#include <memory>
#include <stdexcept>

// 会话级单 lane；只有 retired 且线程真正退出后才回收，逐轮不等待。
class PreparationLane {
public:
    bool start(uint64_t version, AsyncPreparation::Work work) {
        poll();
        if (task_) return false;
        task_ = std::make_unique<AsyncPreparation>(version, std::move(work));
        retired_ = false; ++started_; return true;
    }
    void poll() {
        if (!task_) return;
        task_->check_error();
        if (retired_ && task_->exited()) {
            task_->join(); task_.reset(); ++reaped_;
        }
    }
    std::string availability(uint64_t version) {
        poll();
        if (!task_) return "absent";
        if (retired_) return "busy";
        auto reason = task_->bind(version).reason;
        return reason == "ready" && !task_->exited() ? "pending" : reason;
    }
    void claim(uint64_t version) {
        if (availability(version) != "ready") throw std::runtime_error("HTP lane is not claimable");
        task_->join(); task_.reset(); ++reaped_;
    }
    void retire() { if (task_) { task_->cancel(); retired_ = true; } }
    void close() {
        retire();
        if (task_) { task_->join(); task_.reset(); ++reaped_; }
    }
    bool occupied() const { return bool(task_); }
    uint64_t version() const { return task_ ? task_->version : 0; }
    std::string progress() const { return !task_ ? "absent" : retired_ ? "retired" : task_->state(); }
    void wait_for_test() {
        if (!task_ || retired_) throw std::runtime_error("no active preparation to wait for");
        task_->wait(); task_->join();
    }
    size_t started() const { return started_; }
    size_t reaped() const { return reaped_; }
    // 异常退出仍先销毁任务再销毁 context；正式 close 负责传播错误。
    ~PreparationLane() { try { close(); } catch (...) {} }
private:
    std::unique_ptr<AsyncPreparation> task_;
    bool retired_ = false;
    size_t started_ = 0, reaped_ = 0;
};

// 机制测试专用：模拟不可抢占调用返回边界，不模拟时间。
class PreparationBarrier {
public:
    void enter() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true; cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock, [&] { return entered_ || error_; });
        if(error_) std::rethrow_exception(error_);
    }
    void fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(mutex_); error_=error; cv_.notify_all();
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_); released_ = true; cv_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false, released_ = false;
    std::exception_ptr error_;
};

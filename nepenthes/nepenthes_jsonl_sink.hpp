#pragma once

#include "nepenthes_trace.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>

namespace nepenthes {

// Bounded MPMC event queue. Inference threads only copy a fixed-size Event and
// perform atomics; JSON formatting and disk I/O happen on the sink thread.
class JsonlSink final {
    static constexpr std::size_t capacity = 8192;
    struct Cell {
        std::atomic<std::size_t> sequence{0};
        Event event{};
    };

public:
    explicit JsonlSink(const std::string& path, std::string run_id = "tandem-standalone") : output_(path, std::ios::out | std::ios::app), run_id_(std::move(run_id)) {
        for (std::size_t i = 0; i < capacity; ++i) {
            queue_[i].sequence.store(i, std::memory_order_relaxed);
        }
        if (!output_) {
            throw std::runtime_error("cannot open NEPENTHES JSONL sink: " + path);
        }
        running_.store(true, std::memory_order_release);
        writer_ = std::thread([this] { drain(); });
        Trace::install(&JsonlSink::callback, this);
    }

    JsonlSink(const JsonlSink&) = delete;
    JsonlSink& operator=(const JsonlSink&) = delete;

    ~JsonlSink() { close(); }

    void close() noexcept {
        Trace::disable();
        running_.store(false, std::memory_order_release);
        if (writer_.joinable()) writer_.join();
        output_.flush();
    }

    std::uint64_t dropped() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    static void callback(const Event& event, void* opaque) noexcept {
        static_cast<JsonlSink*>(opaque)->push(event);
    }

    void push(const Event& event) noexcept {
        std::size_t pos = enqueue_.load(std::memory_order_relaxed);
        for (;;) {
            Cell& cell = queue_[pos & (capacity - 1)];
            const std::size_t seq = cell.sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    cell.event = event;
                    cell.sequence.store(pos + 1, std::memory_order_release);
                    return;
                }
            } else if (diff < 0) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return;
            } else {
                pos = enqueue_.load(std::memory_order_relaxed);
            }
        }
    }

    bool pop(Event& event) noexcept {
        std::size_t pos = dequeue_.load(std::memory_order_relaxed);
        Cell& cell = queue_[pos & (capacity - 1)];
        const std::size_t seq = cell.sequence.load(std::memory_order_acquire);
        const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
        if (diff != 0) return false;
        dequeue_.store(pos + 1, std::memory_order_relaxed);
        event = cell.event;
        cell.sequence.store(pos + capacity, std::memory_order_release);
        return true;
    }

    static const char* name(EventKind kind) noexcept {
        switch (kind) {
            case EventKind::request_start: return "request_start";
            case EventKind::request_end: return "request_end";
            case EventKind::model_load: return "model_load";
            case EventKind::model_unload: return "model_unload";
            case EventKind::routing: return "routing";
            case EventKind::cache_hit: return "cache_hit";
            case EventKind::cache_miss: return "cache_miss";
            case EventKind::eviction: return "eviction";
            case EventKind::prefetch: return "prefetch";
            case EventKind::kv_cache: return "kv_cache";
            case EventKind::allocation: return "allocation";
            case EventKind::free: return "free";
            case EventKind::transfer: return "transfer";
            case EventKind::kernel: return "kernel";
            case EventKind::sync: return "sync";
            case EventKind::stall: return "stall";
            case EventKind::oom: return "oom";
            case EventKind::fallback: return "fallback";
            case EventKind::error: return "error";
        }
        return "error";
    }

    void drain() noexcept {
        Event event{};
        while (running_.load(std::memory_order_acquire) ||
               dequeue_.load(std::memory_order_relaxed) != enqueue_.load(std::memory_order_relaxed)) {
            if (!pop(event)) {
                std::this_thread::sleep_for(std::chrono::microseconds(250));
                continue;
            }
            const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const auto event_id = next_event_id_.fetch_add(1, std::memory_order_relaxed);
            output_ << "{\\"schema_version\\":\\"1.0\\",\\"run_id\\":\\"" << run_id_
                    << "\\",\\"event_id\\":\\"tandem-" << event_id
                    << "\\",\\"source\\":\\"tandem.cpp\\",\\"timestamp_unix_ns\\":" << wall_ns
                    << ",\\"monotonic_ns\\":" << event.monotonic_ns
                    << ",\\"request_id\\":\\"" << event.request_id
                    << "\\",\\"event_kind\\":\\"" << name(event.kind)
                    << "\\",\\"payload\\":{\\"object_id\\":" << event.object_id
                    << ",\\"bytes\\":" << event.bytes << ",\\"duration_ns\\":" << event.duration_ns
                    << ",\\"layer\\":" << event.layer << ",\\"expert\\":" << event.expert
                    << ",\\"stream_id\\":" << event.stream_id << ",\\"flags\\":" << event.flags << "}}\\n";
        output_.flush();
    }

    std::string run_id_;\n    std::atomic<std::uint64_t> next_event_id_{1};\n    std::array<Cell, capacity> queue_{};
    std::atomic<std::size_t> enqueue_{0};
    std::atomic<std::size_t> dequeue_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<bool> running_{false};
    std::ofstream output_;
    std::thread writer_;
};

} // namespace nepenthes

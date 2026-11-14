#pragma once

#include "nepenthes_trace.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace nepenthes {

// One process-wide sink. Construct it before worker threads and close it only
// after all event producers have stopped.
class JsonlSink final {
    static constexpr std::size_t capacity = 8192;
    struct Cell {
        std::atomic<std::size_t> sequence{0};
        Event event{};
    };

public:
    explicit JsonlSink(const std::string & path, std::string run_id = "tandem")
        : run_id_json_(escape_json(run_id)),
          output_(path, std::ios::out | std::ios::app) {
        for (std::size_t i = 0; i < capacity; ++i) {
            queue_[i].sequence.store(i, std::memory_order_relaxed);
        }
        if (!output_) {
            throw std::runtime_error("cannot open NEPENTHES JSONL sink: " + path);
        }
        running_.store(true, std::memory_order_release);
        writer_ = std::thread([this] { drain(); });
        active_.store(this, std::memory_order_release);
        set_sink(&JsonlSink::callback);
    }

    JsonlSink(const JsonlSink &) = delete;
    JsonlSink & operator=(const JsonlSink &) = delete;
    ~JsonlSink() { close(); }

    void close() noexcept {
        bool expected = false;
        if (!closed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
        set_sink(nullptr);
        active_.store(nullptr, std::memory_order_release);
        running_.store(false, std::memory_order_release);
        if (writer_.joinable()) writer_.join();
        output_.flush();
        if (!output_) write_errors_.fetch_add(1, std::memory_order_relaxed);
    }

    std::uint64_t dropped() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }
    std::uint64_t write_errors() const noexcept {
        return write_errors_.load(std::memory_order_relaxed);
    }

private:
    static std::string escape_json(const std::string & value) {
        static constexpr char hex[] = "0123456789abcdef";
        std::string out;
        out.reserve(value.size());
        for (const unsigned char ch : value) {
            switch (ch) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (ch < 0x20) {
                        out += "\\u00";
                        out += hex[(ch >> 4) & 0x0f];
                        out += hex[ch & 0x0f];
                    } else {
                        out += static_cast<char>(ch);
                    }
            }
        }
        return out;
    }

    static const char * type_name(EventType type) noexcept {
        switch (type) {
            case EventType::request_start: return "request_start";
            case EventType::request_end: return "request_end";
            case EventType::model_load: return "model_load";
            case EventType::model_unload: return "model_unload";
            case EventType::routing: return "routing";
            case EventType::cache_hit: return "cache_hit";
            case EventType::cache_miss: return "cache_miss";
            case EventType::eviction: return "eviction";
            case EventType::prefetch: return "prefetch";
            case EventType::kv_cache: return "kv_cache";
            case EventType::allocation: return "allocation";
            case EventType::free_event: return "free";
            case EventType::transfer: return "transfer";
            case EventType::kernel: return "kernel";
            case EventType::sync: return "sync";
            case EventType::stall: return "stall";
            case EventType::oom: return "oom";
            case EventType::fallback: return "fallback";
            case EventType::error: return "error";
        }
        return "error";
    }

    static void callback(const Event & event) noexcept {
        JsonlSink * sink = active_.load(std::memory_order_acquire);
        if (sink) sink->push(event);
    }

    void push(const Event & event) noexcept {
        std::size_t pos = enqueue_.load(std::memory_order_relaxed);
        for (;;) {
            Cell & cell = queue_[pos & (capacity - 1)];
            const std::size_t seq = cell.sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
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

    bool pop(Event & event) noexcept {
        const std::size_t pos = dequeue_.load(std::memory_order_relaxed);
        Cell & cell = queue_[pos & (capacity - 1)];
        const std::size_t seq = cell.sequence.load(std::memory_order_acquire);
        const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);
        if (diff != 0) return false;
        dequeue_.store(pos + 1, std::memory_order_relaxed);
        event = cell.event;
        cell.sequence.store(pos + capacity, std::memory_order_release);
        return true;
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
            const auto id = next_id_.fetch_add(1, std::memory_order_relaxed);
            output_ << "{\"schema_version\":\"1.0\",\"run_id\":\"" << run_id_json_
                    << "\",\"event_id\":\"tandem-" << id
                    << "\",\"source\":\"tandem.cpp\",\"timestamp_unix_ns\":" << wall_ns
                    << ",\"monotonic_ns\":" << event.monotonic_ns
                    << ",\"request_id\":" << event.request_id
                    << ",\"event_kind\":\"" << type_name(event.type)
                    << "\",\"label\":\"" << escape_json(event.label ? event.label : "")
                    << "\",\"value0\":" << event.value0
                    << ",\"value1\":" << event.value1
                    << ",\"duration_ns\":" << event.duration_ns
                    << ",\"flags\":" << event.flags << "}\n";
            if (!output_) write_errors_.fetch_add(1, std::memory_order_relaxed);
        }
        output_.flush();
        if (!output_) write_errors_.fetch_add(1, std::memory_order_relaxed);
    }

    inline static std::atomic<JsonlSink *> active_{nullptr};
    std::string run_id_json_;
    std::atomic<std::uint64_t> next_id_{1};
    std::array<Cell, capacity> queue_{};
    std::atomic<std::size_t> enqueue_{0};
    std::atomic<std::size_t> dequeue_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> write_errors_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> closed_{false};
    std::ofstream output_;
    std::thread writer_;
};

} // namespace nepenthes

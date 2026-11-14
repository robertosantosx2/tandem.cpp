#pragma once

// NEPENTHES observability hook for tandem.cpp / llama.cpp.
// C++17, header-only, no heap allocation on the emit path.
//
// The sink must be installed before worker threads start and removed only
// after they have stopped. The sink itself must be non-blocking and
// thread-safe. Labels are borrowed string literals and must be consumed
// synchronously by the sink.

#include <atomic>
#include <chrono>
#include <cstdint>

namespace nepenthes {

enum class EventType : std::uint8_t {
    request_start, request_end,
    model_load, model_unload,
    routing, cache_hit, cache_miss, eviction, prefetch,
    kv_cache, allocation, free_event, transfer, kernel,
    sync, cpu_work, stall, oom, fallback, error,
};

enum EventFlags : std::uint16_t {
    phase_instant = 0,
    phase_begin = 1u << 0,
    phase_end = 1u << 1,
    flag_failed = 1u << 2,
    flag_host = 1u << 3,
    flag_device = 1u << 4,
};

struct Event {
    std::uint64_t monotonic_ns = 0;
    std::uint64_t duration_ns = 0;
    std::uint64_t request_id = 0; // zero means not correlated
    std::uint64_t value0 = 0;
    std::uint64_t value1 = 0;
    EventType type = EventType::error;
    std::uint16_t flags = phase_instant;
    const char * label = nullptr; // borrowed; sink consumes during callback
};

using Sink = void (*)(const Event &) noexcept;

inline std::atomic<Sink> & sink_slot() noexcept {
    static std::atomic<Sink> slot{nullptr};
    return slot;
}

inline std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Install/disable only at a controlled lifecycle boundary, never while
// producer threads are concurrently being created or destroyed.
inline void set_sink(Sink sink) noexcept {
    sink_slot().store(sink, std::memory_order_release);
}

inline void emit(EventType type, const char * label = nullptr,
                 std::uint64_t value0 = 0, std::uint64_t value1 = 0,
                 std::uint16_t flags = phase_instant,
                 std::uint64_t request_id = 0,
                 std::uint64_t duration_ns = 0) noexcept {
    const Sink sink = sink_slot().load(std::memory_order_acquire);
    if (sink == nullptr) return;
    Event event;
    event.monotonic_ns = now_ns();
    event.duration_ns = duration_ns;
    event.request_id = request_id;
    event.value0 = value0;
    event.value1 = value1;
    event.type = type;
    event.flags = flags;
    event.label = label;
    sink(event);
}

class ScopedEvent {
public:
    ScopedEvent(EventType type, const char * label,
                std::uint64_t value0 = 0, std::uint64_t value1 = 0,
                std::uint64_t request_id = 0) noexcept
        : type_(type), label_(label), value0_(value0), value1_(value1),
          request_id_(request_id), start_ns_(now_ns()) {
        emit(type_, label_, value0_, value1_, phase_begin, request_id_);
    }

    ScopedEvent(const ScopedEvent &) = delete;
    ScopedEvent & operator=(const ScopedEvent &) = delete;

    ~ScopedEvent() noexcept {
        const std::uint64_t end_ns = now_ns();
        emit(type_, label_, value0_, value1_, phase_end, request_id_,
             end_ns >= start_ns_ ? end_ns - start_ns_ : 0);
    }

private:
    EventType type_;
    const char * label_;
    std::uint64_t value0_;
    std::uint64_t value1_;
    std::uint64_t request_id_;
    std::uint64_t start_ns_;
};

} // namespace nepenthes

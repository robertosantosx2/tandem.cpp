#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

namespace nepenthes {
enum class EventKind : std::uint16_t {
    request_start, request_end, model_load, model_unload, routing,
    cache_hit, cache_miss, eviction, prefetch, kv_cache, allocation,
    free, transfer, kernel, sync, stall, oom, fallback, error
};
constexpr std::uint32_t phase_begin = 1u << 0;
constexpr std::uint32_t phase_end = 1u << 1;
struct Event {
    EventKind kind{EventKind::error};
    std::uint64_t monotonic_ns{0}, request_id{0}, object_id{0}, bytes{0}, duration_ns{0};
    std::uint32_t layer{0}, expert{0}, stream_id{0}, flags{0};
};
using Sink = void (*)(const Event&, void*) noexcept;
inline std::uint64_t monotonic_now_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
class Trace final {
public:
    Trace() = delete;
    static void install(Sink sink, void* context = nullptr) noexcept {
        context_.store(context, std::memory_order_relaxed);
        sink_.store(sink, std::memory_order_release);
    }
    static void disable() noexcept {
        sink_.store(nullptr, std::memory_order_release);
        context_.store(nullptr, std::memory_order_relaxed);
    }
    static bool enabled() noexcept { return sink_.load(std::memory_order_acquire) != nullptr; }
    static void emit(Event event) noexcept {
        const Sink sink = sink_.load(std::memory_order_acquire);
        if (!sink) return;
        if (!event.monotonic_ns) event.monotonic_ns = monotonic_now_ns();
        sink(event, context_.load(std::memory_order_relaxed));
    }
private:
    inline static std::atomic<Sink> sink_{nullptr};
    inline static std::atomic<void*> context_{nullptr};
};
class ScopedEvent final {
public:
    explicit ScopedEvent(Event event) noexcept : event_(event), started_(monotonic_now_ns()) {}
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
    ~ScopedEvent() noexcept {
        event_.monotonic_ns = started_;
        const auto end = monotonic_now_ns();
        event_.duration_ns = end >= started_ ? end - started_ : 0;
        Trace::emit(event_);
    }
private:
    Event event_;
    std::uint64_t started_;
};
} // namespace nepenthes

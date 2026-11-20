#pragma once

// Optional NVTX point markers for explicit decode-batch windows.
// No NVTX headers or link-time NVTX dependency: resolve nvtxMarkA at runtime.
// Enabled only when NEPENTHES_NVTX=1 and a usable libnvToolsExt is available.

#include "nepenthes_trace.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace nepenthes {
namespace nvtx_detail {

using MarkFn = void (*)(const char *);

inline MarkFn mark_function() noexcept {
    static std::once_flag once;
    static MarkFn mark = nullptr;
    static void * library = nullptr;
    std::call_once(once, [] {
        mark = reinterpret_cast<MarkFn>(dlsym(RTLD_DEFAULT, "nvtxMarkA"));
        if (mark == nullptr) {
            library = dlopen("libnvToolsExt.so.1", RTLD_LAZY | RTLD_LOCAL);
            if (library != nullptr) {
                mark = reinterpret_cast<MarkFn>(dlsym(library, "nvtxMarkA"));
            }
        }
    });
    (void) library; // Keep the library loaded for the process lifetime.
    return mark;
}

inline bool enabled() noexcept {
    static const bool value = [] {
        const char * setting = std::getenv("NEPENTHES_NVTX");
        return setting != nullptr && std::strcmp(setting, "1") == 0;
    }();
    return value;
}

inline bool format_batch_message(char * output, std::size_t capacity,
                                 const char * phase, std::uint64_t token) noexcept {
    if (output == nullptr || capacity == 0 || phase == nullptr || token == 0 ||
        (std::strcmp(phase, "start") != 0 && std::strcmp(phase, "end") != 0)) {
        return false;
    }
    const int written = std::snprintf(output, capacity,
        "nepenthes.batch.%s token=%llu", phase,
        static_cast<unsigned long long>(token));
    return written > 0 && static_cast<std::size_t>(written) < capacity;
}

inline bool point(const char * phase, std::uint64_t token) noexcept {
    if (!enabled() || token == 0) return false;
    const MarkFn mark = mark_function();
    if (mark == nullptr) return false;

    char message[96]{};
    if (!format_batch_message(message, sizeof(message), phase, token)) return false;

    const std::uint64_t before = now_ns();
    mark(message);
    const std::uint64_t after = now_ns();
    const char * label = std::strcmp(phase, "start") == 0 ? "batch.start" : "batch.end";
    emit_with_token(EventType::nvtx_marker, label, before, after, phase_instant,
                    0, 0, token);
    return true;
}

} // namespace nvtx_detail

class BatchNvtxScope {
public:
    explicit BatchNvtxScope(std::uint64_t token) noexcept
        : token_(token), active_(nvtx_detail::point("start", token)) {}
    BatchNvtxScope(const BatchNvtxScope &) = delete;
    BatchNvtxScope & operator=(const BatchNvtxScope &) = delete;
    ~BatchNvtxScope() noexcept {
        if (active_) nvtx_detail::point("end", token_);
    }
private:
    std::uint64_t token_;
    bool active_;
};

} // namespace nepenthes

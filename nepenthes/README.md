# NEPENTHES instrumentation for tandem.cpp

This directory defines the low-overhead event contract shared with the
LEONES universal collector.

## Current scope

- \`nepenthes_trace.hpp\`: C++17, header-only, allocation-free emit interface.
- \`test_nepenthes_trace.cpp\`: standalone smoke test for disabled/enabled
  sinks, event metadata and scoped duration events.

The hook is **not yet wired into llama.cpp/tandem.cpp runtime call sites**.
That integration is a separate patch layer because this repository carries
\`tandem.patch\` against the pinned llama.cpp base rather than a full source
checkout. Do not interpret a passing smoke test as proof that runtime events
are being captured.

## Event contract

Events carry a steady-clock timestamp in nanoseconds, optional duration,
optional request correlation ID, two numeric values, a type, flags and a
borrowed static label. A zero request ID means correlation is unavailable;
the instrumentation must not synthesize request identity.

The sink must consume the event synchronously, must not retain the borrowed
label pointer, and must be non-blocking and thread-safe. Install it before
worker threads start; disable it after workers stop. Do not hot-swap the sink
while producers are active.

Suggested call-site mapping:

| Runtime area | Event types |
|---|---|
| request lifecycle | request_start, request_end |
| model lifecycle | model_load, model_unload |
| MoE routing | routing |
| expert cache | cache_hit, cache_miss, eviction, prefetch |
| KV cache | kv_cache |
| memory | allocation, free_event, oom |
| CPU/GPU movement | transfer |
| kernel dispatch | kernel |
| synchronization / CPU worker waits | sync, stall |
| degraded execution | fallback, error |

CUDA activity timestamps imported from Nsight remain in their original
clock domain. Do not compare them directly with steady-clock event times
without an explicit clock calibration.

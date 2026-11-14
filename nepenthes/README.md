# NEPENTHES instrumentation for tandem.cpp

This directory contains the low-overhead event contract, JSONL sink, runtime
patch layers and tests shared with the LEONES universal collector.

## Active capture

The build applies the patch layers in this order:

1. `tandem.patch` — Tandem.cpp engine changes against the pinned llama.cpp base.
2. `nepenthes/runtime.patch` — trace hook plus initial graph, synchronization,
   MoE cache and server routing callsites.
3. `nepenthes/runtime-sink.patch` — bounded asynchronous JSONL sink and
   initial MoE CPU-worker wait/compute/publication and lookahead-prefetch probes.

Enable capture before starting the server:

```bash
export NEPENTHES_TRACE_JSONL="$HOME/nepenthes-run.jsonl"
export NEPENTHES_RUN_ID="qwen35-35b-$(date +%Y%m%d-%H%M%S)"
```

The sink is created by the MoE cache when the engine initializes it. It is
disabled by default. Its queue is bounded; overflow increments a dropped-event
counter. The writer thread owns file I/O. The sink is closed after the MoE
worker/helper threads have stopped.

## Current coverage

| Runtime area | Status |
|---|---|
| Graph compute duration | Hook in runtime patch; graph-level, not individual CUDA kernels |
| Context synchronization | Hook in runtime patch |
| MoE cache hit/miss and eviction | Hook in runtime patch |
| Server task-to-slot routing | Hook in runtime patch |
| JSONL sink lifecycle | Active when NEPENTHES_TRACE_JSONL is set |
| MoE CPU worker wait/compute/done | Initial callsites in runtime-sink.patch |
| CPU output publication | Initial callsites in runtime-sink.patch |
| Lookahead prefetch | Initial begin/end events and byte estimate |
| Request start/end | Not yet wired |
| KV-cache operations | Not yet wired |
| Allocation/free/OOM | Not yet wired |
| Transfer events | Nsight importer captures CUDA Runtime memcpy activities; direct sink callsites pending |
| Individual CUDA kernels | Nsight importer captures kernel activities; direct sink callsites pending |
| Fallback/degraded execution | Not yet wired |

Passing the smoke tests does not prove that all requested runtime categories
are captured. CI checks the C++ sink and that all patch layers apply to the
pinned base; it does not build the full CUDA server or run inference.

## Event and clock rules

Events carry steady-clock timestamps and optional duration/request correlation.
A zero request ID means correlation is unavailable; identity must never be
synthesized. CUDA activity timestamps imported from Nsight remain in their
original clock domain and must not be compared directly with steady-clock
events without explicit clock calibration.

The sink callback is designed to be non-blocking and thread-safe. The queue
has a fixed capacity and reports drops. Runtime instrumentation must preserve
the source engine's semantics and must not add blocking I/O to worker threads.

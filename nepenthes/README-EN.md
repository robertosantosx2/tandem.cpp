# NEPENTHES instrumentation for tandem.cpp

This directory contains the event contract, bounded JSONL sink, deterministic runtime-probe applier and tests shared with the LEONES universal collector.

## Apply and activate capture

Apply the source layers in this order:

1. `tandem.patch` — Tandem.cpp engine changes against the pinned llama.cpp base.
2. `nepenthes/runtime.patch` — trace hook plus graph, synchronization, MoE cache and server-routing call sites.
3. `python3 nepenthes/apply_runtime_probes.py /path/to/llama.cpp` — copies the JSONL sink into `src/` and adds the remaining runtime probes. The script uses exact source anchors and fails closed if the pinned source has drifted.

Set the capture variables before starting the server:

```bash
export NEPENTHES_TRACE_JSONL="$HOME/nepenthes-run.jsonl"
export NEPENTHES_RUN_ID="qwen35-35b-$(date +%Y%m%d-%H%M%S)"
```

The sink is created lazily at the first graph-compute entry. Capture is disabled when `NEPENTHES_TRACE_JSONL` is unset. The writer thread owns file I/O; producer callbacks only enqueue events. Queue capacity is fixed and overflow is counted. Shutdown disables new callbacks, waits for callbacks already in flight, drains the bounded queue, and joins the writer before the sink is destroyed. Producer threads should still be stopped at the normal engine lifecycle boundary before process teardown.

## Runtime coverage

| Area | Instrumentation |
|---|---|
| Request lifecycle | Server request start/end with task ID as `request_id` |
| Graph execution | Graph-compute duration; graph-level, not kernel timing |
| Synchronization | Context synchronization and MoE worker lifecycle events |
| MoE cache | Existing hit/miss/eviction plus cell reuse |
| KV cache | Allocation, teardown, clear, update duration, ubatch insertion, cell reuse/release, plus timed `seq_rm`, `seq_cp`, `seq_keep`, `seq_add` and `seq_div` operations with sequence/range metadata |
| CPU workers | Job enqueue, worker idle wait, compute span, job completion, CPU/prefill output publication, and GPU-side host wait for CPU output |
| Prefetch | Lookahead enqueue begin/end, estimated bytes, and backend-event wait |
| Memory allocation/free | Generic backend CPU/GPU buffer alloc/free and OOM, plus CUDA device alloc, pool return/release and pinned-host allocation/free |
| OOM/fallback | Device allocation OOM, buffer allocation OOM, pinned-host-to-CPU fallback |
| Transfers | H2D/D2H, 2D H2D/D2H and D2D/peer call-site spans |
| CUDA operation dispatch | Per-`ggml_op` dispatch span and operation identity |
| Individual CUDA kernels | Actual kernel names/timings remain sourced from Nsight/CUPTI activities imported by LEONES |

CUDA call-site spans around asynchronous operations describe host-side submission/dispatch, not GPU execution time. For actual individual kernel durations and all runtime memcpy activity, use the LEONES Nsight SQLite importer. Do not compare Nsight timestamps with steady-clock events without explicit clock-domain calibration.

## Correlation and validation limits

Request start/end carry the server task ID. Internal worker and CUDA events currently use `request_id=0`; no cross-thread request identity is fabricated. The CPU worker wait is the worker's wait for a job, while GPU-side waits and kernel activities are taken from the Nsight trace.

CI verifies event and JSONL sink tests, applies `tandem.patch` and `runtime.patch` to the pinned llama.cpp revision, runs the deterministic probe applier, checks whitespace, and builds the CPU `llama-server` target. It does not compile the `.cu` translation unit with nvcc, execute inference, or prove every event category appears in one capture. A CUDA-enabled build and reproducible runtime capture remain required before the complete observability phase can be accepted.

The sink callback must remain non-blocking and thread-safe. Instrumentation must preserve engine semantics and must not add blocking file I/O to worker threads.

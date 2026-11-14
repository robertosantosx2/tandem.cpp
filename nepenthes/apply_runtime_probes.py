#!/usr/bin/env python3
"""Apply NEPENTHES runtime probes to the pinned llama.cpp tree.

Run after tandem.patch and runtime.patch have been applied:
    python3 nepenthes/apply_runtime_probes.py /path/to/llama.cpp
Every edit is anchored to an exact source fragment; source drift fails closed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_once(root: Path, rel: str, old: str, new: str, label: str) -> None:
    path = root / rel
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor in {rel}, found {count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def replace_all(root: Path, rel: str, old: str, new: str, expected: int, label: str) -> None:
    path = root / rel
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} anchors in {rel}, found {count}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: apply_runtime_probes.py LLAMA_CPP_ROOT", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    here = Path(__file__).resolve().parent
    if not (root / "src/llama-context.cpp").is_file():
        raise RuntimeError(f"not a llama.cpp source tree: {root}")
    if not (root / "src/nepenthes_trace.hpp").is_file():
        raise RuntimeError("runtime.patch must be applied first (src/nepenthes_trace.hpp missing)")

    # The sink is copied into src/ so both core and CUDA translation units can include it.
    shutil.copyfile(here / "nepenthes_jsonl_sink.hpp", root / "src/nepenthes_jsonl_sink.hpp")

    # Start the process sink lazily from the first graph compute. It remains opt-in.
    replace_once(
        root, "src/llama-context.cpp",
        '#include "nepenthes_trace.hpp"\n',
        '#include "nepenthes_trace.hpp"\n#include "nepenthes_jsonl_sink.hpp"\n',
        "context sink include",
    )
    replace_once(
        root, "src/llama-context.cpp",
        '                   bool   batched) {\n    nepenthes::ScopedEvent nepenthes_compute',
        '                   bool   batched) {\n    nepenthes::ensure_sink_from_env();\n    nepenthes::ScopedEvent nepenthes_compute',
        "graph-compute sink activation",
    )

    moe = "src/llama-moe-cache.cpp"
    # Measure lookahead enqueue and the later backend-event wait separately.
    replace_once(
        root, moe,
        '    void la_join(ggml_backend_t backend) {\n'
        '        if (la_pending) {\n'
        '            ggml_backend_event_wait(backend, la_event);\n'
        '            la_pending = false;',
        '    void la_join(ggml_backend_t backend) {\n'
        '        if (la_pending) {\n'
        '            const std::uint64_t wait_start = nepenthes::now_ns();\n'
        '            ggml_backend_event_wait(backend, la_event);\n'
        '            nepenthes::emit(nepenthes::EventType::prefetch, "moe.lookahead_prefetch.wait", la_pend_list.size(), 0, nepenthes::phase_instant, 0, nepenthes::now_ns() - wait_start);\n'
        '            la_pending = false;',
        "lookahead wait",
    )
    replace_once(
        root, moe,
        '            const int64_t t0 = ggml_time_us();\n'
        '            const layer & lt = layers[tl];',
        '            const int64_t t0 = ggml_time_us();\n'
        '            const std::uint64_t pf_start = nepenthes::now_ns();\n'
        '            nepenthes::emit(nepenthes::EventType::prefetch, "moe.lookahead_prefetch.enqueue", static_cast<std::uint64_t>(tl), 0, nepenthes::phase_begin);\n'
        '            const layer & lt = layers[tl];',
        "lookahead enqueue begin",
    )
    replace_once(
        root, moe,
        '            t_la_us += ggml_time_us() - t0;\n'
        '            la_job.store(-1, std::memory_order_relaxed);',
        '            t_la_us += ggml_time_us() - t0;\n'
        '            std::uint64_t pf_bytes_per_expert = 0;\n'
        '            for (const ggml_tensor * w : lt.experts) pf_bytes_per_expert += w->nb[2];\n'
        '            const std::uint64_t pf_bytes = pf_bytes_per_expert * fills.size();\n'
        '            const std::uint64_t pf_end = nepenthes::now_ns();\n'
        '            nepenthes::emit(nepenthes::EventType::prefetch, "moe.lookahead_prefetch.enqueue", static_cast<std::uint64_t>(tl), pf_bytes, nepenthes::phase_end, 0, pf_end >= pf_start ? pf_end - pf_start : 0);\n'
        '            la_job.store(-1, std::memory_order_relaxed);',
        "lookahead enqueue end",
    )

    # Worker idle wait, compute span, completion publication and CPU output publication.
    replace_once(
        root, moe,
        '            int32_t il;\n'
        '            idle_wait([&] { return (il = job_layer.load(std::memory_order_acquire)) != -1; });\n'
        '            if (il == -2) {',
        '            int32_t il;\n'
        '            const std::uint64_t wait_start = nepenthes::now_ns();\n'
        '            idle_wait([&] { return (il = job_layer.load(std::memory_order_acquire)) != -1; });\n'
        '            const std::uint64_t wait_end = nepenthes::now_ns();\n'
        '            nepenthes::emit(nepenthes::EventType::stall, "moe.cpu_worker.wait", il >= 0 ? static_cast<std::uint64_t>(il) : 0, 0, nepenthes::phase_instant, 0, wait_end - wait_start);\n'
        '            if (il == -2) {',
        "CPU worker idle wait",
    )
    replace_once(
        root, moe,
        '            compute_cpu(layers[il]);\n'
        '            if (layers[il].pf_job) {',
        '            nepenthes::emit(nepenthes::EventType::sync, "moe.cpu_worker.start", static_cast<std::uint64_t>(il));\n'
        '            {\n'
        '                nepenthes::ScopedEvent compute_event(nepenthes::EventType::cpu_work, "moe.cpu_worker.compute", static_cast<std::uint64_t>(il));\n'
        '                compute_cpu(layers[il]);\n'
        '            }\n'
        '            if (layers[il].pf_job) {',
        "CPU worker compute",
    )
    replace_once(
        root, moe,
        '            job_layer.store(-1, std::memory_order_relaxed);\n'
        '            job_done.store(true, std::memory_order_release);\n'
        '            if (pr) {',
        '            job_layer.store(-1, std::memory_order_relaxed);\n'
        '            job_done.store(true, std::memory_order_release);\n'
        '            nepenthes::emit(nepenthes::EventType::sync, "moe.cpu_worker.done", static_cast<std::uint64_t>(il));\n'
        '            if (pr) {',
        "CPU worker completion",
    )
    replace_once(
        root, moe,
        '    void post_cpu_out(layer & l) {\n'
        '        memcpy(l.cpu_out->data, l.out.data(), l.out.size()*sizeof(float));\n'
        '        std::atomic_thread_fence(std::memory_order_release);\n'
        '        volatile uint32_t * f = (volatile uint32_t *) l.flag->data;\n'
        '        f[0] = f[0] + 1;\n'
        '    }',
        '    void post_cpu_out(layer & l) {\n'
        '        memcpy(l.cpu_out->data, l.out.data(), l.out.size()*sizeof(float));\n'
        '        std::atomic_thread_fence(std::memory_order_release);\n'
        '        volatile uint32_t * f = (volatile uint32_t *) l.flag->data;\n'
        '        f[0] = f[0] + 1;\n'
        '        nepenthes::emit(nepenthes::EventType::sync, "moe.cpu_worker.publish", static_cast<std::uint64_t>(&l - layers.data()), static_cast<std::uint64_t>(l.out.size() * sizeof(float)), nepenthes::phase_instant);\n'
        '    }',
        "CPU output publication",
    )
    replace_once(
        root, moe,
        '    void post_pf_out(layer & l) {\n'
        '        memcpy(pf_out->data, l.out.data(), l.out.size()*sizeof(float));\n'
        '        std::atomic_thread_fence(std::memory_order_release);\n'
        '        volatile uint32_t * f = (volatile uint32_t *) l.pf_flag->data;\n'
        '        f[0] = f[0] + 1;\n'
        '    }',
        '    void post_pf_out(layer & l) {\n'
        '        memcpy(pf_out->data, l.out.data(), l.out.size()*sizeof(float));\n'
        '        std::atomic_thread_fence(std::memory_order_release);\n'
        '        volatile uint32_t * f = (volatile uint32_t *) l.pf_flag->data;\n'
        '        f[0] = f[0] + 1;\n'
        '        nepenthes::emit(nepenthes::EventType::sync, "moe.cpu_worker.prefill_publish", static_cast<std::uint64_t>(&l - layers.data()), static_cast<std::uint64_t>(l.out.size() * sizeof(float)), nepenthes::phase_instant);\n'
        '    }',
        "CPU prefill publication",
    )

    # KV cache allocation, teardown, clear, writes and cell reuse/eviction.
    kvh = "src/llama-kv-cache.h"
    replace_once(
        root, kvh, "~llama_kv_cache() = default;", "~llama_kv_cache();",
        "KV destructor declaration",
    )
    kv = "src/llama-kv-cache.cpp"
    replace_once(
        root, kv, '#include "llama-kv-cache.h"\n',
        '#include "llama-kv-cache.h"\n#include "nepenthes_trace.hpp"\n',
        "KV trace include",
    )
    replace_once(
        root, kv,
        '        const size_t memory_size_v = size_v_bytes();',
        '        const size_t memory_size_v = size_v_bytes();\n'
        '        nepenthes::emit(nepenthes::EventType::kv_cache, "kv_cache.allocate", memory_size_k + memory_size_v, kv_size, nepenthes::phase_instant);',
        "KV allocation",
    )
    replace_once(
        root, kv, 'void llama_kv_cache::clear(bool data) {',
        'llama_kv_cache::~llama_kv_cache() {\n'
        '    nepenthes::emit(nepenthes::EventType::kv_cache, "kv_cache.free", size_k_bytes() + size_v_bytes(), get_size(), nepenthes::phase_instant);\n'
        '}\n\n'
        'void llama_kv_cache::clear(bool data) {',
        "KV teardown",
    )
    replace_once(
        root, kv, 'void llama_kv_cache::clear(bool data) {',
        'void llama_kv_cache::clear(bool data) {\n'
        '    nepenthes::emit(nepenthes::EventType::kv_cache, "kv_cache.clear", data ? 1 : 0, get_size(), nepenthes::phase_instant);',
        "KV clear",
    )
    replace_once(
        root, kv,
        'void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {\n'
        '    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]\n'
        '    if (other) {\n'
        '        return;\n'
        '    }',
        'void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {\n'
        '    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]\n'
        '    if (other) {\n'
        '        return;\n'
        '    }\n'
        '    nepenthes::emit(nepenthes::EventType::kv_cache, "kv_cache.apply_ubatch", ubatch.n_tokens, sinfo.size() * sinfo.n_stream(), nepenthes::phase_instant);',
        "KV apply ubatch",
    )
    replace_once(
        root, kv,
        '                const llama_seq_id seq_id = cells.seq_get(idx);\n'
        '                const llama_pos    pos    = cells.pos_get(idx);',
        '                const llama_seq_id seq_id = cells.seq_get(idx);\n'
        '                const llama_pos    pos    = cells.pos_get(idx);\n'
        '                nepenthes::emit(nepenthes::EventType::eviction, "kv_cache.cell_reuse", idx, static_cast<std::uint64_t>(seq_id), nepenthes::flag_host);',
        "KV cell reuse",
    )
    replace_once(
        root, kv,
        '            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {\n'
        '                if (new_head == cells.size()) {',
        '            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {\n'
        '                nepenthes::emit(nepenthes::EventType::kv_cache, "kv_cache.cell_release", i, static_cast<std::uint64_t>(seq_id), nepenthes::phase_instant);\n'
        '                if (new_head == cells.size()) {',
        "KV cell release",
    )
    replace_once(
        root, kv,
        '                cells.rm(i);\n'
        '                if (new_head == cells.size()) {',
        '                cells.rm(i);\n'
        '                nepenthes::emit(nepenthes::EventType::kv_cache, "kv_cache.cell_release_all", i, s, nepenthes::phase_instant);\n'
        '                if (new_head == cells.size()) {',
        "KV release all",
    )
    replace_once(
        root, kv,
        '    if (other) {\n'
        '        return true;\n'
        '    }\n\n'
        '    bool updated = false;',
        '    if (other) {\n'
        '        return true;\n'
        '    }\n\n'
        '    nepenthes::ScopedEvent kv_update(nepenthes::EventType::kv_cache, "kv_cache.update", get_size());\n'
        '    bool updated = false;',
        "KV update duration",
    )

    # Generic backend buffer lifecycle covers ordinary CPU allocations as well as
    # backend buffers. CUDA-specific probes below add device-level detail.
    backend = "ggml/src/ggml-backend.cpp"
    replace_once(
        root, backend, '#include "ggml-backend-impl.h"\n',
        '#include "ggml-backend-impl.h"\n#include "../../src/nepenthes_trace.hpp"\n',
        "backend trace include",
    )
    replace_once(
        root, backend,
        '    return buft->iface.alloc_buffer(buft, size);',
        '    nepenthes::ScopedEvent alloc_event(nepenthes::EventType::allocation, "backend.buffer.alloc", size, ggml_backend_buft_is_host(buft) ? 1 : 0);\n'
        '    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, size);\n'
        '    if (buffer == nullptr) {\n'
        '        nepenthes::emit(nepenthes::EventType::oom, "backend.buffer.alloc_oom", size, ggml_backend_buft_is_host(buft) ? 1 : 0, nepenthes::flag_failed);\n'
        '    }\n'
        '    return buffer;',
        "generic backend buffer allocation",
    )
    replace_once(
        root, backend,
        '    if (buffer->iface.free_buffer != NULL) {\n',
        '    nepenthes::emit(nepenthes::EventType::free_event, "backend.buffer.free", buffer->size, ggml_backend_buft_is_host(buffer->buft) ? 1 : 0);\n'
        '    if (buffer->iface.free_buffer != NULL) {\n',
        "generic backend buffer free",
    )

    # CUDA allocations/frees, host-pinned fallback, transfer boundaries and op dispatch.
    cu = "ggml/src/ggml-cuda/ggml-cuda.cu"
    replace_once(
        root, cu, '#include "ggml-cuda.h"\n',
        '#include "ggml-cuda.h"\n#include "../../../src/nepenthes_trace.hpp"\n',
        "CUDA trace include",
    )
    replace_once(
        root, cu,
        'static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {\n'
        '    ggml_cuda_set_device(device);',
        'static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {\n'
        '    ggml_cuda_set_device(device);\n'
        '    nepenthes::ScopedEvent nepenthes_alloc(nepenthes::EventType::allocation, "cuda.device_alloc", size, device);',
        "CUDA device allocation",
    )
    replace_once(
        root, cu, '    return err;\n}\n\n#if defined(GGML_USE_HIP)',
        '    if (err == cudaErrorMemoryAllocation) {\n'
        '        nepenthes::emit(nepenthes::EventType::oom, "cuda.device_alloc_oom", size, device, nepenthes::flag_device | nepenthes::flag_failed);\n'
        '    }\n'
        '    return err;\n}\n\n#if defined(GGML_USE_HIP)',
        "CUDA device allocation OOM",
    )
    replace_once(
        root, cu, '                CUDA_CHECK(cudaFree(b.ptr));',
        '                nepenthes::emit(nepenthes::EventType::free_event, "cuda.pool.release", b.size, device, nepenthes::flag_device);\n'
        '                CUDA_CHECK(cudaFree(b.ptr));',
        "CUDA pool release",
    )
    replace_all(
        root, cu, '    void free(void * ptr, size_t size) override {\n',
        '    void free(void * ptr, size_t size) override {\n'
        '        nepenthes::emit(nepenthes::EventType::free_event, "cuda.pool.return", size, device, nepenthes::flag_device);\n',
        2, "CUDA pool return",
    )
    replace_once(
        root, cu,
        'static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n'
        '    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;\n'
        '    delete ctx;',
        'static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n'
        '    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;\n'
        '    nepenthes::emit(nepenthes::EventType::free_event, "cuda.buffer.free", buffer->size, ctx->device, nepenthes::flag_device);\n'
        '    delete ctx;',
        "CUDA buffer free",
    )
    replace_once(
        root, cu,
        'static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {\n'
        '    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;\n',
        'static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {\n'
        '    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;\n'
        '    nepenthes::ScopedEvent nepenthes_transfer(nepenthes::EventType::transfer, "cuda.h2d", size, ctx->device);\n',
        "CUDA H2D",
    )
    replace_once(
        root, cu,
        '    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));',
        '    nepenthes::ScopedEvent nepenthes_transfer(nepenthes::EventType::transfer, "cuda.d2h", size, ctx->device);\n'
        '    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));',
        "CUDA D2H",
    )
    replace_once(
        root, cu,
        'static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,\n'
        '        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {\n'
        '    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;\n',
        'static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,\n'
        '        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {\n'
        '    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;\n'
        '    nepenthes::ScopedEvent nepenthes_transfer(nepenthes::EventType::transfer, "cuda.h2d_2d", size * n_copies, ctx->device);\n',
        "CUDA H2D 2D",
    )
    replace_once(
        root, cu,
        '    CUDA_CHECK(cudaMemcpy2DAsync(\n'
        '        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));',
        '    nepenthes::ScopedEvent nepenthes_transfer(nepenthes::EventType::transfer, "cuda.d2h_2d", size * n_copies, ctx->device);\n'
        '    CUDA_CHECK(cudaMemcpy2DAsync(\n'
        '        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));',
        "CUDA D2H 2D",
    )
    replace_once(
        root, cu,
        'static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {\n'
        '    if (ggml_backend_buffer_is_cuda(src->buffer)) {',
        'static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {\n'
        '    if (ggml_backend_buffer_is_cuda(src->buffer)) {\n'
        '        nepenthes::ScopedEvent nepenthes_transfer(nepenthes::EventType::transfer, "cuda.d2d_or_peer", ggml_nbytes(src), 0);',
        "CUDA D2D/peer transfer",
    )
    replace_once(
        root, cu,
        '        (void)cudaGetLastError();\n'
        '        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));',
        '        (void)cudaGetLastError();\n'
        '        nepenthes::emit(nepenthes::EventType::oom, "cuda.buffer_alloc_oom", size, buft_ctx->device, nepenthes::flag_device | nepenthes::flag_failed);\n'
        '        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));',
        "CUDA buffer allocation OOM",
    )
    replace_once(
        root, cu,
        'static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n'
        '    CUDA_CHECK(cudaFreeHost(buffer->context));',
        'static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n'
        '    nepenthes::emit(nepenthes::EventType::free_event, "cuda.pinned_host.free", buffer->size, 0, nepenthes::flag_host);\n'
        '    CUDA_CHECK(cudaFreeHost(buffer->context));',
        "pinned host free",
    )
    replace_once(
        root, cu,
        '    return ptr;\n}\n\nstatic ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer',
        '    nepenthes::emit(nepenthes::EventType::allocation, "cuda.pinned_host.alloc", size, 0, nepenthes::flag_host);\n'
        '    return ptr;\n}\n\nstatic ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer',
        "pinned host allocation",
    )
    replace_once(
        root, cu,
        '    if (ptr == nullptr) {\n'
        '        // fallback to cpu buffer\n'
        '        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);',
        '    if (ptr == nullptr) {\n'
        '        // fallback to cpu buffer\n'
        '        nepenthes::emit(nepenthes::EventType::fallback, "cuda.pinned_host.to_cpu", size, 0, nepenthes::flag_host | nepenthes::flag_failed);\n'
        '        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);',
        "pinned host CPU fallback",
    )
    replace_once(
        root, cu,
        'static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {\n',
        'static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {\n'
        '    nepenthes::ScopedEvent nepenthes_kernel(nepenthes::EventType::kernel, ggml_op_name(dst->op), static_cast<std::uint64_t>(dst->op), ggml_nbytes(dst));\n',
        "CUDA op dispatch",
    )

    # Request lifecycle IDs at the server boundary; lower-level async worker events
    # deliberately keep request_id=0 until a safe cross-thread propagation path exists.
    server = "tools/server/server-context.cpp"
    replace_once(
        root, server,
        '#include "../../src/nepenthes_trace.hpp"\n',
        '#include "../../src/nepenthes_trace.hpp"\n#include "../../src/nepenthes_jsonl_sink.hpp"\n',
        "server sink include",
    )
    replace_once(
        root, server,
        '    void send_final_response(server_slot & slot) {\n',
        '    void send_final_response(server_slot & slot) {\n'
        '        nepenthes::emit(nepenthes::EventType::request_end, "server.request_end", slot.task->id, slot.id, nepenthes::phase_instant, slot.task->id);\n',
        "request end",
    )
    replace_once(
        root, server,
        '                    const int id_task = task.id;\n',
        '                    const int id_task = task.id;\n'
        '                    nepenthes::ensure_sink_from_env();\n'
        '                    nepenthes::emit(nepenthes::EventType::request_start, "server.request_start", id_task, static_cast<std::uint64_t>(task.type), nepenthes::phase_instant, id_task);\n',
        "request start",
    )

    print("NEPENTHES runtime probes applied successfully.")
    print("Note: CUDA kernel timings and memcpy activities are also captured by the LEONES Nsight SQLite importer.")
    print("Note: request_id is populated at the server boundary; internal worker/CUDA events remain uncorrelated (0).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NEPENTHES runtime probe application failed: {exc}", file=sys.stderr)
        raise

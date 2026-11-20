# NEPENTHES producer-side NVTX batch markers — design gate

Status: design / research proposal, not implementation.

## Goal
Expose the opaque correlation token already assigned to each validated rendered decode slice (Tandem.cpp PR #5) to Nsight Systems through an NVTX marker range, so LEONES can identify batch-scoped activity windows conservatively.

## Constraints
- Marker scope is the exact decode slice at the producer boundary where its membership snapshot is validated. Do not widen the range to an HTTP request or scheduler lifetime.
- Encode the opaque batch token in a stable, parseable marker payload. The downstream importer must scope identity by process and host thread and require calibrated temporal containment and unique window; token collisions across scopes must not join.
- The batch token means many-to-many membership, not per-request GPU ownership. Never assign an activity to one member solely from the token.
- Do not emit a marker when membership validation fails or when the execution slice has no stable token.
- Marker begin/end must pair under normal return, early return and exception/unwind paths. Use RAII.
- NVTX must be optional and disabled without its headers/library; no new mandatory dependency or performance-sensitive blocking path.
- Marker emission must occur on the host thread that executes the relevant decode slice. Do not copy token to unrelated poller/background threads.
- Any CUDA/Nsight activity attribution remains windowed and subject to LEONES guardrails; markers alone are not causal proof.

## Implementation sequence
1. Inspect pinned llama.cpp integration and existing Tandem patch to find the exact render/decode-slice boundary and token scope.
2. Determine an optional NVTX compile/link strategy compatible with supported CI builds; verify disabled mode.
3. Implement RAII marker scope using the batch token and explicitly documented payload encoding.
4. Test marker parsing/payload, begin/end lifecycle, invalid-member suppression, nested scopes, disabled mode, and compile configuration.
5. Run patch-application and CPU/CUDA compile smoke on the pinned base, plus importer fixtures in LEONES.
6. Only after software evidence passes, update canonical LEONES INDEX. Physical paired capture (C07) and overhead (C08) remain separate acceptance gates.

## Existing evidence
- Tandem.cpp PR #5: host batch token and member snapshot propagation, merged.
- LEONES PR #218: NVTX batch importer and many-to-many membership view, merged.
- LEONES PR #219: conservative process/thread/window guard, merged.
These establish software groundwork only; they do not validate producer NVTX or physical GPU attribution.

## Blockers / validation limits
This document does not claim an implementation, successful compile, real NVTX emission, C07, C08, or physical engine acceptance. The issue tracker is disabled in this repository, so this design is tracked as a pull-request document pending review.

# EVICT — Cost-Aware Adaptive Verification for Speculative Decoding

Implementation of **EVICT** (Expert-aware Verification via Identifying Cost-effective
Tree prefixes), from *"Making Every Verified Token Count: Adaptive Verification for
MoE Speculative Decoding"* (Pan et al., arXiv:2605.00342), adapted to this vLLM fork.

## 1. Key reframing: chain, not tree

The paper targets a **token tree** (EAGLE-3 style) and selects an ancestor-closed
subtree. This fork's EAGLE/EAGLE-3/MTP drafting is a **linear chain** of
`num_speculative_tokens` (`K`) tokens per request
(`llm_base_proposer.py` builds `draft_token_ids_list` sequentially → `[B, K]`).

In a chain, the only ancestor-closed subtrees are **prefixes**, so the whole
algorithm collapses to a 1-D problem — *choose how many of the K already-drafted
tokens to actually verify*:

| Paper (tree)                                  | This fork (chain)                                    |
|-----------------------------------------------|------------------------------------------------------|
| `Score(v) = ∏ q` along root→v                 | `score[k] = ∏_{j≤k} q[j]` — cumulative product       |
| sort nodes by Score (ancestor-closed)         | chain is *already* sorted (`q ≤ 1` ⇒ monotone ↓)     |
| `Ê[A(T_k)] = Σ top-k Score`                   | `Ê[A(m)] = Σ_{j<m} score[j]` — prefix sum            |
| `U(k) = Ê[A(T_k)] / C(k)`                      | `U(m) = Ê[A(m)] / C(m)`                              |
| `k* = argmax_k U`; verify subtree `T_{k*}`    | `m* = argmax_m U`; verify first `m*` chain tokens    |

`q[j]` is the draft model's probability of its chosen token at chain position `j`.

Because verifying fewer tokens shrinks the target forward's token count, for an MoE
target the **union of activated experts** drops — the paper's core win — at the cost
of a (bounded, lossless) reduction in accepted length.

## 2. Losslessness

Truncating the verified prefix never changes the output distribution: every emitted
token is still an exact target sample (the standard speculative-decoding guarantee).
Reducing `m` only reduces how many speculative tokens we *attempt*; under greedy
decoding the generated sequence is **identical** with EVICT on/off (only the number
of decode steps may grow). Under sampling, each token remains a valid target sample.

## 3. Cost model `C(m)`

`C(m)` = latency of a speculative step that verifies `m` draft tokens. The draft is
already fully generated before truncation (the chain is built in one `propose()`),
so EVICT saves only **target-forward + verify** cost, not draft cost. Therefore

```
C(m) ≈ target_forward(m) + verify(m)
```

profiled offline. `build_cost_table.py` measures `C(m)` directly: for each
`m = 1..max_spec_tokens` it runs one **`max_num_seqs=1` (B=1)** session with
`num_speculative_tokens = m` fixed and averages the exported `target_forward_ms`
+ `verify_ms` over that run's timed steps (via `vllm/v1/spec_decode/timing.py`).
B=1 makes each step verify exactly `m` positions, so the average is the
per-request cost the selector assumes — not a batch-size average. (A future
refinement is to bin a single mixed run by `num_verified_positions` instead of
one run per `m`; that requires exporting `num_verified_positions` as a metric,
which the timing branch does not yet do.) See
`vllm/v1/spec_decode/evict/build_cost_table.py`.

Only the *shape* of `C(m)` over `m` matters for `argmax`. As `m` grows, `Ê[A(m)]`
grows sub-linearly (diminishing, since `score` shrinks) while `C(m)` grows roughly
linearly, so `U(m)` peaks at `m*`.

If no profiled table is supplied, an affine fallback `C(m) = c0 + c1·m` is used
(only the ratio `c1/c0` affects `m*`); this is an approximation for experimentation,
logged at startup. The file-based table is the paper-faithful path.

## 4. Components

| File | Role |
|------|------|
| `vllm/v1/spec_decode/evict/cost_table.py` | `CostTable`: load/validate profiled `C(m)`; affine fallback; `as_tensor`. Dependency-light (torch only). |
| `vllm/v1/spec_decode/evict/selector.py` | Pure-torch `select_kstar`: gather q → cumprod → cumsum → `/C(m)` → argmax. GPU-vectorized, no Python loop over `m`. Dependency-light. |
| `vllm/v1/spec_decode/evict/build_cost_table.py` | Offline binner: timing export → `C(m)` JSON. |
| `vllm/config/speculative.py` | `evict_*` config fields + fail-closed validation. |
| `vllm/v1/worker/gpu_model_runner.py` | Init selector; truncation hook in the draft-propose closure; metrics tagging. |
| `vllm/v1/spec_decode/metrics.py` | `m*` distribution + saved-positions counters. |

## 5. Integration point (V1 path)

`GPUModelRunner.propose_draft_token_ids` produces `self._draft_token_ids` `[B, K]`
and `self._draft_probs` `[B, K, V]` (via `take_last_draft_probs()`), then
`_copy_draft_token_ids_to_cpu` ships them to the scheduler, which schedules them for
**next** step's verification.

EVICT hooks **between** those two calls: compute `m*` from the just-drafted probs and
slice `self._draft_token_ids = [:, :m*]` (and `_draft_probs` to match). The scheduler
then schedules only `m*` tokens, so the next target forward verifies `m*` positions.

Decided at draft time (probs known then), applied to *next* step's verify — exactly
"is this just-drafted token worth verifying?".

## 6. MVP vs full

- **MVP (this change, default OFF):** batch-uniform `m*` via tensor slice
  (`prev_num_spec_tokens` stride stays uniform; exact per-request for `B=1`, the
  paper's regime). Eager-friendly, GPU-vectorized selector, offline/affine cost
  table, lossless. Requires `draft_sample_method="probabilistic"` (draft probs only
  exposed then; greedy-request confidence is a follow-up). Goal: correctness +
  measurable saved verify positions.
- **Full (follow-up):** per-request `m*` (variable-length draft output), and
  multi-length captured verify graphs + selector fused into the captured region
  (the paper notes eager EVICT *loses* on wall-clock; the win needs CUDA graphs).

## 7. Scope / limitations (MVP)

- `B>1`: `m*` is batch-uniform (reduction over per-request `m*`, default `max`).
- `draft_sample_method="greedy"` is rejected fail-closed at config time. Even
  under `"probabilistic"`, individual greedy requests (`temperature=0`) expose
  no draft probabilities, so EVICT no-ops for them at runtime (warned once).
- Cost table is static; online refit is future work.
- Composes downstream of Dynamic-SD: it picks `K`, EVICT trims verify to `m* ≤ K`.

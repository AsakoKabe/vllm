# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare speculative decoding with and without EVICT adaptive verification.

Runs the same greedy workload under up to three configurations and reports the
metrics implemented on this branch, with end-to-end **speedup** as the headline:

    * ``vanilla_ar``     - no speculation (throughput baseline for speedup); optional
    * ``spec_baseline``  - speculative decoding, ``evict_enabled=False``
    * ``spec_evict``     - speculative decoding, ``evict_enabled=True``

For each spec config it reads ``LLM.get_metrics()`` and derives, per step:
accepted length E[A], acceptance rate, mean verified positions, the stage
timings (target_forward T_T, draft_total T_D, verify T_reject, per-position
draft), the layer-averaged distinct-expert count U_r, and - when the
single-position bin is populated - T_T(0), T_T(K), eta(K) and an analytical
speedup from the paper's decomposition.

IMPORTANT: EVICT is a no-op at ``--temperature 0`` (greedy) — its truncation
guard skips any step containing a greedy request, so ``spec_evict`` == baseline.
To exercise EVICT, run with ``--temperature 0.7`` (and ``--max-num-seqs 1`` for
the paper's B=1 regime, which also populates the T_T(0)/eta bins and makes the
batch-uniform m* exact). At temperature 0 the script also checks the outputs are
byte-identical (EVICT is lossless under greedy); at temperature > 0 that check
is skipped (losslessness is distributional, not token-exact).

Requires a GPU. Both spec configs use ``draft_sample_method='probabilistic'``
(required by EVICT) so the only difference is ``evict_enabled``.

Examples:
    # Exercise EVICT (temperature > 0, B=1) — the truncation actually fires.
    # --cache reuses vanilla_ar + spec_baseline across runs so iterating on
    # EVICT only re-runs spec_evict.
    python examples/features/speculative_decoding/evict_vs_baseline.py \
        --method eagle3 --num-spec-tokens 4 --temperature 0.7 --max-num-seqs 1 \
        --cache evict_cache.json

    # EAGLE-3 on Llama-3.1-8B (dense; quick smoke test)
    python examples/features/speculative_decoding/evict_vs_baseline.py \
        --method eagle3 --num-spec-tokens 4 --num-prompts 16

    # The paper's MoE setup: Qwen3-30B-A3B + EAGLE-3, with a profiled cost table
    python examples/features/speculative_decoding/evict_vs_baseline.py \
        --model Qwen/Qwen3-30B-A3B --method eagle3 --eagle-dir <head> \
        --num-spec-tokens 8 --evict-cost-table cost_table.json \
        --enable-return-routed-experts
"""

import gc
import hashlib
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Metric, Vector

PROMPTS = [
    "Explain how speculative decoding accelerates LLM inference.",
    "Write a short story about a robot learning to paint.",
    "Summarize the theory of relativity in three sentences.",
    "List five practical tips for writing maintainable Python code.",
    "Describe the water cycle for a curious ten-year-old.",
    "What are the trade-offs between throughput and latency in serving?",
    "Give a recipe for a simple vegetable soup.",
    "Compare supervised and reinforcement learning.",
]

# Exported metric names (reader strips the Prometheus ``_total`` suffix).
STAGE_METRICS = {
    "target_forward": "vllm:spec_decode_target_forward_microseconds",
    "verify": "vllm:spec_decode_verify_microseconds",
    "sample": "vllm:spec_decode_sample_microseconds",
    "draft_total": "vllm:spec_decode_draft_total_microseconds",
}
NUM_STEPS_METRIC = "vllm:spec_decode_num_timed_steps"
PER_POS_METRIC = "vllm:spec_decode_draft_forward_microseconds_per_pos"
TF_US_BY_POS_METRIC = "vllm:spec_decode_target_forward_microseconds_by_positions"
TF_COUNT_BY_POS_METRIC = "vllm:spec_decode_target_forward_count_by_positions"
DISTINCT_EXPERTS_METRIC = "vllm:spec_decode_distinct_experts_milli"
NUM_DRAFTS_METRIC = "vllm:spec_decode_num_drafts"
NUM_DRAFT_TOKENS_METRIC = "vllm:spec_decode_num_draft_tokens"
NUM_ACCEPTED_TOKENS_METRIC = "vllm:spec_decode_num_accepted_tokens"
EVICT_STEPS_METRIC = "vllm:spec_decode_evict_steps"
EVICT_KSTAR_SUM_METRIC = "vllm:spec_decode_evict_kstar_sum"
EVICT_SAVED_METRIC = "vllm:spec_decode_evict_saved_positions"
EVICT_HIST_METRIC = "vllm:spec_decode_evict_kstar_hist"

_SCALAR_NAMES = frozenset(
    set(STAGE_METRICS.values())
    | {
        NUM_STEPS_METRIC,
        DISTINCT_EXPERTS_METRIC,
        NUM_DRAFTS_METRIC,
        NUM_DRAFT_TOKENS_METRIC,
        NUM_ACCEPTED_TOKENS_METRIC,
        EVICT_STEPS_METRIC,
        EVICT_KSTAR_SUM_METRIC,
        EVICT_SAVED_METRIC,
    }
)
_VECTOR_NAMES = frozenset(
    {
        PER_POS_METRIC,
        TF_US_BY_POS_METRIC,
        TF_COUNT_BY_POS_METRIC,
        EVICT_HIST_METRIC,
    }
)


def parse_args():
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method",
        type=str,
        default="eagle3",
        choices=["eagle", "eagle3", "mtp"],
        help="Chain draft method (EVICT requires a chain method).",
    )
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. EVICT is inactive at 0.0 (greedy); use >0 "
        "(e.g. 0.7) so EVICT actually truncates. The lossless text check only "
        "applies at 0.0.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Cap concurrent sequences. Set 1 for the paper's B=1 regime so "
        "T_T(0)/eta bins populate and batch-uniform m* is exact.",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-return-routed-experts",
        action="store_true",
        help="Capture routed experts so U_r is populated (MoE targets only).",
    )
    # EVICT knobs (affine fallback cost model unless a table is given).
    parser.add_argument("--evict-cost-table", type=str, default=None)
    parser.add_argument("--evict-cost-intercept", type=float, default=1.0)
    parser.add_argument("--evict-cost-per-token", type=float, default=0.5)
    parser.add_argument("--evict-min-k", type=int, default=1)
    parser.add_argument(
        "--evict-batch-reduce",
        type=str,
        default="max",
        choices=["max", "min", "median"],
    )
    parser.add_argument(
        "--evict-allowed-kstar",
        type=str,
        default=None,
        help="Comma-separated allowed m* lengths (e.g. '1,4,8'). Quantizes the "
        "selector to this set and captures FULL CUDA graphs for the matching "
        "verify lengths, so truncated verifies keep full-graph replay and T_T "
        "actually scales with m*. Unset = no quantization (truncated verifies "
        "fall to PIECEWISE padded to K+1 and save no wall-clock).",
    )
    parser.add_argument(
        "--skip-ar",
        action="store_true",
        help="Skip the vanilla-AR baseline (report only EVICT-vs-baseline ratio).",
    )
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to write the results JSON. "
        "Default: evict_results_<timestamp>.json in the cwd.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a results JSON.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to a prediction cache. When set, vanilla_ar and spec_baseline "
        "(which do not depend on EVICT) are reused from it if the run signature "
        "matches, so only spec_evict re-runs. The cache is (re)written each run.",
    )
    return parser.parse_args()


def build_speculative_config(args, evict_enabled: bool) -> dict:
    if args.method in ("eagle", "eagle3"):
        eagle_dir = args.eagle_dir or (
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
            if args.method == "eagle"
            else "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        )
        cfg = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    else:  # mtp
        cfg = {"method": "mtp", "num_speculative_tokens": args.num_spec_tokens}

    # EVICT needs draft probabilities; keep both spec configs identical here so
    # the only difference between baseline and EVICT is evict_enabled.
    cfg["draft_sample_method"] = "probabilistic"
    if evict_enabled:
        cfg.update(
            evict_enabled=True,
            evict_cost_table_path=args.evict_cost_table,
            evict_cost_intercept=args.evict_cost_intercept,
            evict_cost_per_token=args.evict_cost_per_token,
            evict_min_k=args.evict_min_k,
            evict_batch_reduce=args.evict_batch_reduce,
        )
        if args.evict_allowed_kstar:
            cfg["evict_allowed_kstar"] = [
                int(m) for m in args.evict_allowed_kstar.split(",")
            ]
    return cfg


def snapshot_metrics(metrics: list[Metric]) -> dict:
    """Collapse counters/vectors across engines into name -> value/list."""
    snap: dict[str, object] = {}
    for metric in metrics:
        if isinstance(metric, Counter) and metric.name in _SCALAR_NAMES:
            snap[metric.name] = float(snap.get(metric.name, 0.0)) + metric.value
        elif isinstance(metric, Vector) and metric.name in _VECTOR_NAMES:
            acc = list(snap.get(metric.name, []))  # type: ignore[arg-type]
            if len(metric.values) > len(acc):
                acc.extend([0] * (len(metric.values) - len(acc)))
            for i, v in enumerate(metric.values):
                acc[i] += v
            snap[metric.name] = acc
    return snap


def _diff_scalar(after: dict, before: dict, name: str) -> float:
    return float(after.get(name, 0.0)) - float(before.get(name, 0.0))


def _diff_vector(after: dict, before: dict, name: str) -> list[int]:
    a = list(after.get(name, []))
    b = list(before.get(name, []))
    n = max(len(a), len(b))
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    return [a[i] - b[i] for i in range(n)]


def derive(after: dict, before: dict, num_spec_tokens: int) -> dict:
    """Compute the per-run diagnostics from a before/after metric snapshot."""
    steps = _diff_scalar(after, before, NUM_STEPS_METRIC)
    out: dict = {"num_timed_steps": int(steps)}
    if steps <= 0:
        return out

    for stage, name in STAGE_METRICS.items():
        out[f"{stage}_ms"] = _diff_scalar(after, before, name) / steps / 1000.0

    per_pos = _diff_vector(after, before, PER_POS_METRIC)
    out["per_pos_draft_ms"] = [us / steps / 1000.0 for us in per_pos]

    drafts = _diff_scalar(after, before, NUM_DRAFTS_METRIC)
    draft_tokens = _diff_scalar(after, before, NUM_DRAFT_TOKENS_METRIC)
    accepted = _diff_scalar(after, before, NUM_ACCEPTED_TOKENS_METRIC)
    out["accepted_length"] = 1.0 + accepted / drafts if drafts else float("nan")
    out["acceptance_rate"] = accepted / draft_tokens if draft_tokens else float("nan")

    distinct_milli = _diff_scalar(after, before, DISTINCT_EXPERTS_METRIC)
    out["avg_distinct_experts"] = distinct_milli / 1000.0 / steps

    # Target forward binned by verified positions: T_T(j) = us[j+1]/count[j+1].
    us_by = _diff_vector(after, before, TF_US_BY_POS_METRIC)
    cnt_by = _diff_vector(after, before, TF_COUNT_BY_POS_METRIC)

    def t_t(positions: int) -> float | None:
        if positions < len(cnt_by) and cnt_by[positions] > 0:
            return us_by[positions] / cnt_by[positions] / 1000.0
        return None

    total_cnt = sum(cnt_by[1:]) if len(cnt_by) > 1 else 0
    out["mean_verified_positions"] = (
        sum(k * cnt_by[k] for k in range(1, len(cnt_by))) / total_cnt
        if total_cnt
        else float("nan")
    )
    # Dominant verification length (most frequent bin) and its T_T.
    dom_k = max(range(1, len(cnt_by)), key=lambda k: cnt_by[k]) if total_cnt else 0
    out["dominant_positions"] = dom_k
    out["t_t0_ms"] = t_t(1)  # single-position forward == T_T(0), paper index
    out["t_t_dom_ms"] = t_t(dom_k) if dom_k else None
    out["t_t_full_ms"] = t_t(num_spec_tokens + 1)  # T_T(K), full-chain forward

    # Analytical speedup vs AR from the decomposition (needs a T_T(0) sample):
    #   SpeedUp = E[A] * T_T(0) / (K * T_D(1) + T_T(dom) + T_reject)
    # using the realized draft-forward T_D(1) = mean per-position draft.
    t_t0 = out["t_t0_ms"]
    t_t_ref = out["t_t_dom_ms"] or out["t_t_full_ms"]
    t_d1 = out["per_pos_draft_ms"][0] if out["per_pos_draft_ms"] else 0.0
    denom = num_spec_tokens * t_d1 + (t_t_ref or 0.0) + out["verify_ms"]
    if t_t0 and t_t_ref and denom > 0:
        out["analytical_speedup"] = out["accepted_length"] * t_t0 / denom
    else:
        out["analytical_speedup"] = None

    # EVICT truncation effect (populated only for the EVICT config).
    evict_steps = _diff_scalar(after, before, EVICT_STEPS_METRIC)
    if evict_steps > 0:
        out["evict_steps"] = int(evict_steps)
        out["evict_mean_kstar"] = (
            _diff_scalar(after, before, EVICT_KSTAR_SUM_METRIC) / evict_steps
        )
        out["evict_saved_positions"] = int(
            _diff_scalar(after, before, EVICT_SAVED_METRIC)
        )
        hist = _diff_vector(after, before, EVICT_HIST_METRIC)
        out["evict_kstar_hist"] = hist
    return out


def run_config(args, name: str, spec_config: dict | None, prompts, sp) -> dict:
    print(f"\n===== running config: {name} =====")
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_log_stats=False,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if spec_config is not None:
        llm_kwargs["speculative_config"] = spec_config
        llm_kwargs["spec_decode_timing"] = True
        # EVICT truncation runs only under SYNCHRONOUS scheduling. vLLM enables
        # async spec-decode scheduling by default for EAGLE, which sets
        # _evict_active=False and makes EVICT a silent no-op. Force sync for both
        # spec configs so EVICT actually fires and baseline is a fair comparison.
        llm_kwargs["async_scheduling"] = False
        if args.enable_return_routed_experts:
            llm_kwargs["enable_return_routed_experts"] = True
    llm = LLM(**llm_kwargs)

    # Warm up (JIT/CUDA-graph capture) on a couple of prompts, then snapshot so
    # the timed measurement excludes warmup.
    llm.generate(prompts[: min(2, len(prompts))], sampling_params=sp)
    before = snapshot_metrics(llm.get_metrics())

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sp)
    wall_s = time.perf_counter() - t0
    after = snapshot_metrics(llm.get_metrics())

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    texts = [o.outputs[0].text for o in outputs]
    result = {
        "name": name,
        "wall_s": wall_s,
        "out_tokens": out_tokens,
        "throughput": out_tokens / wall_s if wall_s > 0 else float("nan"),
        "texts": texts,
    }
    if spec_config is not None:
        result.update(derive(after, before, args.num_spec_tokens))
    if args.print_output:
        print(f"  sample: {texts[0][:120]!r}")

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def _fmt(value, spec: str = "8.3f") -> str:
    if value is None:
        return f"{'n/a':>8}"
    if isinstance(value, float) and value != value:  # NaN
        return f"{'n/a':>8}"
    return f"{value:{spec}}"


def print_report(
    results: list[dict], num_spec_tokens: int, check_lossless: bool = True
) -> None:
    by_name = {r["name"]: r for r in results}
    ar = by_name.get("vanilla_ar")
    base = by_name.get("spec_baseline")
    evict = by_name.get("spec_evict")

    print("\n" + "=" * 68)
    print("THROUGHPUT / SPEEDUP (headline)")
    print("=" * 68)
    print(f"  {'config':<16}{'tokens/s':>12}{'tokens':>10}{'wall_s':>10}")
    for r in results:
        print(
            f"  {r['name']:<16}{_fmt(r['throughput'], '12.2f')}"
            f"{r['out_tokens']:>10}{_fmt(r['wall_s'], '10.2f')}"
        )
    if ar:
        for r in (base, evict):
            if r:
                sp = r["throughput"] / ar["throughput"]
                print(f"  speedup vs AR [{r['name']}]: {sp:6.3f}x")
    if base and evict:
        rel = evict["throughput"] / base["throughput"]
        print(f"  EVICT / baseline throughput ratio: {rel:6.3f}x")

    rows = [
        ("accepted_length E[A]", "accepted_length", "8.3f"),
        ("acceptance_rate", "acceptance_rate", "8.3f"),
        ("mean_verified_positions", "mean_verified_positions", "8.3f"),
        ("dominant_positions", "dominant_positions", "8d"),
        ("target_forward T_T (ms)", "target_forward_ms", "8.3f"),
        ("draft_total T_D (ms)", "draft_total_ms", "8.3f"),
        ("verify T_reject (ms)", "verify_ms", "8.3f"),
        ("sample (ms)", "sample_ms", "8.3f"),
        ("U_r (distinct experts)", "avg_distinct_experts", "8.3f"),
        ("T_T(0) (ms)", "t_t0_ms", "8.3f"),
        ("T_T(K) (ms)", "t_t_full_ms", "8.3f"),
        ("analytical speedup", "analytical_speedup", "8.3f"),
        ("num_timed_steps", "num_timed_steps", "8d"),
        ("EVICT mean m*", "evict_mean_kstar", "8.3f"),
        ("EVICT saved positions", "evict_saved_positions", "8d"),
        ("EVICT steps", "evict_steps", "8d"),
    ]
    spec_results = [r for r in (base, evict) if r]
    if spec_results:
        print("\n" + "=" * 68)
        print("DECOMPOSITION METRICS (per timed spec step)")
        print("=" * 68)
        header = f"  {'metric':<26}" + "".join(f"{r['name']:>14}" for r in spec_results)
        print(header)
        for label, key, spec in rows:
            cells = "".join(_fmt(r.get(key), f"14{spec[1:]}") for r in spec_results)
            print(f"  {label:<26}{cells}")

    if evict and evict.get("evict_kstar_hist"):
        hist = evict["evict_kstar_hist"]
        dist = ", ".join(f"m*={m}:{c}" for m, c in enumerate(hist) if c > 0)
        print(f"\n  EVICT m* distribution [spec_evict]: {dist}")

    if base and evict:
        print("\n" + "-" * 68)
        if not check_lossless:
            print(
                "lossless check: skipped (temperature > 0; EVICT losslessness is "
                "distributional, not token-exact)"
            )
        else:
            identical = base["texts"] == evict["texts"]
            status = "PASS" if identical else "FAIL"
            print(f"lossless check (baseline text == EVICT text): {status}")
            if not identical:
                mism = sum(1 for a, b in zip(base["texts"], evict["texts"]) if a != b)
                print(f"  WARNING: {mism} prompt(s) differ - EVICT must be lossless!")


def _jsonable(result: dict) -> dict:
    """Drop the bulky raw texts, keep a digest plus every numeric metric."""
    out = {k: v for k, v in result.items() if k != "texts"}
    joined = "\x00".join(result.get("texts", []))
    out["text_sha256"] = hashlib.sha256(joined.encode()).hexdigest()
    return out


def save_results(results: list[dict], args, path: str) -> None:
    """Write config + per-config metrics + speedups to JSON for later diffing."""
    by_name = {r["name"]: r for r in results}
    ar = by_name.get("vanilla_ar")
    base = by_name.get("spec_baseline")
    evict = by_name.get("spec_evict")

    speedups: dict = {}
    if ar and base:
        speedups["spec_baseline_vs_ar"] = base["throughput"] / ar["throughput"]
    if ar and evict:
        speedups["spec_evict_vs_ar"] = evict["throughput"] / ar["throughput"]
    if base and evict:
        speedups["evict_vs_baseline"] = evict["throughput"] / base["throughput"]

    # Token-exact lossless check only makes sense at greedy (temperature 0).
    if base and evict and args.temperature == 0.0:
        lossless = base["texts"] == evict["texts"]
    else:
        lossless = None
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "configs": {r["name"]: _jsonable(r) for r in results},
        "speedups": speedups,
        "lossless": lossless,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nsaved results to {path}")


# Configs whose predictions do not depend on EVICT knobs -> cacheable so that
# only spec_evict re-runs while iterating on EVICT.
_CACHEABLE = ("vanilla_ar", "spec_baseline")


def _cache_signature(args) -> dict:
    """Run parameters that affect the cacheable configs' outputs."""
    keys = (
        "model",
        "method",
        "eagle_dir",
        "num_spec_tokens",
        "num_prompts",
        "output_len",
        "temperature",
        "seed",
        "max_num_seqs",
        "tp",
        "max_model_len",
    )
    return {k: getattr(args, k) for k in keys}


def load_cache(path: str | None, signature: dict) -> dict:
    """Return cached cacheable-config results whose signature matches, else {}."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("signature") != signature:
        print(f"cache signature changed at {path}; re-running all configs.")
        return {}
    return {
        name: r for name, r in data.get("results", {}).items() if name in _CACHEABLE
    }


def save_cache(path: str | None, signature: dict, results: list[dict]) -> None:
    if not path:
        return
    keep = {r["name"]: r for r in results if r["name"] in _CACHEABLE}
    with open(path, "w") as f:
        json.dump({"signature": signature, "results": keep}, f)
    print(f"prediction cache written to {path}")


def main(args) -> None:
    prompts = (PROMPTS * (args.num_prompts // len(PROMPTS) + 1))[: args.num_prompts]
    sp = SamplingParams(
        temperature=args.temperature, seed=args.seed, max_tokens=args.output_len
    )

    signature = _cache_signature(args)
    cached = load_cache(args.cache, signature)

    def get(name: str, spec_config: dict | None) -> dict:
        if name in cached:
            print(f"\n===== {name}: reused from cache ({args.cache}) =====")
            return cached[name]
        return run_config(args, name, spec_config, prompts, sp)

    results = []
    if not args.skip_ar:
        results.append(get("vanilla_ar", None))
    results.append(get("spec_baseline", build_speculative_config(args, False)))
    # spec_evict always re-runs: it is the config being iterated on.
    results.append(
        run_config(
            args,
            "spec_evict",
            build_speculative_config(args, evict_enabled=True),
            prompts,
            sp,
        )
    )
    print_report(
        results, args.num_spec_tokens, check_lossless=(args.temperature == 0.0)
    )

    if not args.no_save:
        path = args.save_json or f"evict_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, args, path)
    save_cache(args.cache, signature, results)


if __name__ == "__main__":
    main(parse_args())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run EVICT cost-aware adaptive verification for speculative decoding.

EVICT (arXiv:2605.00342) trims the drafted token chain to its cost-effective
prefix m* before target verification, where m* maximizes the utility
``U(m) = E[A(m)] / C(m)`` (estimated accepted length over profiled step cost).
Verifying fewer tokens shrinks the target forward and, for an MoE target, the
union of activated experts — lossless, since only speculation depth is reduced.

This script has two modes:

* ``--explain`` (no GPU): walk through the m* selection on synthetic draft
  confidences using the real selector and cost table, to show the algorithm.

* default (GPU + model): run EAGLE-3 speculative decoding with EVICT enabled.
  EVICT requires ``draft_sample_method="probabilistic"`` (draft probabilities
  must be exposed) and the synchronous draft path
  (``disable_async_output_proc=True``). Per-step m* and saved verify positions
  are logged by the worker (see the "EVICT: mean m*=..." log lines).

Examples:
    # No GPU — illustrate m* selection
    python examples/features/speculative_decoding/evict_adaptive_verification.py \
        --explain

    # The paper's MoE setup: Qwen3-30B-A3B + EAGLE-3
    python examples/features/speculative_decoding/evict_adaptive_verification.py \
        --model Qwen/Qwen3-30B-A3B --eagle-dir <qwen3-eagle3-head> \
        --num-spec-tokens 8 --cost-table qwen3_30b_a3b_evict_cost.json

    # Llama-3.1-8B + EAGLE-3 with the affine cost fallback (no profiled table)
    python examples/features/speculative_decoding/evict_adaptive_verification.py \
        --num-spec-tokens 6 --cost-per-token 1.0
"""

from vllm.utils.argparse_utils import FlexibleArgumentParser

PROMPTS = [
    "Explain how speculative decoding accelerates LLM inference.",
    "Write a short story about a robot learning to paint.",
    "Summarize the theory of relativity in three sentences.",
    "List five practical tips for writing maintainable Python code.",
]


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Illustrate m* selection on synthetic data (no GPU required).",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--eagle-dir", type=str, default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    )
    parser.add_argument("--num-spec-tokens", type=int, default=6)
    parser.add_argument(
        "--cost-table",
        type=str,
        default=None,
        help="Profiled C(m) JSON (from build_cost_table). Affine fallback if unset.",
    )
    parser.add_argument("--cost-intercept", type=float, default=1.0)
    parser.add_argument("--cost-per-token", type=float, default=1.0)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def explain(args) -> None:
    """Show m* = argmax_m E[A(m)] / C(m) on a few synthetic draft chains."""
    import torch

    from vllm.v1.spec_decode.evict.cost_table import CostTable
    from vllm.v1.spec_decode.evict.selector import (
        expected_accepted_length,
        select_kstar,
    )

    k = args.num_spec_tokens
    if args.cost_table is not None:
        table = CostTable.from_file(args.cost_table)
    else:
        table = CostTable.affine(args.cost_intercept, args.cost_per_token)
    cost_per_m = table.as_tensor(k)

    # Synthetic per-position draft confidences: an "easy" chain (high, flat),
    # a "decaying" chain, and a "hard" chain (low after the first token).
    scenarios = {
        "easy (flat 0.95)": torch.full((1, k), 0.95),
        "decaying": torch.linspace(0.95, 0.3, k).unsqueeze(0),
        "hard (drops fast)": torch.tensor([[0.8] + [0.15] * (k - 1)]),
    }

    print("=" * 64)
    print(f"EVICT m* selection (K={k}, cost C(m)={cost_per_m.tolist()})")
    print("=" * 64)
    for name, conf in scenarios.items():
        ehat = expected_accepted_length(conf)[0]
        utility = ehat / cost_per_m
        kstar = int(select_kstar(conf, cost_per_m, min_k=args.min_k)[0])
        print(f"\n{name}: q={[round(x, 3) for x in conf[0].tolist()]}")
        print(f"  E[A(m)] = {[round(x, 3) for x in ehat.tolist()]}")
        print(f"  U(m)    = {[round(x, 3) for x in utility.tolist()]}")
        print(f"  -> m* = {kstar} (verify {kstar}/{k} drafted tokens)")
    print(
        "\nLower-utility tail tokens are dropped: they add verify cost "
        "(MoE experts) but little expected acceptance."
    )


def run_llm(args) -> None:
    from vllm import LLM, SamplingParams

    speculative_config = {
        "method": "eagle3",
        "model": args.eagle_dir,
        "num_speculative_tokens": args.num_spec_tokens,
        # EVICT needs draft probabilities exposed.
        "draft_sample_method": "probabilistic",
        "evict_enabled": True,
        "evict_min_k": args.min_k,
    }
    if args.cost_table is not None:
        speculative_config["evict_cost_table_path"] = args.cost_table
    else:
        speculative_config["evict_cost_intercept"] = args.cost_intercept
        speculative_config["evict_cost_per_token"] = args.cost_per_token

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        speculative_config=speculative_config,
        # EVICT is wired through the synchronous draft path.
        disable_async_output_proc=True,
        # Per-stage timing also lets you compare verify latency vs EVICT off.
        spec_decode_timing=True,
        disable_log_stats=False,
    )

    prompts = (PROMPTS * (args.num_prompts // len(PROMPTS) + 1))[: args.num_prompts]
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.output_len, seed=1234
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    if args.print_output:
        for prompt, output in zip(prompts, outputs):
            print("-" * 50)
            print(f"prompt: {prompt}")
            print(f"generated: {output.outputs[0].text}")

    print("-" * 50)
    print(
        "EVICT ran with EAGLE-3. Per-step m* and saved verify positions are in "
        "the worker logs ('EVICT: mean m*=...'). Note: EVICT is inactive for "
        "greedy (temperature=0) requests, which do not expose draft probs."
    )


def main(args) -> None:
    if args.explain:
        explain(args)
    else:
        run_llm(args)


if __name__ == "__main__":
    main(parse_args())

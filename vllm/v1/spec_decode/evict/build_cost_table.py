# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline builder for the EVICT cost table ``C(m)``.

``C(m)`` is the verify-side latency of a speculative step that verifies ``m``
draft tokens — i.e. ``target_forward(m) + verify(m)`` — which is exactly what
EVICT trades against estimated accepted length. The draft is generated in full
before truncation, so draft cost is excluded.

This profiles each ``m`` independently by running short generations with
``num_speculative_tokens = m`` and reading the per-stage timing exported through
``LLM.get_metrics()`` (requires this build's ``--spec-decode-timing``). The mean
per-step ``target_forward + verify`` at each ``m`` becomes ``C(m)``, written as a
JSON table consumable by ``CostTable.from_file`` / ``evict_cost_table_path``.

Requires a GPU and the target (and, for EAGLE/MTP, the draft) model.

Example (the paper's MoE setup)::

    python -m vllm.v1.spec_decode.evict.build_cost_table \
        --model Qwen/Qwen3-30B-A3B --method eagle3 \
        --eagle-dir <qwen3-eagle3-head> --max-spec-tokens 8 \
        --out qwen3_30b_a3b_evict_cost.json
"""

import json

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Metric

logger = init_logger(__name__)

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

TARGET_FORWARD_US = "vllm:spec_decode_target_forward_microseconds"
VERIFY_US = "vllm:spec_decode_verify_microseconds"
NUM_STEPS = "vllm:spec_decode_num_timed_steps"


def _mean_verify_cost_ms(metrics: list[Metric]) -> float | None:
    """Mean per-step ``target_forward + verify`` latency (ms)."""
    total_us = 0
    num_steps = 0
    for metric in metrics:
        if metric.name in (TARGET_FORWARD_US, VERIFY_US):
            assert isinstance(metric, Counter)
            total_us += metric.value
        elif metric.name == NUM_STEPS:
            assert isinstance(metric, Counter)
            num_steps += metric.value
    if num_steps == 0:
        return None
    return total_us / num_steps / 1000.0


def _speculative_config(args, num_spec_tokens: int) -> dict:
    if args.method in ("eagle", "eagle3"):
        eagle_dir = args.eagle_dir or (
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
            if args.method == "eagle"
            else "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        )
        return {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": num_spec_tokens,
            # EVICT needs draft probabilities at runtime; profile under the same
            # draft sampling so the cost reflects the deployed configuration.
            "draft_sample_method": "probabilistic",
        }
    if args.method == "mtp":
        return {
            "method": "mtp",
            "num_speculative_tokens": num_spec_tokens,
            "draft_sample_method": "probabilistic",
        }
    raise ValueError(f"Unsupported method for EVICT cost profiling: {args.method}")


def profile_cost(args, num_spec_tokens: int) -> float | None:
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        speculative_config=_speculative_config(args, num_spec_tokens),
        disable_log_stats=False,
        spec_decode_timing=True,
        # Force B=1 per step so the averaged target_forward+verify is the
        # per-request cost of verifying `num_spec_tokens` positions — the cost
        # the selector assumes — rather than an average over the varying batch
        # sizes that continuous batching would otherwise produce.
        max_num_seqs=1,
    )
    prompts = (PROMPTS * (args.num_prompts // len(PROMPTS) + 1))[: args.num_prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.output_len)
    llm.generate(prompts, sampling_params=sampling_params)
    cost = _mean_verify_cost_ms(llm.get_metrics())
    del llm
    return cost


def parse_args():
    parser = FlexibleArgumentParser(description="Build an EVICT C(m) cost table.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--method", type=str, default="eagle3", choices=["eagle", "eagle3", "mtp"]
    )
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument(
        "--max-spec-tokens", type=int, default=8, help="Profile m=1..this."
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--out", type=str, required=True, help="Output JSON path.")
    return parser.parse_args()


def main(args) -> None:
    costs: dict[str, float] = {}
    for m in range(1, args.max_spec_tokens + 1):
        logger.info("Profiling EVICT cost for m=%d ...", m)
        cost = profile_cost(args, m)
        if cost is None:
            raise RuntimeError(
                f"No timed steps recorded at m={m}; is --spec-decode-timing on?"
            )
        costs[str(m)] = cost
        logger.info("C(%d) = %.4f ms", m, cost)

    payload = {"unit": "ms", "model": args.model, "method": args.method, "costs": costs}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote EVICT cost table (%d entries) to %s", len(costs), args.out)


if __name__ == "__main__":
    main(parse_args())

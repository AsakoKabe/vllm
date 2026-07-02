# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the atomic per-stage speculative-decoding timing metrics.

Runs speculative decoding with ``--spec-decode-timing`` enabled, then reads the
per-stage GPU timings exported through ``LLM.get_metrics()`` and checks that
they are present and non-zero. Requires a GPU.

Examples:
    # EAGLE-3 on Llama-3.1-8B (dense; quick smoke test)
    python examples/features/speculative_decoding/spec_decode_timing.py \
        --method eagle3 --num-spec-tokens 4 --num-prompts 8

    # The paper's MoE setup: Qwen3-30B-A3B + EAGLE-3
    python examples/features/speculative_decoding/spec_decode_timing.py \
        --model Qwen/Qwen3-30B-A3B --method eagle3 \
        --eagle-dir <qwen3-eagle3-head> --num-spec-tokens 8

    # ngram (no draft model; per-position draft timing is empty)
    python examples/features/speculative_decoding/spec_decode_timing.py \
        --method ngram --num-spec-tokens 4
"""

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

# stage label -> exported counter name (cumulative microseconds)
STAGE_METRICS = {
    "target_forward": "vllm:spec_decode_target_forward_microseconds",
    "verify": "vllm:spec_decode_verify_microseconds",
    "sample": "vllm:spec_decode_sample_microseconds",
    "draft_total": "vllm:spec_decode_draft_total_microseconds",
}
NUM_STEPS_METRIC = "vllm:spec_decode_num_timed_steps"
PER_POS_METRIC = "vllm:spec_decode_draft_forward_microseconds_per_pos"

# Methods whose draft is produced by a model forward (so per-position draft
# timing is expected). ngram/suffix use a lookup and only populate draft_total.
MODEL_DRAFT_METHODS = {"eagle", "eagle3", "mtp", "draft_model"}


def parse_args():
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method",
        type=str,
        default="eagle3",
        choices=["ngram", "eagle", "eagle3", "mtp", "draft_model"],
    )
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def build_speculative_config(args) -> dict:
    if args.method in ("eagle", "eagle3"):
        eagle_dir = args.eagle_dir
        if eagle_dir is None:
            eagle_dir = (
                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
                if args.method == "eagle"
                else "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
            )
        return {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    if args.method == "draft_model":
        assert args.draft_model, "--draft-model is required for method=draft_model"
        return {
            "method": args.method,
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "max_model_len": args.max_model_len,
        }
    if args.method == "mtp":
        return {"method": "mtp", "num_speculative_tokens": args.num_spec_tokens}
    return {"method": "ngram", "num_speculative_tokens": args.num_spec_tokens}


def collect_timing(metrics: list[Metric]) -> dict:
    """Aggregate the timing metrics across engine instances."""
    totals_us = {stage: 0 for stage in STAGE_METRICS}
    num_steps = 0
    per_pos_us: list[int] = []
    name_to_stage = {name: stage for stage, name in STAGE_METRICS.items()}
    for metric in metrics:
        if metric.name in name_to_stage:
            assert isinstance(metric, Counter)
            totals_us[name_to_stage[metric.name]] += metric.value
        elif metric.name == NUM_STEPS_METRIC:
            assert isinstance(metric, Counter)
            num_steps += metric.value
        elif metric.name == PER_POS_METRIC:
            assert isinstance(metric, Vector)
            if len(metric.values) > len(per_pos_us):
                per_pos_us.extend([0] * (len(metric.values) - len(per_pos_us)))
            for pos, value in enumerate(metric.values):
                per_pos_us[pos] += value
    return {"totals_us": totals_us, "num_steps": num_steps, "per_pos_us": per_pos_us}


def print_timing(timing: dict) -> None:
    steps = timing["num_steps"]
    print("-" * 50)
    print(f"timed speculative steps: {steps}")
    if steps == 0:
        return
    print("mean per-step stage latency:")
    for stage in STAGE_METRICS:
        mean_ms = timing["totals_us"][stage] / steps / 1000.0
        print(f"  {stage:<15}: {mean_ms:8.3f} ms")
    if timing["per_pos_us"]:
        per_pos_ms = ", ".join(
            f"{us / steps / 1000.0:.3f}" for us in timing["per_pos_us"]
        )
        print(f"  per-position draft (ms): [{per_pos_ms}]")
    print("-" * 50)


def verify_timing(timing: dict, method: str) -> None:
    assert timing["num_steps"] > 0, (
        "No timed speculative steps recorded. Ensure spec_decode_timing is on "
        "and speculative decoding is active."
    )
    assert timing["totals_us"]["target_forward"] > 0, "target_forward timing is zero"
    if method in MODEL_DRAFT_METHODS:
        assert timing["totals_us"]["draft_total"] > 0, "draft_total timing is zero"
        assert any(us > 0 for us in timing["per_pos_us"]), (
            "per-position draft timing is empty for a model-based draft method"
        )
    print("OK: spec-decode timing metrics are present and non-zero.")


def main(args) -> None:
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        speculative_config=build_speculative_config(args),
        # Required so Prometheus metrics are collected in offline mode, and so
        # the new per-stage timings are recorded and exported.
        disable_log_stats=False,
        spec_decode_timing=True,
    )

    prompts = (PROMPTS * (args.num_prompts // len(PROMPTS) + 1))[: args.num_prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.output_len)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    if args.print_output:
        for prompt, output in zip(prompts, outputs):
            print("-" * 50)
            print(f"prompt: {prompt}")
            print(f"generated: {output.outputs[0].text}")

    timing = collect_timing(llm.get_metrics())
    print_timing(timing)
    verify_timing(timing, args.method)


if __name__ == "__main__":
    main(parse_args())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke / safety checks for EVICT adaptive verification.

Losslessness note: speculative decoding is lossless by construction — every
emitted token is an exact target sample regardless of how many draft tokens are
verified. EVICT only reduces the verified prefix length m*, so it never changes
the output distribution; it only changes how many decode steps are taken. A
byte-for-byte ON/OFF equality test is *not* meaningful under sampling (the RNG
is consumed differently when fewer tokens are accepted per step), and under
greedy decoding this fork does not expose draft probabilities, so EVICT is
inactive. These tests therefore check (a) enabling EVICT under greedy is a safe
no-op, and (b) EVICT runs end-to-end under probabilistic draft sampling.

Requires a GPU and model access; not part of the pure-unit suite.
"""

import pytest

from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE3_DIR = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
NUM_SPEC_TOKENS = 6

PROMPTS = [
    "Explain how speculative decoding accelerates LLM inference.",
    "List five practical tips for writing maintainable Python code.",
    "Summarize the theory of relativity in three sentences.",
]


def _make_llm(evict_enabled: bool, draft_sample_method: str = "probabilistic") -> LLM:
    spec = {
        "method": "eagle3",
        "model": EAGLE3_DIR,
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "draft_sample_method": draft_sample_method,
        "evict_enabled": evict_enabled,
        # Aggressive affine cost so m* clearly truncates on easy prompts.
        "evict_cost_intercept": 1.0,
        "evict_cost_per_token": 1.0,
    }
    return LLM(
        model=MODEL,
        speculative_config=spec,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        # EVICT is wired through the synchronous draft path.
        disable_async_output_proc=True,
        enforce_eager=True,
    )


@pytest.mark.skip(reason="GPU + model access required; run manually in a GPU env.")
def test_evict_under_greedy_is_a_safe_noop():
    # All requests greedy (temperature=0): EVICT must be inactive (it never
    # truncates greedy chains, whose draft "confidence" is unrelated to the
    # deterministic acceptance), so the output is identical to EVICT off.
    # draft_sample_method stays "probabilistic" (greedy draft is fail-closed at
    # config time); the request-level temperature=0 triggers the no-op path.
    sampling = SamplingParams(temperature=0.0, max_tokens=96)

    baseline = _make_llm(evict_enabled=False)
    base_out = [o.outputs[0].token_ids for o in baseline.generate(PROMPTS, sampling)]
    del baseline

    evict = _make_llm(evict_enabled=True)
    evict_out = [o.outputs[0].token_ids for o in evict.generate(PROMPTS, sampling)]
    del evict

    for prompt, base_ids, evict_ids in zip(PROMPTS, base_out, evict_out):
        assert list(base_ids) == list(evict_ids), (
            f"Enabling EVICT changed the greedy output for prompt: {prompt!r}"
        )


@pytest.mark.skip(reason="GPU + model access required; run manually in a GPU env.")
def test_evict_runs_under_probabilistic_sampling():
    # EVICT is active only when draft probabilities are exposed (probabilistic
    # draft sampling + temperature > 0). Smoke-test that the pipeline runs and
    # produces non-empty output.
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=96, seed=1234)

    evict = _make_llm(evict_enabled=True, draft_sample_method="probabilistic")
    outputs = evict.generate(PROMPTS, sampling)
    del evict

    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert len(output.outputs[0].token_ids) > 0

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for spec-decode stage timing (Phase 1 scaffold).

These cover the host-side aggregation logic and the disabled-timer path; the
CUDA-event timing itself requires a GPU and is exercised in integration tests.
"""

from vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging,
    SpecDecodingStats,
    _ms_to_us,
)
from vllm.v1.spec_decode.timing import SpecDecodeTimer, SpecDecodeTimingStats


def _timing(
    target: float = 1.0,
    verify: float = 0.5,
    sample: float = 0.25,
    draft: float = 2.0,
    per_pos: list[float] | None = None,
) -> SpecDecodeTimingStats:
    return SpecDecodeTimingStats(
        target_forward_ms=target,
        verify_ms=verify,
        sample_ms=sample,
        draft_total_ms=draft,
        draft_forward_ms_per_pos=[0.7, 0.6] if per_pos is None else per_pos,
    )


def _collect(log_fn_args: list[str]):
    return lambda *args: log_fn_args.append(args[0] % args[1:])


def test_timing_stats_defaults():
    stats = SpecDecodeTimingStats()
    assert stats.target_forward_ms == 0.0
    assert stats.draft_forward_ms_per_pos == []


def test_observe_timing_sets_fields():
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    assert not stats.has_timing
    stats.observe_timing(_timing(per_pos=[0.7, 0.6, 0.5]))
    assert stats.has_timing
    assert stats.target_forward_ms == 1.0
    assert stats.verify_ms == 0.5
    assert stats.sample_ms == 0.25
    assert stats.draft_total_ms == 2.0
    assert stats.draft_forward_ms_per_pos == [0.7, 0.6, 0.5]


def test_observe_timing_copies_per_pos_list():
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    src = [1.0, 2.0]
    stats.observe_timing(_timing(per_pos=src))
    src.append(3.0)
    assert stats.draft_forward_ms_per_pos == [1.0, 2.0]


def test_ms_to_us_rounds_to_int():
    assert _ms_to_us(1.0) == 1000
    assert _ms_to_us(0.0015) == 2
    assert isinstance(_ms_to_us(0.333), int)


def test_logging_aggregates_timing():
    logging = SpecDecodingLogging()
    for _ in range(2):
        stats = SpecDecodingStats.new(num_spec_tokens=2)
        stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
        stats.observe_timing(_timing(target=2.0, per_pos=[1.0, 0.5]))
        logging.observe(stats)
    assert len(logging.target_forward_ms) == 2

    messages: list[str] = []
    logging.log(log_fn=_collect(messages))
    assert any("SpecDecoding timing" in m for m in messages)
    # log() resets accumulators.
    assert logging.target_forward_ms == []


def test_mean_per_pos_handles_variable_k():
    means = SpecDecodingLogging._mean_per_pos([[1.0, 2.0, 3.0], [3.0, 4.0]])
    assert means == [2.0, 3.0, 3.0]


def test_logging_without_timing_skips_timing_line():
    logging = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=2)
    logging.observe(stats)
    messages: list[str] = []
    logging.log(log_fn=_collect(messages))
    assert not any("SpecDecoding timing" in m for m in messages)


def test_timer_disabled_is_noop():
    timer = SpecDecodeTimer(enabled=False, num_spec_tokens=4)
    timer.begin_step()
    with timer.time_stage("target_forward"):
        pass
    with timer.time_stage("draft", pos=0):
        pass
    assert timer.drain() is None

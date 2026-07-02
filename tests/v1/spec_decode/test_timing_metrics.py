# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for spec-decode stage timing (Phase 1 scaffold).

These cover the host-side aggregation logic and the disabled-timer path; the
CUDA-event timing itself requires a GPU and is exercised in integration tests.
"""

import numpy as np

from vllm.v1.spec_decode.evict.stats import EvictStats
from vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging,
    SpecDecodingStats,
    _ms_to_us,
)
from vllm.v1.spec_decode.timing import (
    SpecDecodeTimer,
    SpecDecodeTimingStats,
    compute_avg_distinct_experts,
)


def _timing(
    target: float = 1.0,
    verify: float = 0.5,
    sample: float = 0.25,
    draft: float = 2.0,
    per_pos: list[float] | None = None,
    num_verified_positions: int = 0,
) -> SpecDecodeTimingStats:
    return SpecDecodeTimingStats(
        target_forward_ms=target,
        verify_ms=verify,
        sample_ms=sample,
        draft_total_ms=draft,
        draft_forward_ms_per_pos=[0.7, 0.6] if per_pos is None else per_pos,
        num_verified_positions=num_verified_positions,
    )


def _collect(log_fn_args: list[str]):
    return lambda *args: log_fn_args.append(args[0] % args[1:])


def test_timing_stats_defaults():
    stats = SpecDecodeTimingStats()
    assert stats.target_forward_ms == 0.0
    assert stats.draft_forward_ms_per_pos == []
    assert stats.num_verified_positions == 0


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


def test_observe_timing_copies_num_verified_positions():
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    assert stats.num_verified_positions == 0
    stats.observe_timing(_timing(num_verified_positions=4))
    assert stats.num_verified_positions == 4


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


def test_logging_target_by_positions_eta():
    # num_spec_tokens=2 -> a full-draft step verifies K+1=3 positions.
    logging = SpecDecodingLogging()
    for _ in range(2):
        stats = SpecDecodingStats.new(num_spec_tokens=2)
        stats.observe_draft(num_draft_tokens=0, num_accepted_tokens=0)
        # k=1 verified position is the K=0 (single-token) forward -> T_T(0).
        stats.observe_timing(_timing(target=1.0, num_verified_positions=1))
        logging.observe(stats)
    for _ in range(2):
        stats = SpecDecodingStats.new(num_spec_tokens=2)
        stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
        # k=3 verified positions -> T_T(2), the full-speculation forward.
        stats.observe_timing(_timing(target=4.0, num_verified_positions=3))
        logging.observe(stats)

    messages: list[str] = []
    logging.log(log_fn=_collect(messages))
    line = next(m for m in messages if "T_T by positions" in m)
    # Paper K index: bin[1] -> "0:", bin[3] -> "2:".
    assert "0:1.000(2)" in line
    assert "2:4.000(2)" in line
    assert "T_T(0): 1.000" in line
    assert "T_T(2): 4.000" in line
    # eta(2) = T_T(0) / T_T(2) = 1.0 / 4.0.
    assert "eta(2): 0.250" in line


def test_logging_target_by_positions_excludes_prefill():
    # A k=0 (prefill/non-spec) step must not appear in the by-positions line.
    logging = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
    stats.observe_timing(_timing(target=9.0, num_verified_positions=0))
    logging.observe(stats)

    messages: list[str] = []
    logging.log(log_fn=_collect(messages))
    # No verified-position pairs -> the by-positions line is suppressed.
    assert not any("T_T by positions" in m for m in messages)


def test_observe_evict_sets_fields():
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    assert not stats.has_evict
    stats.observe_evict(EvictStats(kstar=2, num_spec=4, num_reqs=3, saved_positions=6))
    assert stats.has_evict
    assert stats.evict_kstar == 2
    assert stats.evict_num_spec == 4
    assert stats.evict_num_reqs == 3
    assert stats.evict_saved_positions == 6


def test_logging_aggregates_evict():
    logging = SpecDecodingLogging()
    for _ in range(3):
        stats = SpecDecodingStats.new(num_spec_tokens=4)
        stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
        # m*=2 of K=4 with B=1 -> saved 2 positions/step; (4-2)/4 = 50% truncated.
        stats.observe_evict(
            EvictStats(kstar=2, num_spec=4, num_reqs=1, saved_positions=2)
        )
        logging.observe(stats)
    assert len(logging.evict_kstar) == 3

    messages: list[str] = []
    logging.log(log_fn=_collect(messages))
    line = next(m for m in messages if "EVICT" in m)
    assert "mean m*: 2.00" in line
    assert "saved verify positions: 6" in line  # 3 steps * 2
    assert "50.0% of chain truncated" in line
    # log() resets accumulators.
    assert logging.evict_kstar == []


def test_logging_without_evict_skips_evict_line():
    logging = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
    logging.observe(stats)
    messages: list[str] = []
    logging.log(log_fn=_collect(messages))
    assert not any("EVICT" in m for m in messages)


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
    timer.set_num_verified_positions(7)
    with timer.time_stage("target_forward"):
        pass
    with timer.time_stage("draft", pos=0):
        pass
    assert timer.drain() is None


class _FakeEvent:
    """A CPU stand-in for torch.cuda.Event so the timer logic is testable."""

    def __init__(self, enable_timing: bool = False) -> None:
        self.recorded = False

    def record(self) -> None:
        self.recorded = True

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return 1.0


def test_timer_double_buffer_one_step_lag(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    timer = SpecDecodeTimer(enabled=True, num_spec_tokens=2)

    # Step 1 records target_forward and tags positions; nothing to drain yet.
    timer.begin_step()
    timer.set_num_verified_positions(3)
    with timer.time_stage("target_forward"):
        pass
    assert timer.drain() is None

    # Step 2 drains step 1's timing (one-step lag).
    timer.begin_step()
    with timer.time_stage("verify"):
        pass
    drained = timer.drain()
    assert drained is not None
    assert drained.target_forward_ms == 1.0
    assert drained.verify_ms == 0.0  # not recorded in step 1
    assert drained.num_verified_positions == 3


def test_timer_per_position_draft(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    timer = SpecDecodeTimer(enabled=True, num_spec_tokens=3)

    timer.begin_step()
    with timer.time_stage("draft", pos=0):
        pass
    with timer.time_stage("draft", pos=1):
        pass

    timer.begin_step()
    drained = timer.drain()
    assert drained is not None
    # Only the two recorded positions are reported.
    assert drained.draft_forward_ms_per_pos == [1.0, 1.0]


class _FakeEventNotReady(_FakeEvent):
    """A CUDA event stand-in whose work has not completed on the device."""

    def query(self) -> bool:
        return False


def test_timer_drain_skips_until_events_complete(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "Event", _FakeEventNotReady)
    timer = SpecDecodeTimer(enabled=True, num_spec_tokens=2)

    timer.begin_step()
    with timer.time_stage("target_forward"):
        pass
    # The previous step recorded events, but they are not complete on the
    # device, so drain() must return None without reading elapsed_time.
    timer.begin_step()
    assert timer.drain() is None


def test_timing_stats_avg_distinct_experts_default():
    assert SpecDecodeTimingStats().avg_distinct_experts == 0.0


def test_compute_avg_distinct_experts():
    # 2 tokens, 3 layers, top_k=2.
    # layer 0: {1, 2, 3} -> 3 distinct; layer 1: {4, 5} -> 2 distinct;
    # layer 2: all-zero (dense/unused) -> excluded.
    routing = np.array(
        [
            [[1, 2], [4, 5], [0, 0]],
            [[3, 1], [4, 5], [0, 0]],
        ],
        dtype=np.int32,
    )
    assert compute_avg_distinct_experts(routing) == (3 + 2) / 2


def test_compute_avg_distinct_experts_expert_zero_counts():
    # Expert id 0 is valid when the layer also routes to other experts.
    routing = np.array([[[0, 1]], [[0, 2]]], dtype=np.int32)  # {0, 1, 2}
    assert compute_avg_distinct_experts(routing) == 3.0


def test_compute_avg_distinct_experts_empty_or_dense():
    assert compute_avg_distinct_experts(np.zeros((0, 4, 2), dtype=np.int32)) == 0.0
    # Fully dense target (every layer all-zero) -> no MoE layers -> 0.0.
    assert compute_avg_distinct_experts(np.zeros((3, 4, 2), dtype=np.int32)) == 0.0


def test_timer_avg_distinct_experts_roundtrip(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    timer = SpecDecodeTimer(enabled=True, num_spec_tokens=2)

    timer.begin_step()
    timer.set_avg_distinct_experts(42.5)
    with timer.time_stage("target_forward"):
        pass

    timer.begin_step()
    drained = timer.drain()
    assert drained is not None
    assert drained.avg_distinct_experts == 42.5

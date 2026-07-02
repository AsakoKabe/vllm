# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomic CUDA-event timing for speculative decoding stages.

The timer brackets each speculative-decode stage (target forward, verification,
sampling, and per-position draft forwards) with CUDA events recorded on the
current stream. Recording is asynchronous, so the hot path never stalls.
``elapsed_time`` is read one step later via a double-buffered slot ring, so the
read also never synchronizes the device. When disabled, ``time_stage`` is a
no-op context and ``drain`` returns ``None``, leaving the hot path untouched.
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

# Scalar stages timed once per speculative step. Per-position draft forwards are
# timed separately and keyed by position index.
SCALAR_STAGES = ("target_forward", "verify", "sample", "draft_total")


@dataclass
class SpecDecodeTimingStats:
    """GPU stage timings for one speculative step, in milliseconds.

    Populated in the worker by draining CUDA events and carried to the scheduler
    on ``ModelRunnerOutput``, where it is folded into ``SpecDecodingStats`` once
    per step.
    """

    target_forward_ms: float = 0.0
    verify_ms: float = 0.0
    sample_ms: float = 0.0
    draft_total_ms: float = 0.0
    draft_forward_ms_per_pos: list[float] = field(default_factory=list)
    # Target verification microbatch size (sum of K_i + 1 over requests). Used
    # later to bin T_T by the number of verified positions for the cost model.
    num_verified_positions: int = 0
    # Layer-averaged distinct-expert count (Ū_r) over the verification
    # microbatch; 0.0 for dense targets or when routing capture is unavailable.
    avg_distinct_experts: float = 0.0


class _EventPair:
    """A reusable (start, end) CUDA event pair with a recorded flag."""

    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.recorded = False

    def record_start(self) -> None:
        self.start.record()

    def record_end(self) -> None:
        self.end.record()
        self.recorded = True

    def ready(self) -> bool:
        return self.end.query()

    def elapsed_ms(self) -> float:
        return self.start.elapsed_time(self.end)

    def reset(self) -> None:
        self.recorded = False


class _Slot:
    """One step's worth of event pairs (scalar stages + per-position draft)."""

    def __init__(self, num_spec_tokens: int) -> None:
        self.scalar = {name: _EventPair() for name in SCALAR_STAGES}
        self.draft_pos = [_EventPair() for _ in range(num_spec_tokens)]
        self.dirty = False
        self.num_verified_positions = 0
        self.avg_distinct_experts = 0.0

    def reset(self) -> None:
        for pair in self.scalar.values():
            pair.reset()
        for pair in self.draft_pos:
            pair.reset()
        self.dirty = False
        self.num_verified_positions = 0
        self.avg_distinct_experts = 0.0

    def ready(self) -> bool:
        pairs = [p for p in self.scalar.values() if p.recorded]
        pairs += [p for p in self.draft_pos if p.recorded]
        return all(p.ready() for p in pairs)


class SpecDecodeTimer:
    """Double-buffered CUDA-event timer for speculative-decode stages.

    Usage per step::

        timer.begin_step()
        prev = timer.drain()  # timings for the previous step, or None
        with timer.time_stage("target_forward"):
            ...
        with timer.time_stage("draft", pos=i):
            ...

    Reporting lags by at least one step; under heavy device backlog a step's
    timing may be dropped rather than block the host (see ``drain``).
    """

    def __init__(self, enabled: bool, num_spec_tokens: int) -> None:
        self.enabled = enabled
        self.num_spec_tokens = num_spec_tokens
        if not enabled:
            return
        # Two slots: one being written this step, the other holding the previous
        # step's (now draining) events.
        self._slots = (_Slot(num_spec_tokens), _Slot(num_spec_tokens))
        self._write_idx = 0

    def begin_step(self) -> None:
        """Rotate to a fresh write slot before recording a new step."""
        if not self.enabled:
            return
        self._write_idx ^= 1
        self._slots[self._write_idx].reset()

    def set_num_verified_positions(self, num_positions: int) -> None:
        """Tag the current step with its target verification microbatch size."""
        if not self.enabled:
            return
        self._slots[self._write_idx].num_verified_positions = num_positions

    def set_avg_distinct_experts(self, avg_distinct_experts: float) -> None:
        """Tag the current step with its layer-averaged distinct-expert count."""
        if not self.enabled:
            return
        self._slots[self._write_idx].avg_distinct_experts = avg_distinct_experts

    @contextlib.contextmanager
    def time_stage(self, stage: str, pos: int | None = None) -> Iterator[None]:
        """Bracket a stage with start/end CUDA events on the current stream."""
        if not self.enabled:
            yield
            return
        slot = self._slots[self._write_idx]
        if pos is not None:
            if pos >= len(slot.draft_pos):
                yield
                return
            pair = slot.draft_pos[pos]
        elif stage in slot.scalar:
            pair = slot.scalar[stage]
        else:
            raise ValueError(
                f"Unknown scalar stage {stage!r}; per-position stages require pos="
            )
        slot.dirty = True
        pair.record_start()
        try:
            yield
        finally:
            pair.record_end()

    def drain(self) -> SpecDecodeTimingStats | None:
        """Read timings for the previous step if its events have completed.

        Returns ``None`` when the previous slot is empty or its events are not
        yet complete on the device, so the call never blocks.
        """
        if not self.enabled:
            return None
        prev = self._slots[self._write_idx ^ 1]
        if not prev.dirty or not prev.ready():
            return None
        prev.dirty = False
        # Draft positions are recorded contiguously from 0 (one forward per
        # speculative step), so stop at the first unrecorded position.
        draft_ms: list[float] = []
        for pair in prev.draft_pos:
            if not pair.recorded:
                break
            draft_ms.append(pair.elapsed_ms())
        return SpecDecodeTimingStats(
            target_forward_ms=self._read(prev, "target_forward"),
            verify_ms=self._read(prev, "verify"),
            sample_ms=self._read(prev, "sample"),
            draft_total_ms=self._read(prev, "draft_total"),
            draft_forward_ms_per_pos=draft_ms,
            num_verified_positions=prev.num_verified_positions,
            avg_distinct_experts=prev.avg_distinct_experts,
        )

    @staticmethod
    def _read(slot: _Slot, stage: str) -> float:
        pair = slot.scalar[stage]
        return pair.elapsed_ms() if pair.recorded else 0.0


def compute_avg_distinct_experts(routing_data: np.ndarray) -> float:
    """Layer-averaged distinct-expert count (Ū_r) for one verification step.

    Args:
        routing_data: Logical expert ids selected per (token, layer, slot),
            shape ``(num_tokens, num_layers, top_k)``, as captured by
            ``RoutedExpertsCapturer`` and sliced to the step's scheduled tokens.

    Returns:
        The mean over MoE layers of the number of distinct experts routed to in
        that layer. Layers with no routing (all-zero, i.e. dense/unused) are
        excluded. Returns 0.0 for an empty batch or a dense (non-MoE) target.
    """
    if routing_data.ndim != 3 or routing_data.shape[0] == 0:
        return 0.0
    counts: list[int] = []
    for layer in range(routing_data.shape[1]):
        experts = np.unique(routing_data[:, layer, :])
        # An all-zero slice means the layer was never routed (dense/unused);
        # expert id 0 is otherwise a valid expert.
        if experts.size == 1 and experts[0] == 0:
            continue
        counts.append(int(experts.size))
    return float(np.mean(counts)) if counts else 0.0

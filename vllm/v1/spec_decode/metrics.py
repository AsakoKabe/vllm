# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

if TYPE_CHECKING:
    from vllm.v1.spec_decode.timing import SpecDecodeTimingStats

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    # Per-step GPU stage timings in milliseconds. Zero unless spec-decode timing
    # is enabled; populated once per step via observe_timing().
    has_timing: bool = False
    target_forward_ms: float = 0.0
    verify_ms: float = 0.0
    sample_ms: float = 0.0
    draft_total_ms: float = 0.0
    draft_forward_ms_per_pos: list[float] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1

    def observe_timing(self, timing: "SpecDecodeTimingStats") -> None:
        """Fold one step's GPU stage timings into the stats (called once/step)."""
        self.has_timing = True
        self.target_forward_ms = timing.target_forward_ms
        self.verify_ms = timing.verify_ms
        self.sample_ms = timing.sample_ms
        self.draft_total_ms = timing.draft_total_ms
        self.draft_forward_ms_per_pos = list(timing.draft_forward_ms_per_pos)


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self, is_diffusion: bool = False):
        # Diffusion (dLLM) models reuse the spec-decode data path with
        # overloaded semantics, so the raw spec-decode framing (drafts, bonus
        # token, per-position vector) is logged with diffusion-native terms.
        self.is_diffusion = is_diffusion
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.target_forward_ms: list[float] = []
        self.verify_ms: list[float] = []
        self.sample_ms: list[float] = []
        self.draft_total_ms: list[float] = []
        self.draft_forward_ms_per_pos_lists: list[list[float]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        if spec_decoding_stats.has_timing:
            self.target_forward_ms.append(spec_decoding_stats.target_forward_ms)
            self.verify_ms.append(spec_decoding_stats.verify_ms)
            self.sample_ms.append(spec_decoding_stats.sample_ms)
            self.draft_total_ms.append(spec_decoding_stats.draft_total_ms)
            self.draft_forward_ms_per_pos_lists.append(
                spec_decoding_stats.draft_forward_ms_per_pos
            )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        self._log_timing(log_fn)
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        if self.is_diffusion:
            self._log_diffusion(
                log_fn,
                num_denoising_steps=num_drafts,
                num_canvas_tokens=num_draft_tokens,
                num_committed_tokens=num_accepted_tokens,
                committed_throughput=accepted_throughput,
            )
            self.reset()
            return

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()

    def _log_timing(self, log_fn):
        if not self.target_forward_ms:
            return
        n = len(self.target_forward_ms)
        target = float(np.sum(self.target_forward_ms)) / n
        draft = float(np.sum(self.draft_total_ms)) / n
        verify = float(np.sum(self.verify_ms)) / n
        sample = float(np.sum(self.sample_ms)) / n
        per_pos = self._mean_per_pos(self.draft_forward_ms_per_pos_lists)
        per_pos_str = ", ".join(f"{p:.3f}" for p in per_pos)
        log_fn(
            "SpecDecoding timing (mean ms/step over %d steps): "
            "target_forward: %.3f, draft_total: %.3f, verify: %.3f, "
            "sample: %.3f, per-position draft: [%s]",
            n,
            target,
            draft,
            verify,
            sample,
            per_pos_str,
        )

    @staticmethod
    def _mean_per_pos(lists: list[list[float]]) -> list[float]:
        max_k = max((len(x) for x in lists), default=0)
        sums = [0.0] * max_k
        counts = [0] * max_k
        for lst in lists:
            for i, value in enumerate(lst):
                sums[i] += value
                counts[i] += 1
        return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(max_k)]

    def _log_diffusion(
        self,
        log_fn,
        num_denoising_steps: int,
        num_canvas_tokens: int,
        num_committed_tokens: int,
        committed_throughput: float,
    ):
        # Each "draft" is one denoising step that re-evaluates the canvas block
        # and finalizes some of its positions.
        mean_committed_per_step = (
            num_committed_tokens / num_denoising_steps
            if num_denoising_steps > 0
            else float("nan")
        )
        mean_steps_per_canvas = (
            num_canvas_tokens / num_committed_tokens
            if num_committed_tokens > 0
            else float("nan")
        )

        log_fn(
            "DiffusionDecoding metrics: "
            "Committed token throughput: %.2f tokens/s, "
            "Mean denoising steps per canvas: %.2f, "
            "Mean tokens committed per denoising step: %.2f, "
            "Committed: %d tokens, "
            "Denoising steps: %d, "
            "Canvas positions evaluated: %d",
            committed_throughput,
            mean_steps_per_canvas,
            mean_committed_per_step,
            num_committed_tokens,
            num_denoising_steps,
            num_canvas_tokens,
        )


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        is_diffusion: bool = False,
        enable_timing: bool = False,
    ):
        # Diffusion (dLLM) models reuse the spec-decode counters but expose them
        # under diffusion-native names; the per-position acceptance vector does
        # not apply, so it is omitted.
        self.is_diffusion = is_diffusion
        self.spec_decoding_enabled = speculative_config is not None or is_diffusion
        if not self.spec_decoding_enabled:
            return

        if is_diffusion:
            counter_specs = [
                ("vllm:diffusion_num_denoising_steps", "Number of denoising steps."),
                (
                    "vllm:diffusion_num_canvas_positions",
                    "Number of canvas positions evaluated.",
                ),
                (
                    "vllm:diffusion_num_committed_tokens",
                    "Number of committed (finalized) tokens.",
                ),
            ]
        else:
            counter_specs = [
                ("vllm:spec_decode_num_drafts", "Number of spec decoding drafts."),
                ("vllm:spec_decode_num_draft_tokens", "Number of draft tokens."),
                ("vllm:spec_decode_num_accepted_tokens", "Number of accepted tokens."),
            ]

        counters = [
            create_metric_per_engine(
                self._counter_cls(name=name, documentation=doc, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for name, doc in counter_specs
        ]
        # num_drafts/num_draft_tokens/num_accepted_tokens map onto denoising
        # steps/canvas positions/committed tokens in the diffusion path.
        self.counter_spec_decode_num_drafts = counters[0]
        self.counter_spec_decode_num_draft_tokens = counters[1]
        self.counter_spec_decode_num_accepted_tokens = counters[2]

        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if not is_diffusion:
            assert speculative_config is not None
            num_spec_tokens = speculative_config.num_speculative_tokens
            pos_labelnames = labelnames + ["position"]
            base_counter = self._counter_cls(
                name="vllm:spec_decode_num_accepted_tokens_per_pos",
                documentation="Accepted tokens per draft position.",
                labelnames=pos_labelnames,
            )
            self.counter_spec_decode_num_accepted_tokens_per_pos = {
                idx: [
                    base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }

        # Stage timing (microseconds; integer counters so values survive the
        # int coercion in metrics.reader). Off unless --spec-decode-timing.
        self.enable_timing = enable_timing
        self.counter_spec_decode_timing: dict[str, list[prometheus_client.Counter]] = {}
        self.counter_spec_decode_draft_forward_us_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if not enable_timing:
            return
        timing_specs = [
            (
                "vllm:spec_decode_target_forward_microseconds",
                "Target forward (verification) time in microseconds.",
            ),
            (
                "vllm:spec_decode_verify_microseconds",
                "Rejection-sampling verification time in microseconds.",
            ),
            ("vllm:spec_decode_sample_microseconds", "Sampling time in microseconds."),
            (
                "vllm:spec_decode_draft_total_microseconds",
                "Total draft-generation time in microseconds.",
            ),
            (
                "vllm:spec_decode_num_timed_steps",
                "Number of speculative steps with recorded timing.",
            ),
        ]
        timing_counters = [
            create_metric_per_engine(
                self._counter_cls(name=name, documentation=doc, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for name, doc in timing_specs
        ]
        self.counter_spec_decode_timing = {
            "target_forward": timing_counters[0],
            "verify": timing_counters[1],
            "sample": timing_counters[2],
            "draft_total": timing_counters[3],
            "num_timed_steps": timing_counters[4],
        }
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        if num_spec_tokens > 0:
            pos_labelnames = labelnames + ["position"]
            base_draft_us = self._counter_cls(
                name="vllm:spec_decode_draft_forward_microseconds_per_pos",
                documentation="Per-position draft forward time in microseconds.",
                labelnames=pos_labelnames,
            )
            self.counter_spec_decode_draft_forward_us_per_pos = {
                idx: [
                    base_draft_us.labels(*lv, str(pos))
                    for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos.get(engine_idx, [])
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])

        if not (self.enable_timing and spec_decoding_stats.has_timing):
            return
        timing = self.counter_spec_decode_timing
        timing["target_forward"][engine_idx].inc(
            _ms_to_us(spec_decoding_stats.target_forward_ms)
        )
        timing["verify"][engine_idx].inc(_ms_to_us(spec_decoding_stats.verify_ms))
        timing["sample"][engine_idx].inc(_ms_to_us(spec_decoding_stats.sample_ms))
        timing["draft_total"][engine_idx].inc(
            _ms_to_us(spec_decoding_stats.draft_total_ms)
        )
        timing["num_timed_steps"][engine_idx].inc(1)
        per_pos = self.counter_spec_decode_draft_forward_us_per_pos.get(engine_idx, [])
        for pos, value in enumerate(spec_decoding_stats.draft_forward_ms_per_pos):
            if pos < len(per_pos):
                per_pos[pos].inc(_ms_to_us(value))


def _ms_to_us(ms: float) -> int:
    """Convert milliseconds to integer microseconds for counter increments."""
    return int(round(ms * 1000.0))

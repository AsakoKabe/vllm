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
    from vllm.v1.spec_decode.evict.stats import EvictStats
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
    # Layer-averaged distinct-expert count (Ū_r) over the verification
    # microbatch; 0.0 for dense targets or when routing capture is unavailable.
    avg_distinct_experts: float = 0.0
    # Number of verified positions this step (sum of K_i + 1 over requests).
    # 0 on non-spec/prefill steps. Used to bin target_forward by position count
    # so T_T(0)=bin[1] and T_T(K)=bin[K+1] fall out of one run.
    num_verified_positions: int = 0

    # EVICT truncation for this step. Set only when EVICT trimmed the chain;
    # populated once per step via observe_evict().
    has_evict: bool = False
    evict_kstar: int = 0
    evict_num_spec: int = 0
    evict_num_reqs: int = 0
    evict_saved_positions: int = 0

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
        self.avg_distinct_experts = timing.avg_distinct_experts
        self.num_verified_positions = timing.num_verified_positions

    def observe_evict(self, evict: "EvictStats") -> None:
        """Fold one step's EVICT truncation decision (called once/step)."""
        self.has_evict = True
        self.evict_kstar = evict.kstar
        self.evict_num_spec = evict.num_spec
        self.evict_num_reqs = evict.num_reqs
        self.evict_saved_positions = evict.saved_positions


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
        self.avg_distinct_experts: list[float] = []
        self.num_verified_positions: list[int] = []
        self.evict_kstar: list[int] = []
        self.evict_num_spec: list[int] = []
        self.evict_saved_positions: list[int] = []
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
            self.avg_distinct_experts.append(spec_decoding_stats.avg_distinct_experts)
            self.num_verified_positions.append(
                spec_decoding_stats.num_verified_positions
            )
        if spec_decoding_stats.has_evict:
            self.evict_kstar.append(spec_decoding_stats.evict_kstar)
            self.evict_num_spec.append(spec_decoding_stats.evict_num_spec)
            self.evict_saved_positions.append(spec_decoding_stats.evict_saved_positions)

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        self._log_timing(log_fn)
        self._log_evict(log_fn)
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
        avg_experts = (
            float(np.sum(self.avg_distinct_experts)) / n
            if self.avg_distinct_experts
            else 0.0
        )
        log_fn(
            "SpecDecoding timing (mean ms/step over %d steps): "
            "target_forward: %.3f, draft_total: %.3f, verify: %.3f, "
            "sample: %.3f, per-position draft: [%s], avg_distinct_experts: %.2f",
            n,
            target,
            draft,
            verify,
            sample,
            per_pos_str,
            avg_experts,
        )
        self._log_target_by_positions(log_fn)

    def _log_target_by_positions(self, log_fn):
        """Log T_T(k) binned by verified positions, plus eta(K)=T_T(0)/T_T(K)."""
        pairs = [
            (k, ms)
            for k, ms in zip(self.num_verified_positions, self.target_forward_ms)
            if k >= 1
        ]
        if not pairs:
            return
        by_k: dict[int, list[float]] = {}
        for k, ms in pairs:
            by_k.setdefault(k, []).append(ms)
        means = {k: float(np.mean(v)) for k, v in by_k.items()}
        # Paper index: T_T(j) is the forward over j+1 positions, so bin[1]=T_T(0).
        bins_str = ", ".join(
            f"{k - 1}:{means[k]:.3f}({len(by_k[k])})" for k in sorted(means)
        )
        t_t0 = means.get(1)
        k_max = max(means)
        t_tk = means.get(k_max)
        eta = t_t0 / t_tk if t_t0 and t_tk else float("nan")
        log_fn(
            "SpecDecoding T_T by positions (mean ms, paper K index, count): [%s]; "
            "T_T(0): %s, T_T(%d): %.3f, eta(%d): %.3f",
            bins_str,
            f"{t_t0:.3f}" if t_t0 is not None else "n/a",
            k_max - 1,
            t_tk,
            k_max - 1,
            eta,
        )

    def _log_evict(self, log_fn):
        """Log EVICT truncation: mean m*, saved verify positions, saved %."""
        if not self.evict_kstar:
            return
        steps = len(self.evict_kstar)
        mean_kstar = float(np.mean(self.evict_kstar))
        mean_k = float(np.mean(self.evict_num_spec))
        saved = int(np.sum(self.evict_saved_positions))
        # Per-step truncated fraction (K - m*)/K is batch-size independent;
        # average it over the truncation steps.
        saved_frac = float(
            np.mean(
                [
                    (k - ks) / k
                    for k, ks in zip(self.evict_num_spec, self.evict_kstar)
                    if k > 0
                ]
            )
        )
        log_fn(
            "SpecDecoding EVICT (%d truncation steps): mean m*: %.2f, mean K: %.2f, "
            "saved verify positions: %d (mean %.1f%% of chain truncated)",
            steps,
            mean_kstar,
            mean_k,
            saved,
            saved_frac * 100.0,
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
        # target_forward binned by number of verified positions k (index == k).
        # T_T(k-1) = sum_us[k] / count[k] / 1000 ms; hence T_T(0)=bin[1],
        # T_T(K)=bin[K+1], and eta(K) = bin[1] / bin[K+1].
        self.counter_spec_decode_target_forward_us_by_positions: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        self.counter_spec_decode_target_forward_count_by_positions: dict[
            int, list[prometheus_client.Counter]
        ] = {}

        # EVICT truncation metrics — independent of --spec-decode-timing; on
        # whenever EVICT is enabled. mean m* = evict_kstar_sum / evict_steps.
        self.enable_evict = bool(
            speculative_config is not None
            and getattr(speculative_config, "evict_enabled", False)
        )
        self.counter_spec_decode_evict: dict[str, list[prometheus_client.Counter]] = {}
        self.counter_spec_decode_evict_kstar_hist: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if self.enable_evict:
            evict_specs = [
                (
                    "vllm:spec_decode_evict_steps",
                    "Steps where EVICT evaluated/truncated the drafted chain.",
                ),
                (
                    "vllm:spec_decode_evict_kstar_sum",
                    "Sum of the chosen verified prefix m* over EVICT steps; "
                    "mean m* = value / evict_steps.",
                ),
                (
                    "vllm:spec_decode_evict_saved_positions",
                    "Verify positions skipped by EVICT: sum of (K - m*) * B.",
                ),
            ]
            evict_counters = [
                create_metric_per_engine(
                    self._counter_cls(
                        name=name, documentation=doc, labelnames=labelnames
                    ),
                    per_engine_labelvalues,
                )
                for name, doc in evict_specs
            ]
            self.counter_spec_decode_evict = {
                "steps": evict_counters[0],
                "kstar_sum": evict_counters[1],
                "saved_positions": evict_counters[2],
            }
            evict_num_spec = (
                speculative_config.num_speculative_tokens
                if speculative_config is not None
                else 0
            )
            if evict_num_spec > 0:
                pos_labelnames = labelnames + ["position"]
                base_kstar_hist = self._counter_cls(
                    name="vllm:spec_decode_evict_kstar_hist",
                    documentation="Count of EVICT steps by chosen m* (index == m*).",
                    labelnames=pos_labelnames,
                )
                self.counter_spec_decode_evict_kstar_hist = {
                    idx: [
                        base_kstar_hist.labels(*lv, str(m))
                        for m in range(evict_num_spec + 1)
                    ]
                    for idx, lv in per_engine_labelvalues.items()
                }

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
            (
                "vllm:spec_decode_distinct_experts_milli",
                "Layer-averaged distinct experts (Ū_r) x1000, summed over timed "
                "steps; mean Ū_r = value / 1000 / num_timed_steps.",
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
            "distinct_experts": timing_counters[5],
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

            # Verified-position bins run 0..K+1 (index k). Index 0 is unused
            # (prefill steps are excluded); k in 1..K+1 hold target_forward for
            # steps that verified exactly k positions.
            num_position_bins = num_spec_tokens + 2
            base_tf_us = self._counter_cls(
                name="vllm:spec_decode_target_forward_microseconds_by_positions",
                documentation=(
                    "Target forward time in microseconds, summed over steps that "
                    "verified exactly 'position' positions."
                ),
                labelnames=pos_labelnames,
            )
            base_tf_count = self._counter_cls(
                name="vllm:spec_decode_target_forward_count_by_positions",
                documentation=(
                    "Number of steps that verified exactly 'position' positions "
                    "(denominator for target_forward_microseconds_by_positions)."
                ),
                labelnames=pos_labelnames,
            )
            self.counter_spec_decode_target_forward_us_by_positions = {
                idx: [base_tf_us.labels(*lv, str(k)) for k in range(num_position_bins)]
                for idx, lv in per_engine_labelvalues.items()
            }
            self.counter_spec_decode_target_forward_count_by_positions = {
                idx: [
                    base_tf_count.labels(*lv, str(k)) for k in range(num_position_bins)
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

        # EVICT truncation (independent of timing; only when EVICT truncated).
        if self.enable_evict and spec_decoding_stats.has_evict:
            evict = self.counter_spec_decode_evict
            evict["steps"][engine_idx].inc(1)
            evict["kstar_sum"][engine_idx].inc(spec_decoding_stats.evict_kstar)
            evict["saved_positions"][engine_idx].inc(
                spec_decoding_stats.evict_saved_positions
            )
            hist = self.counter_spec_decode_evict_kstar_hist.get(engine_idx, [])
            m = spec_decoding_stats.evict_kstar
            if 0 <= m < len(hist):
                hist[m].inc(1)

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
        timing["distinct_experts"][engine_idx].inc(
            int(round(spec_decoding_stats.avg_distinct_experts * 1000))
        )
        per_pos = self.counter_spec_decode_draft_forward_us_per_pos.get(engine_idx, [])
        for pos, value in enumerate(spec_decoding_stats.draft_forward_ms_per_pos):
            if pos < len(per_pos):
                per_pos[pos].inc(_ms_to_us(value))

        # Bin the target forward by the number of verified positions so T_T(k)
        # can be recovered per position count. Prefill/non-spec steps report 0
        # positions and are excluded; k is capped at the vector width (K+1).
        k = spec_decoding_stats.num_verified_positions
        us_bins = self.counter_spec_decode_target_forward_us_by_positions.get(
            engine_idx, []
        )
        count_bins = self.counter_spec_decode_target_forward_count_by_positions.get(
            engine_idx, []
        )
        if 1 <= k < len(us_bins):
            us_bins[k].inc(_ms_to_us(spec_decoding_stats.target_forward_ms))
            count_bins[k].inc(1)


def _ms_to_us(ms: float) -> int:
    """Convert milliseconds to integer microseconds for counter increments."""
    return int(round(ms * 1000.0))

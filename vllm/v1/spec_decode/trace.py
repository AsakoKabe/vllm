# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round speculative-decoding trace.

Writes one JSONL record per speculative round (scheduler step with spec
activity), so every metric is available as a per-round array instead of a
cumulative counter: draft tokens fed/generated, verification submitted /
accepted / rejected / bonus (total and per position), stage timings, the
verification microbatch size, the layer-averaged distinct-expert count U_r,
and the EVICT truncation decision.

Enable with ``--spec-decode-trace-path PATH`` (requires log stats, i.e.
``disable_log_stats=False``). Timing/U_r fields additionally require
``--spec-decode-timing`` (and routed-experts capture for U_r); they are null
otherwise.

File layout: the first line is a meta header ``{"meta": {...}}``; every
following line is one round. Load back with :func:`load_trace`, which returns
column-oriented arrays.

Alignment caveat (double-buffered CUDA-event timing): the timing block of
record N (``target_forward_ms`` .. ``avg_distinct_experts``) was measured on
round N-1, while the acceptance block of record N describes round N itself.
``num_verified_positions`` and ``avg_distinct_experts`` come from the same
timing slot, so pairs like (T_T, U_r) ARE aligned with each other; to pair
timing with acceptance, shift one of the blocks by one record.
"""

import json
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.metrics.loggers import StatLoggerBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import (
        IterationStats,
        MultiModalCacheStats,
        SchedulerStats,
    )

logger = init_logger(__name__)

TRACE_SCHEMA_VERSION = 1


class SpecDecodeTraceLogger(StatLoggerBase):
    """Stat logger that appends one JSONL record per speculative round."""

    def __init__(self, vllm_config: "VllmConfig", engine_index: int = 0):
        self.engine_index = engine_index
        path = vllm_config.observability_config.spec_decode_trace_path
        assert path, "SpecDecodeTraceLogger requires spec_decode_trace_path"
        p = Path(path)
        if engine_index != 0:
            # One file per engine so DP ranks do not interleave writes.
            p = p.with_name(f"{p.stem}.engine{engine_index}{p.suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered so every round is durable without an explicit close
        # hook (StatLoggerBase has none); the handle lives as long as the
        # logger, so a context manager does not apply.
        self._file: IO[str] = open(p, "w", buffering=1)  # noqa: SIM115
        self._step = 0
        model = (
            vllm_config.model_config.model
            if vllm_config.model_config is not None
            else None
        )
        meta = {
            "meta": {
                "schema_version": TRACE_SCHEMA_VERSION,
                "engine_index": engine_index,
                "model": model,
                "num_spec_tokens": getattr(vllm_config, "num_speculative_tokens", 0),
                "timing_lag_note": (
                    "timing fields of record N were measured on round N-1; "
                    "num_verified_positions/avg_distinct_experts share that "
                    "slot, so (T_T, U_r) pairs are aligned with each other"
                ),
            }
        }
        self._file.write(json.dumps(meta) + "\n")
        logger.info("Writing per-round spec-decode trace to %s", p)

    def record(
        self,
        scheduler_stats: "SchedulerStats | None",
        iteration_stats: "IterationStats | None",
        mm_cache_stats: "MultiModalCacheStats | None" = None,
        engine_idx: int = 0,
    ):
        if scheduler_stats is None:
            return
        stats = scheduler_stats.spec_decoding_stats
        if stats is None:
            return
        if not (stats.num_drafts or stats.has_timing or stats.has_evict):
            return

        record: dict[str, Any] = {
            "step": self._step,
            # Verification of this round: how many requests carried a draft
            # (== bonus tokens), how many draft tokens were submitted to the
            # target, and how many were accepted / rejected, per position too.
            "num_drafts": stats.num_drafts,
            "num_draft_tokens": stats.num_draft_tokens,
            "num_accepted_tokens": stats.num_accepted_tokens,
            "num_rejected_tokens": stats.num_draft_tokens - stats.num_accepted_tokens,
            "num_bonus_tokens": stats.num_drafts,
            "num_emitted_tokens": stats.num_accepted_tokens + stats.num_drafts,
            "drafted_per_pos": list(stats.num_draft_tokens_per_pos),
            "accepted_per_pos": list(stats.num_accepted_tokens_per_pos),
        }
        if stats.has_timing:
            record.update(
                target_forward_ms=stats.target_forward_ms,
                verify_ms=stats.verify_ms,
                sample_ms=stats.sample_ms,
                draft_total_ms=stats.draft_total_ms,
                draft_forward_ms_per_pos=list(stats.draft_forward_ms_per_pos),
                num_verified_positions=stats.num_verified_positions,
                avg_distinct_experts=stats.avg_distinct_experts,
            )
        if stats.has_evict:
            record.update(
                evict_kstar=stats.evict_kstar,
                evict_num_spec=stats.evict_num_spec,
                evict_num_reqs=stats.evict_num_reqs,
                evict_saved_positions=stats.evict_saved_positions,
            )
        self._file.write(json.dumps(record) + "\n")
        self._step += 1

    def log_engine_initialized(self):
        pass


def load_trace(path: str | Path) -> dict[str, Any]:
    """Load a trace back as column-oriented per-round arrays.

    Returns ``{"meta": {...}, "<field>": [v_round0, v_round1, ...], ...}``.
    Fields absent in a record (e.g. timing when spec_decode_timing was off, or
    EVICT fields on non-truncated rounds) are filled with ``None`` so every
    array has one entry per round.
    """
    meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj and not records and not meta:
                meta = obj["meta"]
                continue
            records.append(obj)

    fields: list[str] = []
    seen = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    columns: dict[str, Any] = {
        field: [rec.get(field) for rec in records] for field in fields
    }
    columns["meta"] = meta
    return columns

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-round speculative-decoding trace (CPU-only)."""

import json
from pathlib import Path
from types import SimpleNamespace

from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.spec_decode.evict.stats import EvictStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.spec_decode.timing import SpecDecodeTimingStats
from vllm.v1.spec_decode.trace import SpecDecodeTraceLogger, load_trace


def _make_logger(tmp_path, engine_index: int = 0) -> tuple[SpecDecodeTraceLogger, str]:
    path = str(tmp_path / "trace.jsonl")
    vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(spec_decode_trace_path=path),
        model_config=SimpleNamespace(
            model="org/model",
            hf_text_config=SimpleNamespace(num_experts_per_tok=8),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        num_speculative_tokens=4,
    )
    return SpecDecodeTraceLogger(vllm_config, engine_index), path


def _spec_stats(
    drafts: int = 1,
    draft_tokens: int = 4,
    accepted: int = 2,
    timing: SpecDecodeTimingStats | None = None,
    evict: EvictStats | None = None,
) -> SpecDecodingStats:
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    for _ in range(drafts):
        stats.observe_draft(num_draft_tokens=draft_tokens, num_accepted_tokens=accepted)
    if timing is not None:
        stats.observe_timing(timing)
    if evict is not None:
        stats.observe_evict(evict)
    return stats


def test_trace_writes_meta_and_per_round_records(tmp_path):
    trace_logger, path = _make_logger(tmp_path)

    timing = SpecDecodeTimingStats(
        target_forward_ms=10.5,
        verify_ms=0.3,
        sample_ms=0.0,
        draft_total_ms=3.9,
        draft_forward_ms_per_pos=[0.15, 0.15, 0.15, 0.15],
        num_verified_positions=5,
        avg_distinct_experts=22.5,
        moe_max_tokens_per_expert=8.0,
    )
    evict = EvictStats(kstar=4, num_spec=8, num_reqs=1, saved_positions=4)
    trace_logger.record(
        SchedulerStats(spec_decoding_stats=_spec_stats(timing=timing, evict=evict)),
        iteration_stats=None,
    )
    trace_logger.record(
        SchedulerStats(spec_decoding_stats=_spec_stats(accepted=0)),
        iteration_stats=None,
    )

    lines = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert lines[0]["meta"]["schema_version"] == 2
    assert lines[0]["meta"]["model"] == "org/model"
    assert lines[0]["meta"]["moe_top_k"] == 8
    assert lines[0]["meta"]["max_num_seqs"] == 1

    first = lines[1]
    assert first["step"] == 0
    assert first["num_draft_tokens"] == 4
    assert first["num_accepted_tokens"] == 2
    assert first["num_rejected_tokens"] == 2
    assert first["num_bonus_tokens"] == 1
    assert first["num_emitted_tokens"] == 3
    assert first["accepted_per_pos"] == [1, 1, 0, 0]
    assert first["drafted_per_pos"] == [1, 1, 1, 1]
    assert first["target_forward_ms"] == 10.5
    assert first["num_verified_positions"] == 5
    assert first["avg_distinct_experts"] == 22.5
    assert first["moe_max_tokens_per_expert"] == 8.0
    assert first["evict_kstar"] == 4
    assert first["evict_saved_positions"] == 4

    second = lines[2]
    assert second["step"] == 1
    assert second["num_accepted_tokens"] == 0
    # No timing/evict on this round -> fields absent from the record.
    assert "target_forward_ms" not in second
    assert "evict_kstar" not in second


def test_trace_meta_tolerates_minimal_config(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(spec_decode_trace_path=path),
        model_config=None,
    )
    SpecDecodeTraceLogger(vllm_config)
    meta = json.loads(Path(path).read_text().splitlines()[0])["meta"]
    assert meta["moe_top_k"] is None
    assert meta["max_num_seqs"] is None
    assert meta["num_spec_tokens"] == 0


def test_trace_records_scheduler_covariates(tmp_path):
    trace_logger, path = _make_logger(tmp_path)
    cudagraph_stats = SimpleNamespace(
        num_unpadded_tokens=5,
        num_padded_tokens=9,
        num_paddings=4,
        runtime_mode="FULL",
    )
    trace_logger.record(
        SchedulerStats(
            num_running_reqs=1,
            num_waiting_reqs=3,
            kv_cache_usage=0.42,
            total_context_tokens=1234,
            cudagraph_stats=cudagraph_stats,
            spec_decoding_stats=_spec_stats(),
        ),
        iteration_stats=None,
    )
    trace_logger.record(
        SchedulerStats(
            num_running_reqs=2,
            total_context_tokens=2000,
            spec_decoding_stats=_spec_stats(accepted=1),
        ),
        iteration_stats=None,
    )

    lines = [json.loads(line) for line in Path(path).read_text().splitlines()]
    first = lines[1]
    assert first["num_running_reqs"] == 1
    assert first["num_waiting_reqs"] == 3
    assert first["kv_cache_usage"] == 0.42
    assert first["total_context_tokens"] == 1234
    assert first["cudagraph_padded_tokens"] == 9
    assert first["cudagraph_num_paddings"] == 4
    assert first["cudagraph_runtime_mode"] == "FULL"

    second = lines[2]
    assert second["num_running_reqs"] == 2
    assert second["total_context_tokens"] == 2000
    # Covariates always present; cudagraph_* only when the stat is provided.
    assert "kv_cache_usage" in second
    assert "cudagraph_padded_tokens" not in second


def test_trace_skips_empty_steps(tmp_path):
    trace_logger, path = _make_logger(tmp_path)
    # No scheduler stats / no spec activity -> nothing written.
    trace_logger.record(None, iteration_stats=None)
    trace_logger.record(SchedulerStats(spec_decoding_stats=None), iteration_stats=None)
    trace_logger.record(
        SchedulerStats(spec_decoding_stats=SpecDecodingStats.new(num_spec_tokens=4)),
        iteration_stats=None,
    )
    lines = Path(path).read_text().splitlines()
    assert len(lines) == 1  # meta only


def test_trace_engine_index_gets_own_file(tmp_path):
    trace_logger, path = _make_logger(tmp_path, engine_index=2)
    expected = str(tmp_path / "trace.engine2.jsonl")
    trace_logger.record(
        SchedulerStats(spec_decoding_stats=_spec_stats()), iteration_stats=None
    )
    assert len(Path(expected).read_text().splitlines()) == 2


def test_load_trace_returns_column_arrays(tmp_path):
    trace_logger, path = _make_logger(tmp_path)
    timing = SpecDecodeTimingStats(target_forward_ms=7.0)
    trace_logger.record(
        SchedulerStats(spec_decoding_stats=_spec_stats(timing=timing)),
        iteration_stats=None,
    )
    trace_logger.record(
        SchedulerStats(spec_decoding_stats=_spec_stats(accepted=1)),
        iteration_stats=None,
    )

    cols = load_trace(path)
    assert cols["meta"]["num_spec_tokens"] == 4
    assert cols["step"] == [0, 1]
    assert cols["num_accepted_tokens"] == [2, 1]
    # Field present only in round 0 -> None-padded to full length.
    assert cols["target_forward_ms"] == [7.0, None]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the EVICT cost model ``C(m)``."""

import json

import pytest
import torch

from vllm.v1.spec_decode.evict.cost_table import CostTable


def test_affine_as_tensor():
    table = CostTable.affine(intercept=1.0, per_token=0.5)
    assert table.is_affine
    assert torch.allclose(table.as_tensor(3), torch.tensor([1.5, 2.0, 2.5]))


def test_affine_validates_params():
    with pytest.raises(ValueError):
        CostTable.affine(intercept=0.0, per_token=0.5)
    with pytest.raises(ValueError):
        CostTable.affine(intercept=1.0, per_token=-1.0)


def test_from_file_flat_mapping(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"1": 1.5, "2": 2.1, "3": 2.4}))
    table = CostTable.from_file(path)
    assert not table.is_affine
    assert table.max_profiled_m == 3
    assert torch.allclose(table.as_tensor(3), torch.tensor([1.5, 2.1, 2.4]))


def test_from_file_nested_costs_key(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"unit": "ms", "costs": {"1": 1.0, "2": 1.8}}))
    table = CostTable.from_file(path)
    assert torch.allclose(table.as_tensor(2), torch.tensor([1.0, 1.8]))


def test_from_file_fails_closed_on_missing_length(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"1": 1.0, "3": 2.0}))  # missing m=2
    with pytest.raises(ValueError):
        CostTable.from_file(path)


def test_from_file_fails_closed_on_non_positive(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"1": 1.0, "2": 0.0}))
    with pytest.raises(ValueError):
        CostTable.from_file(path)


def test_from_file_must_start_at_one(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"2": 1.0, "3": 2.0}))
    with pytest.raises(ValueError):
        CostTable.from_file(path)


def test_from_file_missing_file():
    with pytest.raises(FileNotFoundError):
        CostTable.from_file("/nonexistent/evict_cost.json")


def test_as_tensor_rejects_beyond_profiled(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"1": 1.0, "2": 1.8}))
    table = CostTable.from_file(path)
    with pytest.raises(ValueError):
        table.as_tensor(3)  # profiled only up to 2


def test_affine_as_tensor_any_depth():
    table = CostTable.affine(intercept=2.0, per_token=1.0)
    assert torch.allclose(table.as_tensor(4), torch.tensor([3.0, 4.0, 5.0, 6.0]))


def test_from_file_records_provenance(tmp_path):
    p = tmp_path / "table.json"
    p.write_text(
        json.dumps(
            {
                "unit": "ms",
                "model": "org/model-a",
                "method": "eagle3",
                "costs": {"1": 1.0, "2": 2.0},
            }
        )
    )
    table = CostTable.from_file(p)
    assert table.source_model == "org/model-a"
    assert table.source_method == "eagle3"


def test_from_file_flat_mapping_has_no_provenance(tmp_path):
    p = tmp_path / "table.json"
    p.write_text(json.dumps({"1": 1.0}))
    table = CostTable.from_file(p)
    assert table.source_model is None
    assert table.source_method is None


def test_validate_source_accepts_match_and_missing_provenance():
    table = CostTable(
        costs={1: 1.0}, source_model="org/model-a", source_method="eagle3"
    )
    table.validate_source("org/model-a", "eagle3")
    # No provenance (hand-written table) -> accepted against anything.
    CostTable(costs={1: 1.0}).validate_source("org/other", "mtp")
    # Unknown deployment side -> accepted.
    table.validate_source(None, None)


def test_validate_source_fails_closed_on_mismatch():
    table = CostTable(
        costs={1: 1.0}, source_model="org/model-a", source_method="eagle3"
    )
    with pytest.raises(ValueError, match="profiled for model"):
        table.validate_source("org/model-b", "eagle3")
    with pytest.raises(ValueError, match="profiled with method"):
        table.validate_source("org/model-a", "mtp")

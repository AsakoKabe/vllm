# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EVICT verification-length selection (pure-torch math)."""

import pytest
import torch

from vllm.v1.spec_decode.evict.selector import (
    expected_accepted_length,
    gather_draft_confidence,
    reduce_batch_kstar,
    select_kstar,
)


def test_gather_draft_confidence_picks_chosen_token_prob():
    # [B=1, K=2, V=3]
    probs = torch.tensor([[[0.1, 0.7, 0.2], [0.5, 0.4, 0.1]]])
    ids = torch.tensor([[1, 0]])  # chose token 1 then token 0
    q = gather_draft_confidence(probs, ids)
    assert torch.allclose(q, torch.tensor([[0.7, 0.5]]))


def test_expected_accepted_length_is_cumsum_of_cumprod():
    conf = torch.tensor([[0.9, 0.8, 0.5]])
    # score = [0.9, 0.72, 0.36]; ehat = [0.9, 1.62, 1.98]
    ehat = expected_accepted_length(conf)
    assert torch.allclose(ehat, torch.tensor([[0.9, 1.62, 1.98]]), atol=1e-6)


def test_select_kstar_picks_utility_peak():
    conf = torch.tensor([[0.9, 0.8, 0.5]])
    cost = torch.tensor([1.0, 1.5, 2.0])  # C(1),C(2),C(3)
    # U = [0.9, 1.08, 0.99] -> argmax at m=2
    kstar = select_kstar(conf, cost, min_k=1)
    assert kstar.tolist() == [2]


def test_select_kstar_high_confidence_with_overhead_verifies_all():
    conf = torch.tensor([[0.99, 0.99, 0.99]])
    cost = torch.tensor([2.0, 3.0, 4.0])  # intercept 1, slope 1
    # U = [0.495, 0.657, 0.735] -> m=3
    assert select_kstar(conf, cost, min_k=1).tolist() == [3]


def test_select_kstar_min_k_clamp():
    conf = torch.tensor([[0.5, 0.1, 0.01]])
    cost = torch.tensor([1.0, 2.0, 3.0])
    # Unclamped peak is m=1; min_k=2 forces >= 2.
    assert select_kstar(conf, cost, min_k=1).tolist() == [1]
    assert select_kstar(conf, cost, min_k=2).tolist() == [2]


def test_select_kstar_batch_independent():
    conf = torch.tensor([[0.9, 0.8, 0.5], [0.5, 0.1, 0.01]])
    cost = torch.tensor([1.0, 1.5, 2.0])
    kstar = select_kstar(conf, cost, min_k=1)
    assert kstar.tolist() == [2, 1]


def test_select_kstar_single_token_chain():
    conf = torch.tensor([[0.3]])
    cost = torch.tensor([1.0])
    assert select_kstar(conf, cost, min_k=1).tolist() == [1]


def test_select_kstar_validates_cost_shape():
    conf = torch.tensor([[0.9, 0.8]])
    with pytest.raises(ValueError):
        select_kstar(conf, torch.tensor([1.0, 2.0, 3.0]), min_k=1)


def test_reduce_batch_kstar_policies():
    kstar = torch.tensor([2, 4, 3])
    assert reduce_batch_kstar(kstar, "max") == 4
    assert reduce_batch_kstar(kstar, "min") == 2
    assert reduce_batch_kstar(kstar, "median") == 3
    with pytest.raises(ValueError):
        reduce_batch_kstar(kstar, "bogus")

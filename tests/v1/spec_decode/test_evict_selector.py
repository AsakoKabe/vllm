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


def test_select_kstar_collapses_on_fabricated_low_confidence():
    # A greedy request in a mixed batch gets a fabricated temperature=1 softmax
    # confidence (~1/V for a large vocab), unrelated to its deterministic
    # acceptance; select_kstar then collapses it to min_k. This is exactly why
    # _apply_evict_truncation excludes any batch containing a greedy request
    # (the all_random guard) — this test documents the input that guard protects.
    conf = torch.full((1, 4), 2e-5)  # ~1/V confidences
    cost = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert select_kstar(conf, cost, min_k=1).tolist() == [1]


def test_reduce_batch_kstar_policies():
    kstar = torch.tensor([2, 4, 3])
    assert reduce_batch_kstar(kstar, "max") == 4
    assert reduce_batch_kstar(kstar, "min") == 2
    assert reduce_batch_kstar(kstar, "median") == 3
    with pytest.raises(ValueError):
        reduce_batch_kstar(kstar, "bogus")


def test_select_kstar_allowed_mask_quantizes_to_set():
    conf = torch.tensor([[0.9, 0.8, 0.5]])
    cost = torch.tensor([1.0, 1.5, 2.0])
    # Unrestricted argmax is m=2 (see test_select_kstar_picks_utility_peak);
    # with only {1, 3} allowed the best of the set must be chosen instead.
    mask = torch.tensor([True, False, True])
    kstar = select_kstar(conf, cost, min_k=1, allowed_mask=mask)
    # U = [0.9, -, 0.99] -> m=3
    assert kstar.tolist() == [3]


def test_select_kstar_allowed_mask_respects_min_k():
    conf = torch.tensor([[0.99, 0.9, 0.1]])
    cost = torch.tensor([1.0, 1.1, 1.2])
    # m=1 is allowed but below min_k=2; the selector must pick from
    # allowed ∩ [min_k, K] = {2}.
    mask = torch.tensor([True, True, False])
    kstar = select_kstar(conf, cost, min_k=2, allowed_mask=mask)
    assert kstar.tolist() == [2]


def test_select_kstar_allowed_mask_no_valid_choice_raises():
    conf = torch.tensor([[0.9, 0.8, 0.5]])
    cost = torch.tensor([1.0, 1.5, 2.0])
    # Only m=1 allowed but min_k=2 -> empty intersection must raise, matching
    # the config-level validation.
    mask = torch.tensor([True, False, False])
    with pytest.raises(ValueError, match="no selectable length"):
        select_kstar(conf, cost, min_k=2, allowed_mask=mask)


def test_select_kstar_allowed_mask_shape_validated():
    conf = torch.tensor([[0.9, 0.8, 0.5]])
    cost = torch.tensor([1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="allowed_mask"):
        select_kstar(conf, cost, allowed_mask=torch.tensor([True, False]))


def test_reduce_batch_kstar_stays_in_allowed_set():
    # max/min/median of per-request kstar values that all lie in an allowed
    # set return an element of the tensor, so the batch-uniform m stays in
    # the set (torch.median returns the lower middle element, not a mean).
    kstar = torch.tensor([1, 4, 4, 8])
    assert reduce_batch_kstar(kstar, "max") == 8
    assert reduce_batch_kstar(kstar, "min") == 1
    assert reduce_batch_kstar(kstar, "median") == 4

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVICT verification-length selection.

Given the draft model's per-position confidence for a linear draft chain, choose
``m*`` — how many of the ``K`` drafted tokens to verify — by maximizing the
utility ``U(m) = E[A(m)] / C(m)``.

This module depends only on ``torch`` so it can be unit-tested in isolation and,
in the full version, fused into a captured CUDA-graph region. All operations are
vectorized over the batch with no Python loop over ``m``.
"""

import torch


def gather_draft_confidence(
    # [B, K, V]
    draft_probs: torch.Tensor,
    # [B, K]
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Probability the draft model assigned to each token it chose.

    Args:
        draft_probs: Per-position draft distributions, shape ``[B, K, V]``.
        draft_token_ids: The drafted token ids, shape ``[B, K]``.

    Returns:
        ``q`` of shape ``[B, K]`` with ``q[b, k] = draft_probs[b, k,
        draft_token_ids[b, k]]``, clamped to ``[0, 1]``.
    """
    if draft_probs.dim() != 3:
        raise ValueError(
            f"draft_probs must be [B, K, V], got {tuple(draft_probs.shape)}"
        )
    if draft_token_ids.shape != draft_probs.shape[:2]:
        raise ValueError(
            "draft_token_ids must be [B, K] matching draft_probs[:2], got "
            f"{tuple(draft_token_ids.shape)} vs {tuple(draft_probs.shape[:2])}"
        )
    idx = draft_token_ids.to(torch.int64).unsqueeze(-1)
    q = torch.gather(draft_probs, dim=-1, index=idx).squeeze(-1)
    return q.clamp_(0.0, 1.0)


def expected_accepted_length(
    # [B, K]
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Estimated accepted length ``E[A(m)]`` for each prefix length ``m``.

    ``score[b, k] = prod_{j=0}^{k} q[b, j]`` (cumulative product along the chain)
    is the estimated probability that the chain is accepted through position
    ``k``. ``E[A(m)] = sum_{j=0}^{m-1} score[b, j]`` (the sum of the first ``m``
    cumulative products) is its prefix sum, i.e. ``ehat[b, m-1]``.

    Args:
        confidence: ``q`` of shape ``[B, K]`` in ``[0, 1]``.

    Returns:
        ``ehat`` of shape ``[B, K]`` where ``ehat[b, m - 1] = E[A(m)]`` for
        ``m in 1..K``.
    """
    if confidence.dim() != 2:
        raise ValueError(f"confidence must be [B, K], got {tuple(confidence.shape)}")
    score = torch.cumprod(confidence.to(torch.float32), dim=1)
    return torch.cumsum(score, dim=1)


def select_kstar(
    # [B, K]
    confidence: torch.Tensor,
    # [K]
    cost_per_m: torch.Tensor,
    min_k: int = 1,
    # [K] bool; True at index m-1 iff length m may be chosen.
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-request optimal verification length ``m*``.

    ``m* = argmax_{m in [min_k, K]} E[A(m)] / C(m)``, optionally restricted to an
    allowed set of lengths (quantized m*): with ``allowed_mask`` given, the
    argmax runs only over lengths whose mask entry is True, so the choice is
    cost-optimal within the set rather than a post-hoc rounding. Used to keep
    the set of verify lengths small enough to capture a FULL CUDA graph per
    length.

    Args:
        confidence: ``q`` of shape ``[B, K]`` in ``[0, 1]``.
        cost_per_m: ``C(m)`` for ``m in 1..K``, shape ``[K]``, strictly positive.
        min_k: Floor on the chosen length (>= 1); guarantees at least ``min_k``
            tokens are verified so EVICT never collapses speculation entirely.
        allowed_mask: Optional bool mask of shape ``[K]``; must have at least one
            True entry at an index ``>= min_k - 1``.

    Returns:
        ``kstar`` of shape ``[B]`` (int64) with values in ``[min_k, K]``.
    """
    if confidence.dim() != 2:
        raise ValueError(f"confidence must be [B, K], got {tuple(confidence.shape)}")
    _, num_spec = confidence.shape
    if cost_per_m.shape != (num_spec,):
        raise ValueError(
            f"cost_per_m must be [K]={num_spec}, got {tuple(cost_per_m.shape)}"
        )
    if num_spec == 0:
        return torch.zeros(
            confidence.shape[0], dtype=torch.int64, device=confidence.device
        )

    min_k = max(1, min(min_k, num_spec))

    ehat = expected_accepted_length(confidence)
    cost = cost_per_m.to(device=ehat.device, dtype=ehat.dtype).clamp_min(1e-9)
    utility = ehat / cost.view(1, num_spec)

    # Restrict the argmax to m >= min_k (and to the allowed set, if given) by
    # masking out disallowed prefixes.
    if min_k > 1 or allowed_mask is not None:
        utility = utility.clone()
        if min_k > 1:
            utility[:, : min_k - 1] = float("-inf")
        if allowed_mask is not None:
            if allowed_mask.shape != (num_spec,):
                raise ValueError(
                    f"allowed_mask must be [K]={num_spec}, got "
                    f"{tuple(allowed_mask.shape)}"
                )
            if not bool(allowed_mask[min_k - 1 :].any()):
                raise ValueError(
                    f"allowed_mask has no selectable length >= min_k={min_k}"
                )
            utility.masked_fill_(
                ~allowed_mask.to(device=utility.device).view(1, num_spec),
                float("-inf"),
            )

    kstar = utility.argmax(dim=1) + 1  # argmax index m-1 -> m
    return kstar.to(torch.int64).clamp_(min_k, num_spec)


def reduce_batch_kstar(
    # [B]
    kstar: torch.Tensor,
    policy: str = "max",
) -> int:
    """Reduce per-request ``m*`` to a single batch-uniform length.

    The MVP truncates the draft tensor uniformly (preserving the uniform
    per-request stride the async path assumes). For ``B == 1`` this is exact.

    Args:
        kstar: Per-request ``m*``, shape ``[B]``.
        policy: ``"max"`` (verify the deepest any request wants — conservative on
            accepted length), ``"min"`` (most aggressive truncation), or
            ``"median"``.

    Returns:
        The batch-uniform ``m*`` as a Python int.
    """
    if kstar.numel() == 0:
        return 0
    if policy == "max":
        return int(kstar.max().item())
    if policy == "min":
        return int(kstar.min().item())
    if policy == "median":
        return int(kstar.median().item())
    raise ValueError(f"unknown reduce policy {policy!r}")

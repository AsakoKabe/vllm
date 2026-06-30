# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVICT: cost-aware adaptive verification for speculative decoding.

EVICT chooses, per speculative step, how many of the K already-drafted chain
tokens are worth verifying by the target model. It maximizes a utility
``U(m) = E[A(m)] / C(m)`` where ``E[A(m)]`` is the estimated accepted length when
verifying the first ``m`` draft tokens and ``C(m)`` is the profiled cost of a step
that verifies ``m`` tokens. Verifying fewer tokens shrinks the target forward and,
for an MoE target, the union of activated experts.

See ``EVICT_PLAN.md`` and arXiv:2605.00342.
"""

from vllm.v1.spec_decode.evict.cost_table import CostTable
from vllm.v1.spec_decode.evict.selector import (
    expected_accepted_length,
    gather_draft_confidence,
    reduce_batch_kstar,
    select_kstar,
)

__all__ = [
    "CostTable",
    "expected_accepted_length",
    "gather_draft_confidence",
    "reduce_batch_kstar",
    "select_kstar",
]

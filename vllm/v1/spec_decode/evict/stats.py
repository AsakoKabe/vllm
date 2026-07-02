# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step EVICT truncation stats, carried worker -> scheduler.

Populated in the worker when EVICT trims the drafted chain to its cost-effective
prefix m*, carried on ``ModelRunnerOutput`` and folded into ``SpecDecodingStats``
once per step. Plain ints only, so it pickles across the worker IPC boundary and
is unit-testable without torch.
"""

from dataclasses import dataclass


@dataclass
class EvictStats:
    """One speculative step's EVICT verification-length decision."""

    # Verified prefix length m* chosen this step (batch-uniform in the MVP).
    kstar: int
    # Drafted chain length K (num_speculative_tokens actually drafted).
    num_spec: int
    # Requests in the batch (B).
    num_reqs: int
    # Verify positions the next target forward skips: (K - m*) * B.
    saved_positions: int

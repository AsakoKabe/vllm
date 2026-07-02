# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVICT cost model ``C(m)``.

``C(m)`` is the latency of a speculative step that verifies ``m`` draft tokens.
The draft chain is fully generated before truncation, so EVICT saves only the
target-forward + verify portion; ``C(m)`` models exactly that and is profiled
offline (see ``build_cost_table.py``). When no profiled table is supplied an
affine fallback ``C(m) = intercept + per_token * m`` is used for experimentation.

Only ``torch`` and the stdlib are imported so the table can be unit-tested in
isolation.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import torch


class CostTable:
    """A monotone-ish, strictly-positive cost ``C(m)`` for ``m >= 1``.

    Build with :meth:`from_file` (profiled, paper-faithful) or :meth:`affine`
    (heuristic fallback). Materialize a per-length tensor with :meth:`as_tensor`.
    """

    def __init__(
        self,
        costs: Mapping[int, float] | None = None,
        *,
        intercept: float | None = None,
        per_token: float | None = None,
        source_model: str | None = None,
        source_method: str | None = None,
    ) -> None:
        # Provenance recorded by the profiler (None for affine/hand-written
        # tables); checked against the deployed config via validate_source().
        self.source_model = source_model
        self.source_method = source_method
        if costs is not None:
            self._validate_costs(costs)
            self._costs: dict[int, float] | None = dict(costs)
            self._intercept = None
            self._per_token = None
        else:
            if intercept is None or per_token is None:
                raise ValueError(
                    "CostTable requires either explicit costs or "
                    "(intercept, per_token) for the affine fallback."
                )
            if intercept <= 0.0 or per_token < 0.0:
                raise ValueError(
                    "affine cost requires intercept > 0 and per_token >= 0, got "
                    f"intercept={intercept}, per_token={per_token}"
                )
            self._costs = None
            self._intercept = float(intercept)
            self._per_token = float(per_token)

    @property
    def is_affine(self) -> bool:
        return self._costs is None

    @classmethod
    def affine(cls, intercept: float = 1.0, per_token: float = 0.5) -> "CostTable":
        """Heuristic ``C(m) = intercept + per_token * m``.

        Only the ratio ``per_token / intercept`` affects ``argmax_m U(m)``.
        """
        return cls(intercept=intercept, per_token=per_token)

    @classmethod
    def from_file(cls, path: str | Path) -> "CostTable":
        """Load a profiled ``C(m)`` table.

        Accepts JSON shaped either as ``{"1": ms, "2": ms, ...}`` or
        ``{"unit": "ms", "costs": {"1": ms, ...}}``. Keys are verification
        lengths ``m`` (>= 1); values are step latencies (any positive unit —
        only relative shape matters). Fails closed on missing/non-positive
        entries or non-contiguous lengths.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"EVICT cost table not found: {p}")
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"EVICT cost table {p} is not valid JSON: {exc}") from exc

        source_model: str | None = None
        source_method: str | None = None
        if isinstance(raw, Mapping) and "costs" in raw:
            model = raw.get("model")
            method = raw.get("method")
            source_model = model if isinstance(model, str) else None
            source_method = method if isinstance(method, str) else None
            raw = raw["costs"]
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"EVICT cost table {p} must be a JSON object mapping m -> cost."
            )

        costs: dict[int, float] = {}
        for key, value in raw.items():
            try:
                m = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"EVICT cost table {p} has non-integer length key {key!r}."
                ) from exc
            costs[m] = float(value)

        cls._validate_costs(costs, source=str(p))
        return cls(costs=costs, source_model=source_model, source_method=source_method)

    def validate_source(
        self,
        expected_model: str | None,
        expected_method: str | None,
        source: str = "<cost table>",
    ) -> None:
        """Fail closed if the table was profiled for a different model/method.

        A stale table is otherwise silently reused whenever its coverage
        suffices, corrupting every U(m) = E[A(m)] / C(m) selection with costs
        from the wrong deployment. Tables without provenance (hand-written or
        pre-provenance) are accepted as-is.
        """
        if (
            self.source_model is not None
            and expected_model is not None
            and self.source_model != expected_model
        ):
            raise ValueError(
                f"EVICT cost table {source} was profiled for model "
                f"{self.source_model!r} but the engine is running "
                f"{expected_model!r}; re-profile the table (or edit its "
                "'model' field if the reuse is intentional)."
            )
        if (
            self.source_method is not None
            and expected_method is not None
            and self.source_method != expected_method
        ):
            raise ValueError(
                f"EVICT cost table {source} was profiled with method "
                f"{self.source_method!r} but the engine uses "
                f"{expected_method!r}; re-profile the table."
            )

    @staticmethod
    def _validate_costs(costs: Mapping[int, float], source: str = "<inline>") -> None:
        if not costs:
            raise ValueError(f"EVICT cost table {source} is empty.")
        lengths = sorted(costs)
        if lengths[0] != 1:
            raise ValueError(
                f"EVICT cost table {source} must start at m=1, got m={lengths[0]}."
            )
        if lengths != list(range(1, lengths[-1] + 1)):
            raise ValueError(
                f"EVICT cost table {source} lengths must be contiguous 1..M, "
                f"got {lengths}."
            )
        for m in lengths:
            if not (costs[m] > 0.0):
                raise ValueError(
                    f"EVICT cost table {source} has non-positive cost {costs[m]} "
                    f"at m={m}."
                )

    @property
    def max_profiled_m(self) -> int | None:
        """Largest ``m`` covered by a profiled table (None for affine)."""
        if self._costs is None:
            return None
        return max(self._costs)

    def cost(self, m: int) -> float:
        """``C(m)`` for a single length ``m >= 1``."""
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if self._costs is not None:
            if m not in self._costs:
                raise KeyError(
                    f"EVICT cost table does not cover m={m} "
                    f"(profiled up to {self.max_profiled_m})."
                )
            return self._costs[m]
        assert self._intercept is not None and self._per_token is not None
        return self._intercept + self._per_token * m

    def as_tensor(
        self,
        max_k: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Materialize ``C(m)`` for ``m in 1..max_k`` as a ``[max_k]`` tensor.

        Profiled tables must cover ``1..max_k`` (fails closed otherwise);
        the affine fallback covers any ``max_k``.
        """
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1, got {max_k}")
        if self._costs is not None:
            covered = self.max_profiled_m or 0
            if max_k > covered:
                raise ValueError(
                    f"EVICT cost table covers m up to {covered} but max_k={max_k} "
                    "was requested; profile a deeper table or lower "
                    "num_speculative_tokens."
                )
        values = [self.cost(m) for m in range(1, max_k + 1)]
        return torch.tensor(values, dtype=dtype, device=device)

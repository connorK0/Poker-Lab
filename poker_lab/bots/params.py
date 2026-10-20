"""Interpretable bot parameters (the continuous ground truth regressed in M4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(frozen=True, slots=True)
class BotParams:
    """Tendencies of a parameterized bot. Probabilities are in [0, 1].

    range_width         Fraction of starting hands (by preflop strength) played in an
                        unraised pot. Drives VPIP.
    raise_share         Probability of entering an unraised pot with a raise rather than a
                        limp. Drives PFR / VPIP.
    three_bet_freq      Facing a single raise with a continuing hand: probability of
                        re-raising rather than calling.
    continue_vs_raise   Facing a raise, the bot continues with the top
                        ``range_width * continue_vs_raise`` of hands. High = sticky.
    cbet_freq           As preflop aggressor, probability of betting the flop when no one has
                        bet yet.
    fold_to_aggression  Probability of folding a marginal (weak/draw) hand facing a postflop
                        bet. Medium hands fold less, air folds more.
    bluff_freq          Probability of betting/raising with air or weak hands when
                        aggression is an option.
    value_aggression    Probability of betting/raising (rather than check/call) with strong
                        hands. Medium hands bet at half this rate.
    bet_size_pot        Postflop bet size as a fraction of the pot. Also scales preflop open
                        sizes and raise sizes.
    """

    range_width: float
    raise_share: float
    three_bet_freq: float
    continue_vs_raise: float
    cbet_freq: float
    fold_to_aggression: float
    bluff_freq: float
    value_aggression: float
    bet_size_pot: float

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not np.isfinite(v):
                raise ValueError(f"{f.name} must be a finite number, got {v!r}")
            lo, hi = PARAM_BOUNDS[f.name]
            if not lo <= v <= hi:
                raise ValueError(f"{f.name}={v} outside [{lo}, {hi}]")
            object.__setattr__(self, f.name, float(v))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BotParams:
        return cls(**{n: float(d[n]) for n in PARAM_NAMES})


PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "range_width": (0.01, 1.0),
    "raise_share": (0.0, 1.0),
    "three_bet_freq": (0.0, 1.0),
    "continue_vs_raise": (0.05, 1.0),
    "cbet_freq": (0.0, 1.0),
    "fold_to_aggression": (0.0, 1.0),
    "bluff_freq": (0.0, 1.0),
    "value_aggression": (0.0, 1.0),
    "bet_size_pot": (0.2, 2.5),
}
PARAM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(BotParams))

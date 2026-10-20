"""Hand-history schema: the contract between all phases.

* ``hand_history``: canonical data model (``HandHistory`` and its parts).
* ``views``: firewalled views of a hand in progress (``PublicHandState``, ``SeatView``).
* ``store``: ``HandHistoryStore``, which keeps hole cards sealed until a hand is complete.
* ``serialization``: strict JSON Lines encoding.
* ``validation``: deep consistency checks (chip conservation, betting sequence, cards).
"""

from poker_lab.schema.cards import (
    FULL_DECK,
    RANKS,
    SUITS,
    card_to_id,
    id_to_card,
    is_valid_card,
    parse_cards,
)
from poker_lab.schema.hand_history import (
    SCHEMA_VERSION,
    STREETS,
    Action,
    ActionType,
    CardObservation,
    CardSource,
    ChipAward,
    HandHistory,
    HandStatus,
    HoleCards,
    SeatInfo,
    ShowdownEntry,
    Street,
    TableConfig,
)
from poker_lab.schema.positions import assign_positions, position_names, seat_order_from_button
from poker_lab.schema.serialization import (
    dumps_hand,
    hand_from_dict,
    hand_to_dict,
    loads_hand,
    read_jsonl,
    write_jsonl,
)
from poker_lab.schema.store import (
    HandHistoryStore,
    HandStoreError,
    SealedHoleCardsError,
    SeatKey,
)
from poker_lab.schema.validation import (
    HandHistoryValidationError,
    compute_uncalled_returns,
    find_problems,
    validate_hand_history,
)
from poker_lab.schema.views import LegalActions, PublicHandState, SeatView

__all__ = [
    "FULL_DECK",
    "RANKS",
    "SCHEMA_VERSION",
    "STREETS",
    "SUITS",
    "Action",
    "ActionType",
    "CardObservation",
    "CardSource",
    "ChipAward",
    "HandHistory",
    "HandHistoryStore",
    "HandHistoryValidationError",
    "HandStatus",
    "HandStoreError",
    "HoleCards",
    "LegalActions",
    "PublicHandState",
    "SealedHoleCardsError",
    "SeatInfo",
    "SeatKey",
    "SeatView",
    "ShowdownEntry",
    "Street",
    "TableConfig",
    "assign_positions",
    "card_to_id",
    "compute_uncalled_returns",
    "dumps_hand",
    "find_problems",
    "hand_from_dict",
    "hand_to_dict",
    "id_to_card",
    "is_valid_card",
    "loads_hand",
    "parse_cards",
    "position_names",
    "read_jsonl",
    "seat_order_from_button",
    "validate_hand_history",
    "write_jsonl",
]

import pytest
from phevaluator.card import Card as PhCard

from poker_lab.schema.cards import (
    FULL_DECK,
    card_rank_index,
    card_suit_index,
    card_to_id,
    id_to_card,
    normalize_card,
    parse_cards,
    validate_card,
)


def test_full_deck_is_52_unique_cards():
    assert len(FULL_DECK) == 52
    assert len(set(FULL_DECK)) == 52


def test_card_ids_match_phevaluator_encoding():
    for card in FULL_DECK:
        assert card_to_id(card) == PhCard(card).id_
        assert id_to_card(card_to_id(card)) == card
        assert card_to_id(card) == card_rank_index(card) * 4 + card_suit_index(card)


@pytest.mark.parametrize("bad", ["", "A", "1s", "Ax", "as", "AS", "Ass", None, 5])
def test_validate_card_rejects_noncanonical(bad):
    with pytest.raises(ValueError, match="invalid card"):
        validate_card(bad)


def test_normalize_and_parse():
    assert normalize_card("aS") == "As"
    assert normalize_card("10h") == "Th"
    assert parse_cards("AsKd") == ("As", "Kd")
    assert parse_cards("As, kd 10c") == ("As", "Kd", "Tc")
    with pytest.raises(ValueError, match="cannot parse"):
        parse_cards("AsK")
    with pytest.raises(ValueError, match="out of range"):
        id_to_card(52)

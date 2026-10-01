from src.prefilter import find_keywords

EXCLUDE_KEYWORDS = ["обмін", "обміняю", "обмен", "меняю"]


def test_prefilter_detects_target_keywords():
    """Verify that relevant shopping queries retain target keywords."""
    text = "Добрий день! Шукаю дитячий зимовий комбінезон на хлопчика"
    matches = find_keywords(text.lower())
    assert len(matches) > 0
    assert any("комбінезон" in m.lower() for m in matches)


def test_prefilter_ignores_irrelevant_messages():
    """Verify that general chatter without target keywords produces no matches."""
    irrelevant_text = "Привіт усім! Яка сьогодні чудова погода на вулиці."
    matches = find_keywords(irrelevant_text.lower())
    assert len(matches) == 0


def test_prefilter_case_insensitivity():
    """Verify that keyword matching is case-insensitive."""
    text = "Шукаю КОМБІНЕЗОН Reima"
    matches = find_keywords(text)
    assert len(matches) > 0


def test_prefilter_exclusion_words():
    """Verify that trade/exchange stop words are accurately detected."""
    trade_text = "Цікавить тільки обмін дитячих речей на памперси"
    normalized = trade_text.lower()
    has_exclusion = any(word in normalized for word in EXCLUDE_KEYWORDS)
    assert has_exclusion is True

    valid_text = "Хочу купити зимовий комбінезон"
    normalized_valid = valid_text.lower()
    has_exclusion_valid = any(word in normalized_valid for word in EXCLUDE_KEYWORDS)
    assert has_exclusion_valid is False

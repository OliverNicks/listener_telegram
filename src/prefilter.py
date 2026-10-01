from pathlib import Path

from flashtext import KeywordProcessor

processor = KeywordProcessor(case_sensitive=False)

# Resolve keywords path relative to project root
KEYWORDS_FILE = Path(__file__).resolve().parents[1] / "config" / "keywords.txt"

# Load all pre-generated keywords
with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:
            processor.add_keyword(word)


def find_keywords(text: str) -> list[str]:
    """Scans text in O(L) time and returns all matched shopping keywords."""
    return processor.extract_keywords(text)

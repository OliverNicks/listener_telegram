from flashtext import KeywordProcessor
# from prefilter import find_keywords

processor = KeywordProcessor(case_sensitive=False)

# Load all pre-generated keywords
with open(r"config\keywords.txt", "r", encoding="utf-8") as f:
  for line in f:
    word = line.strip()
    if word:
      processor.add_keyword(word)


def find_keywords(text: str) -> list[str]:
  """Scans text in O(L) time and returns all matched shopping keywords."""
  return processor.extract_keywords(text)


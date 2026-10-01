import pymorphy3

morph_uk = pymorphy3.MorphAnalyzer(lang="uk")
morph_ru = pymorphy3.MorphAnalyzer(lang="ru")


def expand_words(words, morph):
  expanded = set()
  for word in words:
    parses = morph.parse(word)
    if parses:
      for form in parses[0].lexeme:
        expanded.add(form.word.lower())
    else:
      expanded.add(word.lower())
  return expanded


"""uk_lemmas = [
    # Intent
    "купити",
    "шукати",
    "замовити",
    "розмір",
    "ціна",
    # Clothes & Gear
    "коляска",
    "прогулянковий",
    "муфта",
    "конверт",
    "козирок",
    "автокрісло",
    "велосипед",
    "біговел",
    "самокат",
    "бодік",
    "комбінезон",
    "одяг",
    "взуття",
]

ru_lemmas = [
    # Intent
    "купить",
    "искать",
    "заказать",
    "размер",
    "цена",
    # Clothes & Gear
    "комбинезон",
    "полукомбинезон",
    "штаны",
    "рукавицы",
    "краги",
    "пинетки",
    "комплект",
    "шапка",
    "поддева",
    "термобелье",
    "куртка",
    "манишка",
    "шарф",
    "пальто",
    "плащ",
    "шлем",
    "одежда",
    "сапоги",
    "босоножки",
    "сандалии",
    "ботинки",
    "кроссовки",
    "тапки",
    "туфли",
    "коляска",
    "детский",
    "цельный",
    "флисовый",
    "летний",
    "повседневный",
    "резиновый",
]"""

uk_lemmas_to_add = [
    # Fitting & Trying On
    "поміряти",
    "приміряти",
    "примірка",
    # Additional Intent & Inquiries
    "придбати",
    "підібрати",
    "наявність",
    "замір",
    "вимір",
    "зріст",
    "ріст",
    "великомірити",
    "маломірити",
    # Clothes & Gear (missing compared to RU list + essentials)
    "штани",
    "штаники",
    "штанці",
    "напівкомбінезон",
    "куртка",
    "курточка",
    "костюм",
    "костюмчик",
    "боді",
    "чоловічок",
    "сліп",
    "піжама",
    "шапка",
    "шапочка",
    "шолом",
    "шарф",
    "манішка",
    "пальто",
    "плащ",
    "рукавиці",
    "краги",
    "піддіва",
    "термобілизна",
    "пінетки",
    "комплект",
    "черевики",
    "ботінки",
    "чобітки",
    "сапожки",
    "босоніжки",
    "сандалі",
    "кросівки",
    "тапочки",
    "туфлі",
    # Descriptors & Attributes
    "дитячий",
    "цільний",
    "суцільний",
    "роздільний",
    "флісовий",
    "зимовий",
    "демісезонний",
    "літній",
    "гумовий",
]

ru_lemmas_to_add = [
    # Fitting & Trying On
    "померить",
    "примерить",
    "примерка",
    # Additional Intent & Inquiries
    "подобрать",
    "наличие",
    "замер",
    "рост",
    "ростовка",
    "большемерить",
    "маломерить",
    # Clothes & Gear (missing items)
    "бодик",
    "боди",
    "человечек",
    "слип",
    "пижама",
    "костюм",
    "костюмчик",
    "штанишки",
    "автокресло",
    "велосипед",
    "беговел",
    "самокат",
    "конверт",
    "муфта",
    # Descriptors & Attributes
    "зимний",
    "демисезонный",
    "раздельный",
]
# Expand and combine into a single set
all_keywords = expand_words(uk_lemmas_to_add, morph_uk) | expand_words(
    ru_lemmas_to_add, morph_ru
)

# Filter out very short tokens (e.g. 1-2 letter fragments) to prevent false positives
clean_keywords = {w for w in all_keywords if len(w) > 2}

with open("keywords_to_add.txt", "w", encoding="utf-8") as f:
  f.writelines(f"{word}\n" for word in sorted(clean_keywords))

print(f"Generated {len(clean_keywords)} unique keyword forms into keywords.txt")

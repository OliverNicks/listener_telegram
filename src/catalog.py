import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

load_dotenv()

DEFAULT_SITE_URL = "https://best4baby.com.ua"

# Cohere embed-multilingual-v3.0 with 'ip' space:
# Values typically range from 0.0 to 1.0. A score >= 0.50 - 0.60 indicates a strong match.
SIMILARITY_THRESHOLD = 0.40

def clean_markdown_url(raw_url: str) -> str:
  """Extracts a clean URL by stripping Markdown link formatting or surrounding brackets."""
  if not raw_url:
    return ""

  raw_url = raw_url.strip()

  # Handles markdown formats like [label](https://...) or [https://...](https://...)
  if "](" in raw_url:
    return raw_url.split("](")[-1].rstrip(")")

  # Handles bracketed formats like [https://...]
  if raw_url.startswith("[") and raw_url.endswith("]"):
    return raw_url[1:-1].strip()

  return raw_url

class CategoryCatalog:

    def __init__(
        self,
        persist_dir: str = "./chroma_catalog_db",
        collection_name: str = "best4baby_categories",
    ):
        """Loads the persisted Chroma collection with the matching Cohere embedding model."""
        self.embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "ip"},
        )

    def resolve_url(
          self, query_text: str, min_score: float = SIMILARITY_THRESHOLD) -> str:
        """Finds the most relevant subcategory URL using similarity_search_with_relevance_scores.

        Returns the specific page_url if score >= min_score, else the root site
        URL.
        """
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
              query=query_text, k=1
            )

            if not results:
                print("⚠️ No matches found in Chroma. Returning default site URL.")
                return DEFAULT_SITE_URL

            doc, score = results[0]
            subcategory = doc.metadata.get("subcategory_name", "Unknown")
            raw_url = doc.metadata.get("page_url", DEFAULT_SITE_URL)
            clean_url = clean_markdown_url(raw_url)

            print(f"🔍 Chroma Match: {subcategory} | Relevance Score: {score:.4f}")

            if score >= min_score:
              return clean_url
            
            print(
                f"⚠️ Score {score:.4f} is below threshold {min_score}. Falling back"
                f" to {DEFAULT_SITE_URL}."
            )
            return DEFAULT_SITE_URL

        except Exception as e:
            print(
                f"⚠️ Error resolving category URL ({e}). Returning default site URL."
            )
            return DEFAULT_SITE_URL

"""
Test case:

if __name__ == "__main__":
  # Standalone test
  catalog = CategoryCatalog()

  test_queries = [
      "хочу поміряти на дитину зимові штани в 116р де, хто знає",
      "Шукаю теплий суцільний слитник на зиму для малюка",
      "Де можна купити дитячий манеж або ліжечко?",
  ]

  print("\n--- Running Diagnostic Tests ---")
  for q in test_queries:
    url = catalog.resolve_url(q)
    print(f"Query: {q}\nResolved URL: {url}\n" + "-" * 40)"""
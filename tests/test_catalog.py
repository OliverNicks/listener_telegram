from unittest.mock import MagicMock

import chromadb

from src.catalog import DEFAULT_SITE_URL, CategoryCatalog, clean_markdown_url


def test_clean_markdown_url_formats():
    """Verify URL cleaner properly parses markdown, bracketed, and plain URLs."""
    # Markdown link
    assert clean_markdown_url("[Комбінезони](https://best4baby.com.ua/cat)") == "https://best4baby.com.ua/cat"
    # Bracketed URL
    assert clean_markdown_url("[https://best4baby.com.ua/cat]") == "https://best4baby.com.ua/cat"
    # Plain URL
    assert clean_markdown_url("https://best4baby.com.ua/cat") == "https://best4baby.com.ua/cat"
    # Empty string
    assert clean_markdown_url("") == ""


def test_chromadb_connection_and_query():
    """Verify connecting to ChromaDB, creating a collection, and querying documents with embeddings."""
    # Ephemeral in-memory ChromaDB client for isolated unit testing
    client = chromadb.Client()
    collection = client.create_collection(
        name="test_categories",
        metadata={"hnsw:space": "cosine"}
    )

    # Add sample catalog entries with explicit embeddings
    collection.add(
        ids=["cat_1", "cat_2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=[
            "Дитячі зимові комбінезони Reima Lenne теплі водонепроникні",
            "Дитячі прогулянкові коляски та автокрісла"
        ],
        metadatas=[
            {"page_url": "https://best4baby.com.ua/zymovi-kombinezony", "name": "Комбінезони"},
            {"page_url": "https://best4baby.com.ua/koliasky", "name": "Коляски"}
        ]
    )

    assert collection.count() == 2

    # Query with a vector closest to cat_1
    results = collection.query(
        query_embeddings=[[0.95, 0.05, 0.0]],
        n_results=1
    )

    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "cat_1"
    assert results["metadatas"][0][0]["page_url"] == "https://best4baby.com.ua/zymovi-kombinezony"
    assert results["metadatas"][0][0]["name"] == "Комбінезони"


def test_category_catalog_fallback_url():
    """Verify that CategoryCatalog returns DEFAULT_SITE_URL on low similarity score."""
    catalog = object.__new__(CategoryCatalog)
    mock_vectorstore = MagicMock()

    # Simulate query with score below threshold (e.g. 0.20 < 0.40)
    mock_doc = MagicMock()
    mock_doc.metadata = {"page_url": "https://best4baby.com.ua/random", "subcategory_name": "Random"}
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.20)]

    catalog.vectorstore = mock_vectorstore
    resolved_url = catalog.resolve_url("невідповідний запит", min_score=0.40)

    assert resolved_url == DEFAULT_SITE_URL

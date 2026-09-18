import pandas as pd
from langchain_core.documents import Document
import os
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_catalog_documents(csv_path: str) -> tuple[list[Document], list[str]]:
    df = pd.read_csv(csv_path)
    documents = []
    ids = []

    for _, row in df.iterrows():
        content = (
            f"Категорія: {row['parent_category']}\n"
            f"Підкатегорія: {row['subcategory_name']}\n"
            f"Пошукові запити та синоніми: {row['search_terms']}"
        )

        metadata = {
            "id": str(row["id"]),
            "parent_category": str(row["parent_category"]),
            "subcategory_name": str(row["subcategory_name"]),
            "page_url": str(row["page_url"]),
        }

        documents.append(Document(page_content=content.strip(), metadata=metadata))
        ids.append(str(row["id"]))

    return documents, ids

def build_vector_store():
    csv_file = r"data\best4baby_database_link.csv"
    persist_dir = "./chroma_catalog_db"

    print("Converting CSV rows to documents...")
    docs, doc_ids = load_catalog_documents(csv_file)

    print("Initializing Cohere embeddings...")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    print(f"Embedding {len(docs)} categories into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        ids=doc_ids,
        embedding=embeddings,
        collection_name="best4baby_categories",
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "ip"}
    )

    print(f"Vector store successfully saved to '{persist_dir}'.")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
# Telegram Listener & Fast Intent Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime%20(CPU)-green.svg)](https://onnxruntime.ai/)
[![Vector DB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-throughput, lightweight social listening and buyer-intent classification pipeline for Telegram group chats. 

Combines a deterministic sub-millisecond lexical prefilter, a CPU-optimized ONNX bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`), and local ChromaDB vector retrieval to identify commercial purchasing intent in real time and generate context-aware product recommendations.

---

## Key Features & Benchmarks

* **Sub-Millisecond Lexical Gate:** Discards **90%+ of irrelevant chatter** via optimized Ukrainian lexical stem sets before invoking neural networks, keeping CPU overhead minimal.
* **Low-Latency CPU Inference:** Quantized / CPU-optimized ONNX Runtime engine executing mean-pooled sentence embeddings in **< 15 ms**.
* **Calibrated Intent Matching:** Anchor-based cosine similarity matrix combined with domain-specific heuristic lexical bias scoring.
* **Catalog RAG Pipeline:** Semantic retrieval via a local ChromaDB vector index, mapping unstructured buyer requests directly to SKU deep links and catalog availability.

---

## Architecture Overview

                        [ Telegram Public / Group Chats ]
                                        │
                                        ▼ (Telethon MTProto Client)
                            [ bots/bot_b/listener.py ]
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
             [ Bot A Seeder ]                    [ Community Messages ]
        (Organic Discussion Seeds)                        │
                                                          ▼
                                                [ src/prefilter.py ]
                                            (Fast Lexical / Keyword Gate)
                                                          │ (Pass)
                                                          ▼
                                                [ src/classifier.py ]
                                        ┌───────────────────────────────┐
                                        │ 1. Tokenize (HF Fast)         │
                                        │ 2. ONNX Inference Engine      │
                                        │ 3. Mean Pooling + L2 Norm     │
                                        │ 4. Matrix Dot Product Cosine  │
                                        │ 5. Lexical Calibration Bias   │
                                        └───────────────────────────────┘
                                                          │ (Intent == Buyer)
                                                          ▼
                                                [ src/catalog.py ]
                                        (ChromaDB Semantic Product Match)
                                                          │
                                                          ▼
                                                [ src/drafter.py ]
                                        (Context-Aware Reply Synthesis)
                                                          │
                                                          ▼
                                            [ Organic Reply Published ]

---

## Directory Layout

```text
help_project/
├── bots/
│   ├── bot_a/               # Question seeding module
│   ├── bot_b/listener.py    # Main asynchronous listener daemon
│   └── auth/login.py        # Telethon MTProto session generator
├── config/                  # Lexical filters & keyword dictionaries
├── data/                    # Product catalogs (CSV) & vector store files
├── models/                  # ONNX weights & tokenizer configurations
├── scripts/
│   ├── download_model.py    # Model weights & tokenizer asset fetcher
│   └── ingest.py            # Vector database ingestion pipeline
├── src/                     # Core classifier, embedding, and retriever modules
└── tests/                   # Unit, integration, and latency benchmark tests

```
---

## Quickstart 

## Option A: Local Installation

### 1. Clone & Set Up Environment:

```bash
git clone [https://github.com/OliverNicks/listener_telegram.git](https://github.com/OliverNicks/listener_telegram.git)
cd listener_telegram

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
# .\venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Weights & Ingest Catalog:

```bash
python scripts/download_model.py
python scripts/ingest.py
```

### 3. Configure Credentials & Authenticate:

```bash
cp .env.example .env
# Edit .env with your TELEGRAM_API_ID, TELEGRAM_API_HASH, and related secrets

python bots/auth/login.py
```

### 4. Run Tests & Start Service:
```bash
pytest tests/ -v
python -m bots.bot_b.listener
```

## Option B: Docker Setup

```bash
cp .env.example .env
docker compose build
docker compose up -d
```
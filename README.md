# Telegram Listener & Fast Intent Classifier

[![CI](https://github.com/OliverNicks/listener_telegram/actions/workflows/ci.yml/badge.svg)](https://github.com/OliverNicks/listener_telegram/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime%20(CPU)-green.svg)](https://onnxruntime.ai/)
[![Vector DB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-throughput, lightweight social listening and buyer-intent classification pipeline for Telegram group chats. 

Combines a deterministic sub-millisecond lexical prefilter, a CPU-optimized ONNX bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`), and local ChromaDB vector retrieval to identify commercial purchasing intent in real time and generate context-aware product recommendations.

---

## Tech Stack & Architecture

* **Runtime & Language:** Python 3.11 (asyncio, Telethon MTProto client).
* **NLP & Inference:** ONNX Runtime (quantized CPU bi-encoder), FlashText lexical pre-filter.
* **Vector Store & Retrieval:** ChromaDB, Cohere multilingual embeddings.
* **LLM Reply Synthesis:** OpenAI API (GPT-4o-mini structured outputs).
* **Logging & Observability:** Loguru (colored developer console + daily rotating JSON-lines logs for latency metrics).
* **DevOps & Containerization:** Multi-stage Docker build, non-root system user (`appuser`), Docker Compose with persistent volume bindings.
* **CI/CD Quality Assurance:** GitHub Actions pipeline running Ruff linter & Pytest test suite on every push/PR.

---

## Key Features & Benchmarks

* **Sub-Millisecond Lexical Gate:** Discards **90%+ of irrelevant chatter** via optimized Ukrainian lexical stem sets before invoking neural networks, keeping CPU overhead minimal.
* **Low-Latency CPU Inference:** Quantized / CPU-optimized ONNX Runtime engine executing mean-pooled sentence embeddings in **< 15 ms**.
* **Calibrated Intent Matching:** Anchor-based cosine similarity matrix combined with domain-specific heuristic lexical bias scoring.
* **Catalog RAG Pipeline:** Semantic retrieval via a local ChromaDB vector index, mapping unstructured buyer requests directly to catalog category links.
* **Structured JSON Logging:** High-resolution latency profiling (`latency_ms`) across classification, vector search, and draft generation.
* **Secure Containerization:** Hardened Docker container running as an unprivileged system user (`appuser`), separating stateless logic from stateful vector indexes and sessions.

---

## System Architecture

```text
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
```

---

## Directory Layout

```text
listener_telegram/
├── .github/workflows/       # GitHub Actions CI workflow (Ruff + Pytest)
├── bots/
│   ├── bot_a/               # Question seeding module
│   ├── bot_b/listener.py    # Main asynchronous listener daemon
│   └── auth/login.py        # Telethon MTProto session generator
├── config/                  # Lexical filters & keyword dictionaries
├── data/                    # Product catalogs (CSV) & persistent ChromaDB store
├── models/                  # ONNX weights & tokenizer configurations
├── scripts/
│   ├── download_model.py    # Automated ONNX asset fetcher
│   └── ingest.py            # Vector DB builder & CSV ingestion
├── src/                     # Core classifier, RAG retriever, and logger
│   ├── catalog.py           # ChromaDB vector index resolver
│   ├── classifier.py        # ONNX bi-encoder intent classifier
│   ├── drafter.py           # GPT-4o-mini recommendation synthesizer
│   ├── logger.py            # Centralized Loguru structured logger
│   └── prefilter.py         # Sub-millisecond FlashText keyword gate
├── tests/                   # Pytest test suite (prefilter, embeddings, ChromaDB)
├── Dockerfile               # Multi-stage, non-root production container
├── docker-compose.yml       # Container orchestration with persistent volumes
├── pyproject.toml           # Project metadata, Ruff & Pytest configurations
└── requirements.txt         # Production & development dependencies
```

---

## Quickstart

### Prerequisites
* Python 3.10+
* Telegram API ID & Hash from [my.telegram.org](https://my.telegram.org)
* OpenAI & Cohere API Keys

---

### Option A: Local Installation

#### 1. Clone & Set Up Environment:
```bash
git clone https://github.com/OliverNicks/listener_telegram.git
cd listener_telegram

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Download Weights & Ingest Catalog:
```bash
python scripts/download_model.py
python scripts/ingest.py
```

#### 3. Configure Credentials:
```bash
cp .env.example .env
# Fill in TG_API_ID, TG_API_HASH, OPENAI_API_KEY, COHERE_API_KEY in .env

# Generate initial MTProto session:
python bots/auth/login.py
```

#### 4. Run Tests & Launch Daemon:
```bash
pytest
python -m bots.bot_b.listener
```

---

### Option B: Docker Setup (Recommended for Production)

The service runs inside an unprivileged multi-stage container with persistent volume mounts for the vector database, session credentials, and log files.

```bash
# 1. Build and start service in background
docker compose up -d

# 2. View live structured logs
docker compose logs -f

# 3. Stop the container
docker compose down
```

---

## Testing & Code Quality

The project enforces strict code style and unit test coverage via GitHub Actions on every push and PR:

```bash
# Run code linting & formatting checks:
ruff check .

# Run the unit test suite:
pytest -v
```

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

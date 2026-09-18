# Telegram Listener & Fast Intent Classifier

Lightweight, social listening pipeline for Telegram parent chats. Combines fast lexical prefiltering, an ONNX bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`), and local ChromaDB retrieval to surface commercial buyer inquiries and draft replies.

---

## Key Features

* **Sub-Millisecond Prefilter**: Drops 90%+ irrelevant traffic via Ukrainian lexical stems before invoking the neural net.
* **Low-Latency Inference**: ONNX Runtime engine running mean-pooled sentence embeddings on CPU.
* **Matrix Cosine Matching**: Anchor-based intent matching with heuristic lexical bias calibration.
* **Catalog RAG**: ChromaDB vector index matching user queries against store products and deep links.

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
│   ├── bot_a/               # Question seeder
│   ├── bot_b/listener.py    # Main listener daemon
│   └── auth/login.py        # Telethon session generator
├── config/                  # Keyword dictionaries
├── data/                    # CSV catalogs & vector DB
├── models/                  # ONNX weights & tokenizer config
├── scripts/
│   ├── download_model.py    # Asset fetcher
│   └── ingest.py            # Vector DB builder
├── src/                     # Core classifier & retriever modules
└── tests/                   # Integration and latency tests
```

---

## Quickstart
### 1. Setup Environment

git clone https://github.com/OliverNicks/listener_telegram.git
cd listener_telegram
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt


### 2. Download Model & Ingest Data

python scripts/download_model.py
python scripts/ingest.py

### 3. Authenticate & Run

cp .env.example .env          # Add your Telegram & OpenAI credentials
python bots/auth/login.py      # Authenticate MTProto session
python -m bots.bot_b.listener  # Start listener

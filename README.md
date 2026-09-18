🛰️ Telegram Intent Listener & Lead Generation PipelineAn ultra-low latency, intent-driven inbound lead generation agent for Telegram. This pipeline monitors parent communities, runs high-speed CPU-bound semantic classification using an ONNX-optimized multilingual Transformer, resolves product matching via ChromaDB vector search, and crafts contextual, non-intrusive organic replies using LLMs.⚡ Key Highlights & BenchmarksMetricMeasurement / SpecificationModel Backboneparaphrase-multilingual-MiniLM-L12-v2 (ONNX export)P95 CPU Inference~12–15 ms per message (single-thread INT8 / FP32 ONNX Runtime)Vector Space$384$-dimensional $L_2$-normalized embeddingsSimilarity EngineAnchor-based matrix dot product (np.dot equivalent to Cosine Sim)Pre-Filter Overhead< 0.1 ms regex/lexicon screeningMemory Footprint< 180 MB RSS (Classifier + ONNX runtime session)🏗️ Architecture Overview                         [ Telegram Public / Group Chats ]
                                        │
                                        ▼ (Telethon MTProto Client)
                            [ bots/bot_b/listener.py ]
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
             [ Bot A Seeder ]                     [ Community Messages ]
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
Classification Pipeline MechanicsTokenization: Text is encoded using Hugging Face fast tokenizers into input_ids, attention_mask, and token_type_ids.Mean Pooling: Averages token embeddings across active attention tokens while ignoring padding:$$\mathbf{v}_{\text{pooled}} = \frac{\sum_{i=1}^{L} \mathbf{h}_i \cdot m_i}{\max\left(\sum_{i=1}^{L} m_i, 10^{-9}\right)}$$$L_2$ Normalization: Normalizes vectors to unit length $\Vert{}\mathbf{v}\Vert{}_2 = 1.0$, allowing cosine similarity to be computed as an efficient matrix dot product:$$\text{Sim}(\mathbf{q}, \mathbf{A}) = \mathbf{q} \cdot \mathbf{A}^T$$Anchor Matching & Calibration: Matches queries against cached buyer anchors (e.g., "де купити", "порадьте") and seller anchors (e.g., "продам", "віддам"), applying rule-based score offsets for decisive intent disambiguation.📁 Repository Structurehelp_project/
├── .env.example                      # Template for secrets and credentials
├── .gitignore                        # Git exclusion rules (keeps sessions & DBs secure)
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation
│
├── bots/
│   ├── auth/
│   │   └── login.py                  # Telethon session generator & authentication
│   ├── bot_a/
│   │   ├── question_bot.py           # Discussion starter scheduler
│   │   └── question_generator.py     # AI dilemma / inquiry generator
│   └── bot_b/
│       └── listener.py               # Main monitoring daemon & listener
│
├── config/
│   ├── keywords.txt                  # Monitored lexical triggers
│   └── keywords_to_add.txt           # Staged keyword expansion list
│
├── data/
│   ├── best4baby_database.csv        # Product catalog
│   ├── best4baby_database_link.csv   # Direct product deep-links
│   └── chroma_catalog_db/            # (Local) ChromaDB persistent storage
│
├── models/
│   ├── config.json
│   ├── model.onnx                    # Transformer model exported to ONNX
│   ├── tokenizer.json                # Fast tokenizer vocabulary & configuration
│   └── ...
│
├── scripts/
│   ├── download_model.py             # Zero-dependency asset fetcher
│   ├── ingest.py                     # Catalog ingestion to ChromaDB
│   └── lexemes_generator.py          # Morphological variant expansion
│
├── src/
│   ├── __init__.py
│   ├── catalog.py                    # Vector DB query & catalog resolver
│   ├── classifier.py                 # ONNX Bi-Encoder intent classifier
│   ├── drafter.py                    # LLM response generation
│   └── prefilter.py                  # High-speed initial keyword filter
│
└── tests/
    ├── listener_test.py              # End-to-end simulated listener tests
    └── test_onnx.py                  # ONNX inference unit benchmarks
🚀 Quickstart Guide1. Environment SetupClone the repository and prepare a virtual environment:git clone https://github.com/OliverNicks/listener_telegram.git
cd listener_telegram

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
2. Download Model WeightsFetch the ONNX model graph and tokenizer assets:python scripts/download_model.py
3. Configure CredentialsCopy .env.example to .env and fill in your credentials:cp .env.example .env
# Telegram MTProto API Credentials (obtain from https://my.telegram.org)
TELEGRAM_API_ID=1234567
TELEGRAM_API_HASH=your_api_hash_here

# LLM Providers
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
4. Populate the Product CatalogIngest your product catalog into ChromaDB:python scripts/ingest.py
5. Authenticate SessionsAuthenticate your monitoring account:python bots/auth/login.py
(This will generate a secure .session file in sessions/ which is ignored by Git).6. Run the PipelineStart the listener bot:python -m bots.bot_b.listener
To seed organic conversations in parallel:python -m bots.bot_a.question_bot
🔒 Security & PrivacyZero Session Leakage: All Telethon *.session files and database files (*.db, chroma_catalog_db/) are excluded in .gitignore.Credential Isolation: Secrets must be stored solely inside .env. Never commit API keys or phone numbers.Non-Intrusive Engagement: Configurable similarity thresholds (threshold >= 0.55) prevent spamming irrelevant threads.📄 LicenseInternal project for best4baby.com.ua. All rights reserved.
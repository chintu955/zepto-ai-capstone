# Zepto Data & AI Platform — Capstone Project

Single repository containing three connected modules:

- `/data_pipeline` — scrape → clean → convert → store → query (books.toscrape.com)
- `/analytics` — EDA + predictive modeling pipeline on the Titanic dataset
- `/support_assistant` — a grounded GenAI RAG support assistant over Zepto policy docs

---

## 1. Setup

Each module has its own `requirements.txt` (chosen over one consolidated file, since
the three modules have non-overlapping dependencies — e.g. `beautifulsoup4` is only
needed for scraping, `chromadb`/`langgraph` only for the assistant).

```bash
# clone
git clone https://github.com/<your-username>/zepto-ai-capstone.git
cd zepto-ai-capstone

# create one virtual env (or one per module, your choice)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## 2. How to run each module

### Module 1 — Data Pipeline
```bash
cd data_pipeline
python scrape_and_load.py
```
Produces `raw_books.csv`, `clean_books.csv`, and `zepto_books.db` (SQLite), and prints
the 5 SQL queries + the `pd.read_sql` vs `pd.merge` comparison to the console.

**Design decisions:**
- Fixed conversion rate used: **1 GBP = 105.50 INR** (project-defined constant, no live lookup).
- Rows with unparseable title/availability are dropped; unparseable numeric fields
  (price/rating) are median-imputed rather than dropped, to preserve as much data as possible.
- Schema: `categories(category_id PK, category_name)` ← `books(book_id PK, ..., category_id FK)`.

### Module 2 — Analytics
```bash
cd analytics
python 01_eda.py         # loads titanic dataset once, cleans it, saves titanic.csv + charts
python 02_modeling.py    # reads the same titanic.csv, runs the full modeling pipeline
```
Produces `titanic.csv` (offline fallback), chart PNGs in `charts/`, and
`best_pipeline.joblib` (the full fitted preprocessing + RandomForest pipeline).

**Design decisions:**
- Missing-value thresholds: <5% missing → drop rows; 5–30% → median/mode impute;
  very high missing (`deck`, ~77%) → encode as its own "Missing" category rather than impute.
- All preprocessing (imputer, encoder, scaler) is wrapped in a `ColumnTransformer` +
  `Pipeline`, fit only on the training split, to structurally prevent test-set leakage.
- Random Forest was selected as the deployed classifier and saved as one complete
  `joblib` artifact (preprocessing + model together) so it works directly on raw input.

### Module 3 — Support Assistant
```bash
cd support_assistant
python ingest.py                      # embeds the 8 docs into ChromaDB (local, no API key)
uvicorn main:app --host 0.0.0.0 --port 7860
# in another terminal:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is your delivery fee?"}'
```

Or with Docker:
```bash
cd support_assistant
docker build -t zepto-assistant .
docker run -p 7860:7860 zepto-assistant
```

**Design decisions:**
- `MOCK_LLM` defaults to `1` (unset also counts as mock) — this is the fully graded,
  offline, deterministic path. No API key or network call to any LLM is required.
- Embeddings are generated locally with `sentence-transformers/all-MiniLM-L6-v2` and
  stored in a persistent ChromaDB collection — no account or API key needed.
- See `/support_assistant/README.md` for the full architecture write-up and example
  request/response transcripts.

---

## 3. Git Workflow

This repository's history includes a feature branch (`feature/support-assistant`)
created off `main`, committed to at least twice, and merged back into `main`.
Visible via:
```bash
git log --graph --all --oneline
```

## 4. Repository Structure

```
zepto-ai-capstone/
├── README.md
├── data_pipeline/
│   ├── scrape_and_load.py
│   └── requirements.txt
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── requirements.txt
└── support_assistant/
    ├── docs/doc_01.txt … doc_08.txt
    ├── ingest.py
    ├── prompts.py
    ├── graph.py
    ├── main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

# Support Assistant — Architecture & Example Calls

## Pipeline Architecture (Ingestion → Embedding → Retrieval → Generation)

**1. Ingestion** — `ingest.py` reads the 8 policy files from `docs/`, chunks each
document with `chunk_text()` (fixed-size, ~300 words; each policy doc is short
enough to become a single chunk), and tags each chunk with its `source_doc` id.

**2. Embedding** — Still in `ingest.py`, each chunk is embedded locally with
`sentence-transformers` (`all-MiniLM-L6-v2`, no API key, runs on CPU) and written
into a persistent ChromaDB collection named `zepto_policies` (stored on disk at
`support_assistant/chroma_store/`).

**3. Retrieval** — In `graph.py`, the `retrieve_and_answer` node embeds the
incoming query with the same model and queries the `zepto_policies` ChromaDB
collection for the top-3 most similar chunks (cosine similarity, handled
internally by Chroma). This step always runs for real in both `MOCK_LLM`
states, since it needs no API key or network call.

**4. Generation** — The final answer text is produced differently depending on
`MOCK_LLM`:
- **`MOCK_LLM=1` / unset (default, graded baseline):** no LLM call. The
  `retrieve_and_answer` node returns
  `f"Based on the retrieved context: {top_chunk[:200]}"`; the `direct_answer`
  node (for `general_question` intents) returns a fixed canned string. Both
  the intent classification in `classify_intent` and the answer generation are
  pure Python/keyword logic — fully deterministic and offline.
- **`MOCK_LLM=0` (optional extension):** `classify_intent` and the two answer
  nodes call a real LLM instead — `retrieve_and_answer` builds a prompt with
  `prompts.build_prompt()` (role–context–task–format–length skeleton with a
  negative constraint and a few-shot example) grounded in the retrieved
  chunks; `direct_answer` prompts the LLM with no retrieval. Any output that
  fails Pydantic validation against `AnswerSchema` is retried up to 2 more
  times with a corrective instruction (see `answer_query()` in `graph.py`).

Routing between `retrieve_and_answer` and `direct_answer` is a LangGraph
conditional edge (`route_by_intent`) that depends only on the classified
`intent`, not on `MOCK_LLM`.

```
query
  │
  ▼
[classify_intent] --(keyword heuristic, mock / LLM call, real)--
  │
  ├── policy_question ──► [retrieve_and_answer] ──► ChromaDB top-3 chunks ──► answer
  │
  └── general_question ──► [direct_answer] ──► canned / LLM answer
```

## Example Calls (MOCK_LLM left at default)

**Call 1 — should trigger retrieval:**
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is your delivery fee for small orders?"}'
```
Response:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order",
  "sources": ["doc_01_chunk0"],
  "confidence": 1.0
}
```

**Call 2 — should NOT trigger retrieval:**
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```
Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

> Note: exact retrieved snippet text may vary slightly depending on chunking,
> but will always be pulled verbatim from one of the 8 `docs/*.txt` files.

## Docker

```bash
docker build -t zepto-assistant .
docker run -p 7860:7860 zepto-assistant
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How do I cancel my membership?"}'
```

## Optional Extensions (not required for grading)
- Set `MOCK_LLM=0` and provide a Groq (or other free-tier) API key as an env
  var to enable real LLM classification/generation.
- Push the same Dockerfile to Hugging Face Spaces (free CPU tier) and store
  the API key as a Space secret — never commit it to the repo.

"""
Zepto Capstone - Module 3: LangGraph-orchestrated RAG flow.

Nodes:
  classify_intent    -> policy_question | general_question
  retrieve_and_answer -> retrieves top-3 chunks from ChromaDB, answers
  direct_answer       -> answers without retrieval

All LLM-generation steps branch on the MOCK_LLM env var.
  MOCK_LLM unset or "1" (default, graded baseline) -> deterministic mock logic, no API call
  MOCK_LLM="0" -> optional real-LLM extension (not required for grading)
"""

import os
import chromadb
from typing import TypedDict, List
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from prompts import build_prompt

MOCK_LLM = os.environ.get("MOCK_LLM", "1") == "1"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "zepto_policies"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]

_embed_model = None
_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------
class AnswerSchema(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[dict]
    answer: str
    sources: List[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if MOCK_LLM:
        lowered = query.lower()
        intent = "policy_question" if any(k in lowered for k in POLICY_KEYWORDS) \
            else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call a real LLM to classify instead.
        # e.g. intent = call_llm_classifier(query)
        intent = _real_llm_classify(query)  # pragma: no cover

    state["intent"] = intent
    return state


def _real_llm_classify(query: str) -> str:  # pragma: no cover
    """Placeholder for the optional real-LLM classification path (MOCK_LLM=0)."""
    raise NotImplementedError(
        "Real LLM classification is an optional extension. "
        "Plug in your LLM client here (e.g. Groq) and return "
        "'policy_question' or 'general_question'."
    )


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (runs when intent == policy_question)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real (no API key/network needed)
    model = _get_embed_model()
    collection = _get_collection()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=3)
    chunk_texts = results["documents"][0]
    chunk_ids = results["ids"][0]
    chunk_sources = [m["source_doc"] for m in results["metadatas"][0]]

    retrieved_chunks = [
        {"id": cid, "text": txt, "source_doc": src}
        for cid, txt, src in zip(chunk_ids, chunk_texts, chunk_sources)
    ]
    state["retrieved_chunks"] = retrieved_chunks

    if MOCK_LLM:
        top_chunk = retrieved_chunks[0]["text"]
        snippet = top_chunk[:200]
        state["answer"] = f"Based on the retrieved context: {snippet}"
        state["sources"] = [c["id"] for c in retrieved_chunks]
        state["confidence"] = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the real LLM grounded in chunks.
        context_str = "\n".join(c["text"] for c in retrieved_chunks)
        prompt = build_prompt(query, context_str)
        state["answer"] = _real_llm_generate(prompt)  # pragma: no cover
        state["sources"] = [c["id"] for c in retrieved_chunks]
        state["confidence"] = 0.9  # pragma: no cover

    return state


# ---------------------------------------------------------------------------
# Node 3: direct_answer (runs when intent == general_question)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        state["answer"] = _real_llm_generate(state["query"])  # pragma: no cover
        state["sources"] = []  # pragma: no cover
        state["confidence"] = 0.8  # pragma: no cover

    return state


def _real_llm_generate(prompt: str) -> str:  # pragma: no cover
    """Placeholder for the optional real-LLM generation path (MOCK_LLM=0)."""
    raise NotImplementedError(
        "Real LLM generation is an optional extension. "
        "Plug in your LLM client here (e.g. Groq's free tier)."
    )


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def answer_query(query: str) -> AnswerSchema:
    """Run the graph end-to-end and return a validated AnswerSchema."""
    app = build_graph()
    initial_state: GraphState = {
        "query": query, "intent": "", "retrieved_chunks": [],
        "answer": "", "sources": [], "confidence": 0.0,
    }

    if MOCK_LLM:
        final_state = app.invoke(initial_state)
        return AnswerSchema(
            answer=final_state["answer"],
            sources=final_state["sources"],
            confidence=final_state["confidence"],
        )
    else:
        # Optional extension: retry up to 2 additional times on validation failure
        last_error = None
        for attempt in range(3):
            try:
                final_state = app.invoke(initial_state)
                return AnswerSchema(
                    answer=final_state["answer"],
                    sources=final_state["sources"],
                    confidence=final_state["confidence"],
                )
            except ValidationError as e:  # pragma: no cover
                last_error = e
                continue
        return AnswerSchema(  # pragma: no cover
            answer=f"ERROR: could not produce a valid response after retries: {last_error}",
            sources=[], confidence=0.0,
        )


if __name__ == "__main__":
    for q in ["What is your delivery fee?", "What's the weather today?"]:
        result = answer_query(q)
        print(f"\nQuery: {q}")
        print(result.model_dump_json(indent=2))

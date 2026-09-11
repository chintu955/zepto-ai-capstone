"""
Zepto Capstone - Module 3: FastAPI wrapper around the LangGraph RAG pipeline.
Run locally: uvicorn main:app --host 0.0.0.0 --port 7860
Then POST to http://localhost:7860/ask with {"query": "..."}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from graph import answer_query, AnswerSchema

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AnswerSchema)
def ask(request: AskRequest):
    return answer_query(request.query)


@app.get("/")
def health():
    return {"status": "ok", "service": "zepto-support-assistant"}

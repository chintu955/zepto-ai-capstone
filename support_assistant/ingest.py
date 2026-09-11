"""
Zepto Capstone - Module 3: Ingest the 8 policy documents into ChromaDB
using local sentence-transformers embeddings (all-MiniLM-L6-v2).
Run: python ingest.py
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "zepto_policies"


def chunk_text(text, chunk_size=300):
    """Simple fixed-size chunking (documents are short, so usually 1 chunk each)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks if chunks else [text]


def build_collection():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # fresh collection each run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    ids, texts, metadatas = [], [], []

    for path in doc_paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. doc_01
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"source_doc": doc_id})

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    print(f"Ingested {len(ids)} chunks from {len(doc_paths)} documents into "
          f"ChromaDB collection '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    build_collection()

"""
build_index.py
--------------
Builds the FAISS index from the processed documents using:
  - BAAI/bge-small-en-v1.5       (33MB, retrieval-optimized)

Saves the index and its document list to disk.
Only needs to run once — indexes are reused by retriever.py

Run from project root:
  python rag/build_index.py
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed", "medical")
INDEX_DIR     = os.path.join("rag", "vector_index")
DOCS_FILE     = os.path.join(PROCESSED_DIR, "documents.jsonl")

os.makedirs(INDEX_DIR, exist_ok=True)

MODEL_NAME  = "BAAI/bge-small-en-v1.5"
MODEL_SLUG  = "bge_small"
BATCH_SIZE  = 256

# ─────────────────────────────────────────────
# LOAD DOCUMENTS
# ─────────────────────────────────────────────
def load_documents() -> list:
    print(f"\n[1] Loading documents from {DOCS_FILE}...")
    docs = []
    if not os.path.exists(DOCS_FILE):
        print(f"Warning: {DOCS_FILE} does not exist.")
        return []
        
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    from collections import Counter
    counts = Counter(d.get("source", "unknown") for d in docs)
    print(f"  Total: {len(docs)} documents")
    for src, cnt in counts.items():
        print(f"  {src:<20} {cnt:>7}")
    return docs

# ─────────────────────────────────────────────
# BUILD AND SAVE INDEX
# ─────────────────────────────────────────────
def build_and_save(docs: list):
    print(f"\n{'='*60}")
    print(f"  Building index: {MODEL_NAME}")
    print(f"{'='*60}")

    print(f"  Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    dim   = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")

    texts = [d["text"] for d in docs if "text" in d]
    if not texts:
        print("No valid texts to embed.")
        return

    print(f"  Embedding {len(texts)} documents in batches of {BATCH_SIZE}...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  Embedding [{MODEL_SLUG}]"):
        batch = texts[i : i + BATCH_SIZE]
        embs  = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"  Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors")

    index_path = os.path.join(INDEX_DIR, f"{MODEL_SLUG}.faiss")
    docs_path  = os.path.join(INDEX_DIR, f"{MODEL_SLUG}_docs.pkl")

    faiss.write_index(index, index_path)
    print(f"  FAISS index saved → {index_path}")

    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    print(f"  Documents saved  → {docs_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  SAFEHEALTH — FAISS INDEX BUILDER")
    print("="*60)

    docs = load_documents()
    if docs:
        print(f"\n[2] Building index: {MODEL_NAME}")
        build_and_save(docs)

    print("\n[OK] Index building complete.")

if __name__ == "__main__":
    main()

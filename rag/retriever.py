"""
retriever.py
------------
Handles querying the FAISS vector index, applying MMR for diverse chunks,
and using the semantic similarity score of the top chunk to determine if the query
is in-scope (medical) or out-of-scope.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class MedicalRetriever:
    def __init__(self, index_dir="rag/vector_index", slug="bge_small", threshold=0.65):
        self.index_dir = index_dir
        self.slug = slug
        self.threshold = threshold
        
        # Load embedding model
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"Loading Medical Retriever model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Load FAISS index and documents
        index_path = os.path.join(index_dir, f"{slug}.faiss")
        docs_path = os.path.join(index_dir, f"{slug}_docs.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}. Please run build_index.py first.")
            
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
            
        print(f"Retriever ready! {self.index.ntotal} vectors loaded.")

    def _mmr_rerank(self, query_emb, candidate_embs, candidate_docs, candidate_scores, final_k=3, lam=0.5):
        """
        Maximal Marginal Relevance re-ranking.
        Balances returning highly relevant chunks (lam) vs diverse chunks (1-lam).
        """
        selected_idx = []
        remaining_idx = list(range(len(candidate_docs)))

        for _ in range(min(final_k, len(candidate_docs))):
            if not remaining_idx:
                break

            best_idx = None
            best_score = -np.inf

            for idx in remaining_idx:
                rel_score = candidate_scores[idx]

                if selected_idx:
                    selected_embs = candidate_embs[selected_idx]
                    sim_to_selected = float(np.max(candidate_embs[idx] @ selected_embs.T))
                else:
                    sim_to_selected = 0.0

                mmr_score = lam * rel_score - (1 - lam) * sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_idx.append(best_idx)
            remaining_idx.remove(best_idx)

        return [
            {
                "score": float(candidate_scores[selected_idx[i]]),
                "text": candidate_docs[selected_idx[i]]["text"],
                "source": candidate_docs[selected_idx[i]]["source"],
                "ref": candidate_docs[selected_idx[i]].get("ref", "N/A")
            }
            for i in range(len(selected_idx))
        ]

    def retrieve(self, query: str, top_k=10, final_k=3, lam=0.5) -> dict:
        """
        Retrieves the best chunks. If the top chunk's similarity < self.threshold, 
        returns an empty list of chunks and flags is_in_scope=False.
        """
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        
        # FAISS exact search
        scores, indices = self.index.search(q_emb, top_k)
        
        # Check Rank-1 confidence threshold
        rank1_score = float(scores[0][0])
        print(f"[Retriever] Top semantic score for query '{query}': {rank1_score:.3f}")
        
        if rank1_score < self.threshold:
            print(f"[Retriever] REJECTED. Score {rank1_score:.3f} is below threshold {self.threshold}.")
            return {
                "is_in_scope": False,
                "chunks": [],
                "reason": "Low retrieval confidence. Query is likely out of scope."
            }
            
        # Collect top_k candidates for MMR
        candidate_docs = []
        candidate_scores = []
        candidate_embs = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            
            doc = self.docs[idx]
            text = doc.get("text", "")
            
            # Quality Filter 1: Ignore chunks that are just short headings or captions
            if len(text.strip()) < 150:
                continue
            
            candidate_docs.append(doc)
            candidate_scores.append(float(score))
            
            emb = np.zeros(self.index.d, dtype=np.float32)
            self.index.reconstruct(int(idx), emb)
            candidate_embs.append(emb)
            
        candidate_embs = np.vstack(candidate_embs)
        
        # MMR Selection (top 3)
        final_chunks = self._mmr_rerank(
            query_emb=q_emb[0],
            candidate_embs=candidate_embs,
            candidate_docs=candidate_docs,
            candidate_scores=candidate_scores,
            final_k=final_k,
            lam=lam
        )
        
        return {
            "is_in_scope": True,
            "chunks": final_chunks,
            "reason": f"Success. Top score: {rank1_score:.3f}"
        }

if __name__ == "__main__":
    retriever = MedicalRetriever()
    print("\nTesting medical query...")
    res1 = retriever.retrieve("What should I do if I have a fever?")
    print(res1)
    
    print("\nTesting non-medical query...")
    res2 = retriever.retrieve("Write a python script to solve two sum")
    print(res2)


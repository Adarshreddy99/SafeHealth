"""
guard.py
--------
The central orchestration point for query validation.
Passes the user query through Layer 1 (Regex), Layer 2 (Classifier), and calls
the Retriever to check semantic relevance (Thresholding).
"""

from safety.layer1_rules import Layer1RulesValidator
from safety.layer2_classifier import Layer2ClassifierValidator

class SafetyGuard:
    def __init__(self, retriever_instance):
        self.layer1 = Layer1RulesValidator()
        self.layer2 = Layer2ClassifierValidator()
        # We pass the active retriever instance in so we don't load BGE-small twice
        self.retriever = retriever_instance

    def process_query(self, query: str) -> dict:
        """
        Returns a dict:
        {
            "is_safe": bool,
            "message": str,
            "chunks": list,
            "telemetry": dict
        }
        """
        telemetry = {
            "l1_passed": False,
            "l2_zone": "N/A",
            "l2_score": 0.0,
            "retriever_score": 0.0
        }

        # 1. Regex Rules
        l1_res = self.layer1.validate_query(query)
        if not l1_res["is_safe"]:
            return {"is_safe": False, "message": "Your query was blocked by system safety rules.", "chunks": [], "telemetry": telemetry}
        telemetry["l1_passed"] = True

        # 2. Classifier (DistilBERT)
        l2_res = self.layer2.validate_query(query)
        telemetry["l2_zone"] = l2_res.get("zone", "N/A")
        telemetry["l2_score"] = l2_res.get("score", 0.0)
        
        if not l2_res["is_safe"]:
            return {"is_safe": False, "message": "Your query was flagged as adversarial or inappropriate.", "chunks": [], "telemetry": telemetry}

        # 3. Retrieval Confidence & Scope (BGE-small + FAISS)
        retrieval_res = self.retriever.retrieve(query, top_k=10, final_k=3)
        # We can extract the rank1 score from the chunks if it passed, or from a modification to retriever.
        # But wait, retriever currently prints rank1_score and doesn't explicitly return it easily. 
        # Actually, let's just grab the score of the top chunk returned if it passed!
        if retrieval_res["chunks"]:
            telemetry["retriever_score"] = max([c["score"] for c in retrieval_res["chunks"]])
            
        if not retrieval_res["is_in_scope"]:
            return {"is_safe": False, "message": "I am a medical assistant and can only answer health-related inquiries.", "chunks": [], "telemetry": telemetry}

        # All clear! We have our safe, relevant chunks.
        return {
            "is_safe": True,
            "message": "Query validated.",
            "chunks": retrieval_res["chunks"],
            "telemetry": telemetry
        }

"""
layer2_classifier.py
--------------------
Wrapper for the fine-tuned DistilBERT jailbreak classifier.
This acts as Layer 2, catching complex adversarial prompts that slip past Layer 1 regex.
"""

import sys
import os

# Ensure the root directory is in the path so we can import classifier.predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from classifier.predict import predict
except ImportError:
    print("Warning: Could not import classifier.predict. Layer 2 will mock safety.")
    predict = None

class Layer2ClassifierValidator:
    def __init__(self):
        pass

    def validate_query(self, query: str) -> dict:
        """
        Runs the DistilBERT jailbreak/toxicity classifier.
        Returns a dict indicating if the query passes Layer 2.
        """
        if predict is None:
            return {"is_safe": True, "reason": "Classifier not found, skipping."}

        try:
            result = predict(query)
            
            # If the query is deemed 'block' or 'jailbreak', we catch it here.
            # We can allow 'uncertain' to pass if we want, or block it. 
            # We'll block anything that is explicitly a 'block' zone.
            if result.get("zone") == "block":
                print(f"[Layer 2] Blocked by Classifier. Score: {result.get('score')}")
                return {
                    "is_safe": False,
                    "reason": "Query flagged by semantic classifier as unsafe/jailbreak.",
                    "zone": result.get("zone"),
                    "score": result.get("score")
                }
                
            return {
                "is_safe": True,
                "reason": f"Passed Layer 2.",
                "zone": result.get("zone"),
                "score": result.get("score")
            }
            
        except Exception as e:
            print(f"[Layer 2] Error running classifier: {e}")
            # Fail closed or open? Let's fail open for now so the app doesn't break if model is missing
            return {
                "is_safe": True,
                "reason": "Classifier threw an error, defaulting to transparent passthrough."
            }

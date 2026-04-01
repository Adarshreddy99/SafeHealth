"""
layer1_rules.py
---------------
Fast regex-based guardrail for the Safety Pipeline.
Blocks explicit self-harm, obvious off-topic prompts, and common jailbreak attempt words.
"""

import re

class Layer1RulesValidator:
    def __init__(self):
        # List of explicit keywords or phrases to block instantly (case-insensitive)
        blocklist = [
            r"\b(suicide|kill myself|end my life)\b", 
            r"\b(ignore previous instructions|ignore all previous instructions)\b",
            r"\b(write a python script|write some javascript|write code)\b",
            r"\b(pretend to be|act as if you are|you are now)\b",
            r"\b(bomb|terrorist|illegal drugs)\b"
        ]
        
        # Compile for speed
        self.block_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in blocklist]

    def validate_query(self, query: str) -> dict:
        """
        Returns a dict indicating if the query passes Layer 1.
        """
        for pattern in self.block_patterns:
            if pattern.search(query):
                print(f"[Layer 1] Matched blocked pattern: {pattern.pattern}")
                return {
                    "is_safe": False,
                    "reason": "Query blocked by regex rules (self-harm, prompt injection, or clearly out of scope)."
                }
        
        return {
            "is_safe": True,
            "reason": "Passed Layer 1 (Rules)."
        }

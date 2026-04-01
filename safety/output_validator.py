"""
output_validator.py
-------------------
Post-generation sanity check.
Enforces that the LLM behaves safely (e.g., includes a medical disclaimer)
and doesn't output raw code blocks.
"""

class OutputValidator:
    def __init__(self):
        self.disclaimer = "\n\n**Disclaimer**: This is an AI assistant, not a doctor. Please consult a healthcare professional for clinical advice."

    def validate_and_format(self, text: str) -> str:
        """
        Checks the generated text for bad behaviors, and appends the disclaimer.
        """
        # Block output if it looks like the model got tricked into writing code
        if "```python" in text or "```javascript" in text or "def " in text:
            return "I apologize, but I am restricted to providing medical information and cannot generate code or scripts."
            
        # Ensure it doesn't try to output raw HTML tags that could XSS a frontend
        if "<script>" in text or "<iframe>" in text:
            return "Error: Unsafe characters detected in output."

        # Ensure disclaimer is present
        if "Disclaimer" not in text:
            text += self.disclaimer
            
        return text

"""
prompt_builder.py
-----------------
Takes the user query and the retrieved context chunks, and formats them
strictly into the Llama-2 chat template used by Meditron-7b.
"""

class LlamaPromptBuilder:
    def __init__(self):
        self.system_prompt = (
            "You are SafeHealth, a specialized AI medical assistant. "
            "Your sole purpose is to provide helpful, scientific healthcare information "
            "based exactly on the provided Context. Do NOT provide diagnoses or prescribe medication. "
            "If the answer is not contained in the Context, say 'I don't know based on the provided literature.' "
            "Do exactly what is asked and give a short, concise answer without any formatting."
        )

    def build_prompt(self, user_query: str, retrieved_chunks: list) -> str:
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_str += f"--- Source {i+1} ({chunk['source']}) ---\n"
            context_str += f"{chunk['text']}\n\n"
            
        if not context_str.strip():
            context_str = "No relevant context found."

        # TinyLlama Chat Template
        prompt = f"""<|system|>
{self.system_prompt}</s>
<|user|>
CONTEXT:
{context_str}

USER QUESTION:
{user_query}</s>
<|assistant|>
"""
        return prompt

"""
model_loader.py
---------------
Loads the `meditron-7b-chat.Q4_K_M.gguf` model into memory
using the `llama-cpp-python` library, optimized for pure CPU usage.
"""

import os
from llama_cpp import Llama

class LLMRuntime:
    def __init__(self, model_path="Llama-3.2-3B-Instruct-Q4_K_S.gguf"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}. Please wait for the download to finish.")
            
        print(f"[LLM] Loading Llama-3.2-3B model from {model_path} into CPU memory...")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False,
        )
        print("[LLM] Llama-3.2-3B loaded successfully.")

    def generate(self, prompt: str) -> str:
        print("[LLM] Generating response...")
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
            echo=False
        )
        
        text = output['choices'][0]['text']
        return text.strip()

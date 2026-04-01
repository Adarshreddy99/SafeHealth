"""
chat.py
-------
The main pipeline script.
Runs the REPL loop for users to chat with SafeHealth end-to-end.
"""

import sys
import os

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from safety.guard import SafetyGuard
from rag.retriever import MedicalRetriever
from safety.output_validator import OutputValidator
from llm.model_loader import LLMRuntime
from llm.prompt_builder import LlamaPromptBuilder

def main():
    print("="*60)
    print("  SAFEHEALTH - END-TO-END RAG & SAFETY PIPELINE")
    print("="*60)
    
    # 1. Initialize Retriever
    try:
        retriever = MedicalRetriever()
    except Exception as e:
        print(f"\n[Error] Could not load Retriever: {e}")
        return

    # 2. Initialize Safety Layers
    guard = SafetyGuard(retriever_instance=retriever)
    output_checker = OutputValidator()
    
    # 3. Initialize LLM (Meditron CPU)
    try:
        llm_engine = LLMRuntime()
        prompt_maker = LlamaPromptBuilder()
    except Exception as e:
        print(f"\n[Error] Could not load LLM Runtime: {e}")
        return

    print("\n" + "="*60)
    print("  SYSTEM READY. Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    # The Chat Loop
    while True:
        try:
            query = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if query.strip().lower() in ['quit', 'exit']:
            print("Shutting down SafeHealth. Goodbye.")
            break
            
        if not query.strip():
            continue

        # -- Step 1: Safety & Retrieval
        print("\n  -> Checking safety and gathering context...")
        guard_res = guard.process_query(query)
        
        # --- PRINT METRICS & TELEMETRY ---
        tel = guard_res.get("telemetry", {})
        print("\n  [TELEMETRY PIPELINE]")
        print(f"  | Layer 1 Regex:      {'Pass' if tel.get('l1_passed') else 'Blocked'}")
        print(f"  | Layer 2 Classifier: Zone='{tel.get('l2_zone')}' | Risk Score={tel.get('l2_score', 0):.3f}")
        print(f"  | FAISS Retriever:    Max Score={tel.get('retriever_score', 0):.3f}")
        if guard_res.get("chunks"):
            print("  | Top 3 Chunks Retrieved:")
            for i, c in enumerate(guard_res["chunks"]):
                short_txt = c['text'].replace('\n', ' ')[:70] + "..."
                print(f"      [{i+1}] (Score: {c['score']:.3f}) {c['source']} -> {short_txt}")
        print("  " + "-"*50)
        
        if not guard_res["is_safe"]:
            print(f"SafeHealth (Blocked): {guard_res['message']}\n")
            continue
            
        chunks = guard_res["chunks"]

        # -- Step 2: Prompt Building
        prompt = prompt_maker.build_prompt(query, chunks)

        # -- Step 3: Generation
        print("  -> Asking Llama-3.2-3B...")
        raw_response = llm_engine.generate(prompt)

        # -- Step 4: Python Formatting (Bulletproof Citations)
        final_answer = output_checker.validate_and_format(raw_response)
        
        print("\nSafeHealth:")
        print("ANSWER:")
        print(final_answer)
        
        # Manually append the Top Chunk information directly from FAISS
        if chunks:
            print("\nLITERATURE USED:")
            for i, c in enumerate(chunks[:3]):
                print(f"\n--- Source {i+1}: {c['source']} -> {c.get('ref', 'N/A')} ---")
                print(c['text'].strip())
            
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()

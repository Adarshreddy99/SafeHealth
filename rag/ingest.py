"""
ingest.py
---------
Loads MedQuAD, PubMedQA_labeled, and MedRAG textbook chunks.
Cleans and normalizes all text into a unified list of documents.
Each document carries metadata: source dataset and original reference.

Run from project root:
  python rag/ingest.py
"""

import os
import re
import json
import pandas as pd

RAW_MEDICAL_DIR  = os.path.join("data", "raw", "medical")
PROCESSED_DIR    = os.path.join("data", "processed", "medical")
TEXTBOOKS_DIR    = os.path.join(RAW_MEDICAL_DIR, "textbooks")

MEDQUAD_FILE     = os.path.join(RAW_MEDICAL_DIR, "MedQuAD.parquet")
PUBMED_FILE      = os.path.join(RAW_MEDICAL_DIR, "PubMedQA_labeled.parquet")

OUTPUT_FILE      = os.path.join(PROCESSED_DIR, "documents.jsonl")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # remove non-ascii
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────
def load_medquad() -> list:
    print("\n[MedQuAD] Loading...")
    df = pd.read_parquet(MEDQUAD_FILE)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df)}")

    docs = []
    for _, row in df.iterrows():
        q = clean(str(row.get("question", "")))
        a = clean(str(row.get("answer",   "")))
        if not a or len(a) < 20:
            continue
        text = f"Question: {q}\nAnswer: {a}" if q else a
        docs.append({
            "text"    : text,
            "source"  : "MedQuAD",
            "ref"     : q[:80] if q else "MedQuAD entry",
        })

    print(f"  Documents produced: {len(docs)}")
    return docs


def load_pubmedqa() -> list:
    print("\n[PubMedQA Labeled] Loading...")
    df = pd.read_parquet(PUBMED_FILE)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df)}")

    docs = []
    for _, row in df.iterrows():
        q = clean(str(row.get("question",    "")))
        a = clean(str(row.get("long_answer", "")))
        if not a or len(a) < 20:
            continue
        decision = str(row.get("final_decision", "")).strip()
        text = f"Question: {q}\nAnswer: {a}"
        if decision:
            text += f"\nConclusion: {decision}"
        docs.append({
            "text"    : text,
            "source"  : "PubMedQA",
            "ref"     : q[:80] if q else "PubMedQA entry",
        })

    print(f"  Documents produced: {len(docs)}")
    return docs


def load_textbooks() -> list:
    print("\n[MedRAG Textbooks] Loading...")

    if not os.path.exists(TEXTBOOKS_DIR):
        print(f"  [!] Textbooks folder not found at {TEXTBOOKS_DIR}")
        return []

    # Detect all files — parquet or jsonl
    files = [f for f in os.listdir(TEXTBOOKS_DIR)
             if f.endswith(".parquet") or f.endswith(".jsonl")]
    print(f"  Found {len(files)} files in {TEXTBOOKS_DIR}")

    docs  = []
    for fname in sorted(files):
        fpath = os.path.join(TEXTBOOKS_DIR, fname)
        book_name = os.path.splitext(fname)[0]

        try:
            if fname.endswith(".parquet"):
                df = pd.read_parquet(fpath)

                # MedRAG textbook chunks use 'content' or 'text' column
                text_col = None
                for col in df.columns:
                    if col.lower() in ("content", "text", "chunk", "passage"):
                        text_col = col
                        break

                if text_col is None:
                    print(f"  [!] No text column found in {fname} | cols: {list(df.columns)}")
                    continue

                src_col = None
                for col in df.columns:
                    if col.lower() in ("source", "title", "book", "document"):
                        src_col = col
                        break

                before = len(docs)
                for _, row in df.iterrows():
                    text = clean(str(row[text_col]))
                    if len(text) < 30:
                        continue
                    ref = clean(str(row[src_col])) if src_col else book_name
                    docs.append({
                        "text"   : text,
                        "source" : "Textbook",
                        "ref"    : ref[:120],
                    })
                print(f"  {fname:<50} → {len(docs) - before} chunks")

            elif fname.endswith(".jsonl"):
                before = len(docs)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj  = json.loads(line)
                        text = clean(str(obj.get("content",
                                        obj.get("text",
                                        obj.get("chunk", "")))))
                        if len(text) < 30:
                            continue
                        ref = obj.get("source",
                              obj.get("title",
                              obj.get("book", book_name)))
                        docs.append({
                            "text"   : text,
                            "source" : "Textbook",
                            "ref"    : str(ref)[:120],
                        })
                print(f"  {fname:<50} → {len(docs) - before} chunks")

        except Exception as e:
            print(f"  [!] Error reading {fname}: {e}")
            continue

    print(f"  Total textbook docs: {len(docs)}")
    return docs


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SAFEHEALTH — DOCUMENT INGESTION")
    print("=" * 60)

    docs = []
    docs.extend(load_medquad())
    docs.extend(load_pubmedqa())
    docs.extend(load_textbooks())

    # Final filter — remove anything too short
    docs = [d for d in docs if len(d["text"]) >= 30]

    # Source breakdown
    from collections import Counter
    counts = Counter(d["source"] for d in docs)
    print(f"\n  Source breakdown:")
    for src, cnt in counts.items():
        print(f"    {src:<20} {cnt:>7} documents")
    print(f"\n  Total documents: {len(docs)}")

    # Save as JSONL — one doc per line
    print(f"\n  Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"  Saved {len(docs)} documents.")
    print("\n[OK] Ingestion complete.")
    return docs


if __name__ == "__main__":
    main()

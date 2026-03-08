"""
prepare_data.py
---------------
Loads semantic-router (all splits, no filtering) and jackhhao jailbreak datasets,
normalizes labels, merges, deduplicates, balances, and saves train/val/test splits
to data/processed/jailbreak/
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw", "jailbreak")
PROCESSED_DIR = os.path.join("data", "processed", "jailbreak")

SEMANTIC_TRAIN = os.path.join(RAW_DIR, "semantic.parquet")
SEMANTIC_TEST  = os.path.join(RAW_DIR, "semantic_test.parquet")
SEMANTIC_VAL   = os.path.join(RAW_DIR, "semantic_validation.parquet")

JACKHHAO_TRAIN = os.path.join(RAW_DIR, "jailbreak_dataset_train_balanced.csv")
JACKHHAO_TEST  = os.path.join(RAW_DIR, "jailbreak_dataset_test_balanced.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — LOAD SEMANTIC DATASET (ALL ROWS)
# ─────────────────────────────────────────────
def load_semantic() -> pd.DataFrame:
    print("\n[1] Loading semantic-router dataset...")

    parts = []
    for path, split_name in [
        (SEMANTIC_TRAIN, "train"),
        (SEMANTIC_TEST,  "test"),
        (SEMANTIC_VAL,   "validation"),
    ]:
        df = pd.read_parquet(path)
        parts.append(df)
        print(f"    {split_name}: {len(df)} rows | columns: {list(df.columns)}")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"    Total semantic rows loaded: {len(df_all)}")

    # ── Detect text and label columns ────────────────────────────────────────
    label_col = None
    text_col  = None

    for col in df_all.columns:
        if col.lower() in ("label", "label_text", "type"):
            label_col = col
        if col.lower() in ("text", "prompt", "query", "input"):
            text_col = col

    print(f"    Detected text column  : '{text_col}'")
    print(f"    Detected label column : '{label_col}'")
    print(f"    Label distribution:\n{df_all[label_col].value_counts()}\n")

    # ── Normalize to text / label (0 = benign, 1 = jailbreak) ────────────────
    df_all = df_all[[text_col, label_col]].copy()
    df_all.columns = ["text", "raw_label"]

    def normalize_label(val):
        if isinstance(val, (int, float)):
            return int(val)
        val_str = str(val).strip().lower()
        return 0 if val_str in ("benign", "safe", "0") else 1

    df_all["label"]  = df_all["raw_label"].apply(normalize_label)
    df_all           = df_all[["text", "label"]]
    df_all["source"] = "semantic"

    return df_all


# ─────────────────────────────────────────────
# STEP 2 — LOAD JACKHHAO DATASET
# ─────────────────────────────────────────────
def load_jackhhao() -> pd.DataFrame:
    print("[2] Loading jackhhao dataset...")

    parts = []
    for path, split_name in [
        (JACKHHAO_TRAIN, "train"),
        (JACKHHAO_TEST,  "test"),
    ]:
        df = pd.read_csv(path)
        parts.append(df)
        print(f"    {split_name}: {len(df)} rows | columns: {list(df.columns)}")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"    Total jackhhao rows: {len(df_all)}")
    print(f"    Type distribution:\n{df_all['type'].value_counts()}\n")

    # jackhhao columns: 'prompt' and 'type' (benign / jailbreak)
    df_all           = df_all[["prompt", "type"]].copy()
    df_all.columns   = ["text", "raw_label"]
    df_all["label"]  = df_all["raw_label"].apply(
        lambda x: 0 if str(x).strip().lower() == "benign" else 1
    )
    df_all           = df_all[["text", "label"]]
    df_all["source"] = "jackhhao"

    return df_all


# ─────────────────────────────────────────────
# STEP 3 — CLEAN TEXT
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


# ─────────────────────────────────────────────
# STEP 4 — MERGE, DEDUPLICATE, BALANCE
# ─────────────────────────────────────────────
def merge_and_balance(df_semantic: pd.DataFrame,
                      df_jackhhao: pd.DataFrame) -> pd.DataFrame:
    print("[3] Merging datasets...")

    df = pd.concat([df_semantic, df_jackhhao], ignore_index=True)
    print(f"    Combined rows before cleaning : {len(df)}")

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"    Rows after dedup              : {len(df)}")

    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df = df[(df["word_count"] >= 5) & (df["word_count"] <= 500)]
    df = df.drop(columns=["word_count"]).reset_index(drop=True)
    print(f"    Rows after length filter      : {len(df)}")

    counts = df["label"].value_counts()
    print(f"\n    Label counts before balancing:\n{counts}")

    ratio = counts.max() / counts.min()

    if ratio > 1.5:
        print(f"\n    Imbalance detected (ratio {ratio:.2f}) — undersampling majority...")
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        if len(df_0) > len(df_1):
            df_0 = df_0.sample(n=len(df_1), random_state=42)
        else:
            df_1 = df_1.sample(n=len(df_0), random_state=42)
        df = pd.concat([df_0, df_1], ignore_index=True)
    else:
        print(f"\n    Dataset is balanced (ratio {ratio:.2f}) — no undersampling needed.")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n    Final label counts:\n{df['label'].value_counts()}")
    print(f"    Total samples: {len(df)}\n")

    return df


# ─────────────────────────────────────────────
# STEP 5 — SPLIT AND SAVE
# ─────────────────────────────────────────────
def split_and_save(df: pd.DataFrame):
    print("[4] Splitting into train / val / test (80 / 10 / 10)...")

    df_train, df_temp = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, stratify=df_temp["label"]
    )

    print(f"    Train : {len(df_train)}")
    print(f"    Val   : {len(df_val)}")
    print(f"    Test  : {len(df_test)}")

    for split_df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        out  = split_df[["text", "label"]].reset_index(drop=True)
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        out.to_csv(path, index=False)
        print(f"    Saved -> {path}")

    print("\n[OK] Data preparation complete.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_semantic = load_semantic()
    df_jackhhao = load_jackhhao()
    df_merged   = merge_and_balance(df_semantic, df_jackhhao)
    split_and_save(df_merged)

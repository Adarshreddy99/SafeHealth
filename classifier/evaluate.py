"""
evaluate.py
-----------
Runs evaluation on the test split using TWO models side by side:
  1. Base distilbert-base-uncased (no fine-tuning)
  2. Your fine-tuned SafeHealth classifier

Prints full metrics for both and a side-by-side comparison table.
"""

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────
# PATHS AND SETTINGS
# ─────────────────────────────────────────────
PROCESSED_DIR  = os.path.join("data", "processed", "jailbreak")
MODEL_SAVE_DIR = os.path.join("classifier", "saved_model")
TEST_FILE      = os.path.join(PROCESSED_DIR, "test.csv")

BASE_MODEL = "distilbert-base-uncased"   # pulled from HuggingFace cache
MAX_LEN    = 256
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] Using: {DEVICE}")


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class JailbreakDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def run_inference(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            scores  = probs[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_scores),
    )


# ─────────────────────────────────────────────
# PRINT METRICS FUNCTION
# ─────────────────────────────────────────────
def print_metrics(name: str, preds, labels, scores):
    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    cm        = confusion_matrix(labels, preds)

    safe      = (scores < 0.3).sum()
    uncertain = ((scores >= 0.3) & (scores <= 0.7)).sum()
    block     = (scores > 0.7).sum()
    total     = len(scores)

    print(f"\n{'='*60}")
    print(f"  MODEL: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print()
    print("  Confusion Matrix:")
    print("                   Predicted")
    print("                   Benign   Jailbreak")
    print(f"  Actual Benign      {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  Actual Jailbreak   {cm[1][0]:>5}      {cm[1][1]:>5}")
    print()
    print("  Classification Report:")
    print(classification_report(labels, preds,
                                 target_names=["benign", "jailbreak"],
                                 zero_division=0))
    print("  Three-Zone Score Distribution:")
    print(f"  Safe      (0.0-0.3) : {safe:>5}  ({safe/total*100:.1f}%)")
    print(f"  Uncertain (0.3-0.7) : {uncertain:>5}  ({uncertain/total*100:.1f}%)")
    print(f"  Block     (0.7-1.0) : {block:>5}  ({block/total*100:.1f}%)")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────
def print_comparison(base_metrics: dict, finetuned_metrics: dict):
    print(f"\n{'='*60}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<15} {'Base DistilBERT':>18} {'Fine-Tuned':>15} {'Improvement':>14}")
    print(f"  {'-'*62}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        base_val = base_metrics[metric]
        ft_val   = finetuned_metrics[metric]
        diff     = ft_val - base_val
        sign     = "+" if diff >= 0 else ""
        print(f"  {metric.capitalize():<15} {base_val:>18.4f} {ft_val:>15.4f} {sign}{diff:>13.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n[1] Loading test data...")
    df_test = pd.read_csv(TEST_FILE)
    print(f"    Test samples: {len(df_test)}")
    print(f"    Label distribution:\n{df_test['label'].value_counts()}\n")

    # ── Load base tokenizer (shared by both models) ───────────────────────────
    print("[2] Loading base tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    print("[3] Tokenizing test data...")
    dataset = JailbreakDataset(
        df_test["text"].tolist(),
        df_test["label"].tolist(),
        tokenizer, MAX_LEN
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # ─────────────────────────────────────────
    # RUN 1 — BASE DISTILBERT (NO FINE-TUNING)
    # ─────────────────────────────────────────
    print("\n[4] Loading BASE distilbert-base-uncased (no fine-tuning)...")
    base_model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
    )
    base_model.to(DEVICE)

    print("    Running inference with base model...")
    base_preds, base_labels, base_scores = run_inference(base_model, loader)
    base_metrics = print_metrics("Base distilbert-base-uncased (no fine-tuning)",
                                  base_preds, base_labels, base_scores)

    # ─────────────────────────────────────────
    # RUN 2 — FINE-TUNED SAFEHEALTH CLASSIFIER
    # ─────────────────────────────────────────
    print(f"\n[5] Loading FINE-TUNED SafeHealth classifier from '{MODEL_SAVE_DIR}'...")

    if not os.path.exists(MODEL_SAVE_DIR):
        print(f"\n  [!] Fine-tuned model not found at '{MODEL_SAVE_DIR}'.")
        print("      Run train.py first, then re-run evaluate.py.\n")
        return

    ft_tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE_DIR)
    ft_model     = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    ft_model.to(DEVICE)

    # Re-tokenize with fine-tuned tokenizer (may differ slightly)
    ft_dataset = JailbreakDataset(
        df_test["text"].tolist(),
        df_test["label"].tolist(),
        ft_tokenizer, MAX_LEN
    )
    ft_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE)

    print("    Running inference with fine-tuned model...")
    ft_preds, ft_labels, ft_scores = run_inference(ft_model, ft_loader)
    ft_metrics = print_metrics("Fine-Tuned SafeHealth Classifier",
                                ft_preds, ft_labels, ft_scores)

    # ─────────────────────────────────────────
    # COMPARISON
    # ─────────────────────────────────────────
    print_comparison(base_metrics, ft_metrics)

    # ─────────────────────────────────────────
    # MISCLASSIFIED EXAMPLES (fine-tuned model)
    # ─────────────────────────────────────────
    df_test["pred"]  = ft_preds
    df_test["score"] = ft_scores
    df_wrong = df_test[df_test["label"] != df_test["pred"]]

    print(f"  Fine-tuned misclassified: {len(df_wrong)} / {len(df_test)}")
    if len(df_wrong) > 0:
        print("\n  Sample misclassified examples (up to 5):\n")
        for _, row in df_wrong.head(5).iterrows():
            true_lbl = "benign"    if row["label"] == 0 else "jailbreak"
            pred_lbl = "benign"    if row["pred"]  == 0 else "jailbreak"
            print(f"  Text  : {str(row['text'])[:120]}...")
            print(f"  True  : {true_lbl} | Predicted: {pred_lbl} | Score: {row['score']:.4f}")
            print()

    print("[OK] Evaluation complete.")


if __name__ == "__main__":
    main()

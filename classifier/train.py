"""
train.py
--------
Fine-tunes distilbert-base-uncased on the processed classifier dataset.
Saves the trained model and tokenizer to classifier/saved_model/

LR Schedule: linear warmup for first 10% of steps, then cosine decay to 0.
This is the standard and most effective schedule for fine-tuning transformers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

# ─────────────────────────────────────────────
# PATHS AND SETTINGS
# ─────────────────────────────────────────────
PROCESSED_DIR  = os.path.join("data", "processed", "jailbreak")
MODEL_SAVE_DIR = os.path.join("classifier", "saved_model")
BASE_MODEL = "distilbert-base-uncased"

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VAL_FILE   = os.path.join(PROCESSED_DIR, "val.csv")

MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 2e-5
WARMUP_PCT = 0.1    # 10% of total steps used for warmup

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


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
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─────────────────────────────────────────────
# METRICS FUNCTION
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="binary"),
    }


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    print("\n[1] Loading data...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_val   = pd.read_csv(VAL_FILE)

    print(f"    Train samples : {len(df_train)}")
    print(f"    Val samples   : {len(df_val)}")
    print(f"    Train label distribution:\n{df_train['label'].value_counts()}")

    print(f"\n[2] Loading tokenizer from '{BASE_MODEL}'...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    print("[3] Tokenizing datasets...")
    train_dataset = JailbreakDataset(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        tokenizer, MAX_LEN
    )
    val_dataset = JailbreakDataset(
        df_val["text"].tolist(),
        df_val["label"].tolist(),
        tokenizer, MAX_LEN
    )

    print(f"\n[4] Loading model '{BASE_MODEL}' for sequence classification...")
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "benign", 1: "jailbreak"},
        label2id={"benign": 0, "jailbreak": 1},
    )

    # ─────────────────────────────────────────
    # LR SCHEDULER SETUP
    # Cosine schedule with linear warmup:
    #   - Phase 1 (warmup): LR rises linearly from 0 to 2e-5 over first 10% steps
    #   - Phase 2 (decay):  LR follows cosine curve from 2e-5 down to ~0
    #
    # Why cosine over linear decay:
    #   - Linear decay drops LR steadily which can still cause instability mid-training
    #   - Cosine starts decaying slowly, accelerates in the middle, then slows again
    #     near the end — this gives the model more time to settle in the final epochs
    #   - For binary classification on short prompts, cosine consistently outperforms
    #     linear by 1-2% F1 in practice
    # ─────────────────────────────────────────
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = int(total_steps * WARMUP_PCT)

    print(f"\n[5] LR Scheduler info:")
    print(f"    Total training steps : {total_steps}")
    print(f"    Warmup steps (10%)   : {warmup_steps}")
    print(f"    Scheduler type       : cosine with linear warmup")
    print(f"    Peak LR              : {LR}")
    print(f"    Final LR             : ~0")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\n[6] Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join("classifier", "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # LR and scheduler managed manually above — set low default here
        learning_rate=LR,
        lr_scheduler_type="cosine",           # tells Trainer to use cosine
        warmup_steps=warmup_steps,            # replaces deprecated warmup_ratio
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),    # pass both explicitly
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n[7] Starting training...\n")
    trainer.train()

    print("\n[8] Saving model and tokenizer...")

    # Use trainer.save_model() not model.save_pretrained()
    # load_best_model_at_end=True means the Trainer holds the best checkpoint
    # internally in trainer.model. Saving from outer model variable would
    # save the last epoch weights, not the best epoch weights.
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # Verify saved files
    saved_files = os.listdir(MODEL_SAVE_DIR)
    print(f"\n    Files saved in {MODEL_SAVE_DIR}:")
    for f in saved_files:
        size_mb = os.path.getsize(os.path.join(MODEL_SAVE_DIR, f)) / 1e6
        print(f"      {f:<40} {size_mb:.2f} MB")

    print(f"\n[OK] Training complete. Best model saved to: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
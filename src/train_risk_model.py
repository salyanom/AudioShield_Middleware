"""
train_risk_model.py

Fine-tunes DistilBERT on data/risk_dataset.csv and saves the model to
models/risk_classifier/.  Everything configurable via CONFIG below —
no hardcoded values in the training logic.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_path":  "data/risk_dataset.csv",
    "model_out_dir": "models/risk_classifier",
    "base_model":    "distilbert-base-uncased",
    "max_length":    128,
    "test_size":     0.2,
    "random_seed":   42,
    "batch_size":    16,
    "epochs":        4,
    "learning_rate": 2e-5,
    "warmup_ratio":  0.1,
    "label_col":     "label",
    "text_col":      "text",
}
# ─────────────────────────────────────────────────────────────────────────────


class ResponseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_dataset(cfg):
    df = pd.read_csv(cfg["dataset_path"])
    df = df.dropna(subset=[cfg["text_col"], cfg["label_col"]])
    df[cfg["label_col"]] = df[cfg["label_col"]].astype(int)

    print(f"Dataset loaded: {len(df)} rows")
    print(f"  Safe (0): {(df[cfg['label_col']] == 0).sum()}")
    print(f"  Unsafe (1): {(df[cfg['label_col']] == 1).sum()}")

    return df[cfg["text_col"]].tolist(), df[cfg["label_col"]].tolist()


def build_loaders(texts, labels, tokenizer, cfg):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=labels,
    )

    train_ds = ResponseDataset(train_texts, train_labels, tokenizer, cfg["max_length"])
    val_ds   = ResponseDataset(val_texts,   val_labels,   tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])

    return train_loader, val_loader, val_texts, val_labels


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    texts, labels = load_dataset(cfg)

    # If you have a local copy of the model weights (e.g. downloaded once),
    # set base_model in CONFIG to the local path instead:
    #   "base_model": "/path/to/distilbert-base-uncased"
    # Or download with: huggingface-cli download distilbert-base-uncased
    print(f"Loading base model: {cfg['base_model']}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["base_model"])
    model     = DistilBertForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=2
    ).to(device)

    train_loader, val_loader, val_texts, val_labels = build_loaders(
        texts, labels, tokenizer, cfg
    )

    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0

    for epoch in range(cfg["epochs"]):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits  = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                trues.extend(batch["labels"].cpu().tolist())

        correct  = sum(p == t for p, t in zip(preds, trues))
        val_acc  = correct / len(trues)

        print(f"Epoch {epoch + 1}/{cfg['epochs']} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save(model, tokenizer, cfg, preds, trues, val_acc)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to:    {cfg['model_out_dir']}")


def _save(model, tokenizer, cfg, preds, trues, val_acc):
    out = Path(cfg["model_out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    report = classification_report(trues, preds, target_names=["safe", "unsafe"], output_dict=True)
    matrix = confusion_matrix(trues, preds).tolist()

    meta = {
        "base_model":    cfg["base_model"],
        "val_accuracy":  round(val_acc, 4),
        "classification_report": report,
        "confusion_matrix":      matrix,
        "config":                cfg,
    }

    with open(out / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    train(CONFIG)
from __future__ import annotations

import argparse

import numpy as np
import yaml
from datasets import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training.io import read_jsonl


def load_intents(intents_path: str):
    with open(intents_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    intents = cfg["intents"]
    label2id = {label: id for id, label in enumerate(intents)}
    id2label = {id2: label2 for label2, id2 in label2id.items()}
    return intents, label2id, id2label


def tokenize_fn(tokenizer, max_length: int):
    def _fn(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_length
        )

    return _fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--intents", default="config/intents.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    intents, label2id, id2label = load_intents(args.intents)
    set_seed(int(cfg.get("seed", 42)))

    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)

    def map_labels(rows):
        out = []
        for r in rows:
            lab = r["label"]
            if lab not in label2id:
                raise ValueError(f"Unknown label {lab}. Check {args.intents}.")
            out.append({"text": r["text"], "labels": label2id[lab]})
        return out

    train_ds = Dataset.from_list(map_labels(train_rows))
    val_ds = Dataset.from_list(map_labels(val_rows))

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["base_model"])
    train_ds = train_ds.map(tokenize_fn(tokenizer, int(cfg["max_length"])), batched=True)
    val_ds = val_ds.map(tokenize_fn(tokenizer, int(cfg["max_length"])), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=len(intents), id2label=id2label, label2id=label2id
    )

    ta = TrainingArguments(
        output_dir=args.out,
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["logging_steps"]),
        evaluation_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()

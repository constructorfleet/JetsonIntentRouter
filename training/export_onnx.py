from __future__ import annotations
import argparse
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to HF model dir (saved_model)")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    model = DistilBertForSequenceClassification.from_pretrained(args.model)
    model.eval()
    tok = DistilBertTokenizerFast.from_pretrained(args.model)

    dummy = tok(
        "turn off the lights",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.seq_len
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        args.out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=args.opset,
        do_constant_folding=True
    )

if __name__ == "__main__":
    main()

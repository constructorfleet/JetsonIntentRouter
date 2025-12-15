from __future__ import annotations

from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast


class OrtIntentClassifier:
    def __init__(
            self,
            onnx_path: str,
            intents: List[str],
            seq_len: int = 32,
            model_name: str = "distilbert-base-uncased"
    ):
        self.intents = intents
        self.seq_len = seq_len
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.sess = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"]
        )

    def _encode(self, text: str):
        t = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            return_tensors="np",
        )
        return {
            "input_ids": t["input_ids"].astype(np.int64),
            "attention_mask": t["attention_mask"].astype(np.int64),
        }

    def predict(self, text: str) -> Tuple[str, float]:
        inputs = self._encode(text)
        logits = self.sess.run(["logits"], inputs)[0]  # (1, num_labels)
        probs = _softmax(logits[0])
        idx = int(np.argmax(probs))
        return self.intents[idx], float(probs[idx])

def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

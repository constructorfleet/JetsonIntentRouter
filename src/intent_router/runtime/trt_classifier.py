from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

# Optional deps on Jetson:
#   import tensorrt as trt
#   import pycuda.driver as cuda
#   import pycuda.autoinit
#
# This module is imported only when ROUTER_BACKEND=trt.
import tensorrt as trt
from transformers import DistilBertTokenizerFast


class TrtEngineRunner:
    def __init__(self, engine_path: str, logger_severity=trt.Logger.WARNING):
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_bindings
        self.host = {}
        self.dev = {}

    def _alloc(self, name: str, shape: tuple, dtype: np.dtype):
        size = int(np.prod(shape))
        host = cuda.pagelocked_empty(size, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        self.host[name] = host
        self.dev[name] = dev
        idx = self.engine.get_binding_index(name)
        self.bindings[idx] = int(dev)

    def ensure(self, name: str, shape: tuple, dtype: np.dtype):
        name_missing = name in self.host
        if (
            name_missing
            or self.host[name].size != int(np.prod(shape))
            or self.host[name].dtype != dtype
        ):
            self._alloc(name, shape, dtype)

    def infer(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if input_ids.dtype != np.int32:
            input_ids = input_ids.astype(np.int32)
        if attention_mask.dtype != np.int32:
            attention_mask = attention_mask.astype(np.int32)

        for nm, arr in [("input_ids", input_ids), ("attention_mask", attention_mask)]:
            idx = self.engine.get_binding_index(nm)
            if any(d < 0 for d in self.context.get_binding_shape(idx)):
                self.context.set_binding_shape(idx, tuple(arr.shape))
            self.ensure(nm, tuple(arr.shape), arr.dtype)

        out_name = "logits"
        out_idx = self.engine.get_binding_index(out_name)
        out_shape = tuple(self.context.get_binding_shape(out_idx))
        out_dtype = trt.nptype(self.engine.get_binding_dtype(out_idx))
        self.ensure(out_name, out_shape, out_dtype)

        np.copyto(self.host["input_ids"], input_ids.ravel())
        np.copyto(self.host["attention_mask"], attention_mask.ravel())

        cuda.memcpy_htod_async(self.dev["input_ids"], self.host["input_ids"], self.stream)
        cuda.memcpy_htod_async(self.dev["attention_mask"], self.host["attention_mask"], self.stream)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        cuda.memcpy_dtoh_async(self.host[out_name], self.dev[out_name], self.stream)
        self.stream.synchronize()
        return self.host[out_name].reshape(out_shape).copy()


class TrtIntentClassifier:
    def __init__(
        self,
        engine_path: str,
        intents: List[str],
        seq_len: int = 32,
        model_name: str = "distilbert-base-uncased",
    ):
        self.intents = intents
        self.seq_len = seq_len
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.runner = TrtEngineRunner(engine_path)

    def _encode(self, text: str):
        t = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            return_tensors="np",
        )
        return t["input_ids"].astype(np.int32), t["attention_mask"].astype(np.int32)

    def predict(self, text: str) -> Tuple[str, float]:
        input_ids, attention_mask = self._encode(text)
        logits = self.runner.infer(input_ids, attention_mask)[0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return self.intents[idx], float(probs[idx])


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

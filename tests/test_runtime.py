import importlib
import sys

import numpy as np
import pytest


def test_ort_intent_classifier_predict(monkeypatch):
    class DummyTokenizer:
        def __init__(self, model_name):
            self.model_name = model_name

        @classmethod
        def from_pretrained(cls, model_name):
            return cls(model_name)

        def __call__(self, text, padding, truncation, max_length, return_tensors):
            return {
                "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64),
            }

    class DummySession:
        def __init__(self, onnx_path, providers):
            self.path = onnx_path
            self.providers = providers

        def run(self, outputs, inputs):
            # Strongly prefer last intent.
            return [np.array([[1.0, 2.0, 4.0]], dtype=np.float32)]

    dummy_ort = type("ort", (), {"InferenceSession": DummySession})
    dummy_transformers = type("transformers", (), {"DistilBertTokenizerFast": DummyTokenizer})

    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)
    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)
    sys.modules.pop("intent_router.runtime.ort_classifier", None)
    ort_module = importlib.import_module("intent_router.runtime.ort_classifier")

    clf = ort_module.OrtIntentClassifier("model.onnx", intents=["A", "B", "C"], seq_len=4)

    intent, prob = clf.predict("hello world")

    assert intent == "C"
    assert prob == pytest.approx(0.844, rel=1e-3)
    assert clf.sess.path == "model.onnx"
    assert clf.sess.providers == ["CPUExecutionProvider"]


def test_trt_intent_classifier_predict(monkeypatch):
    class DummyTokenizer:
        def __init__(self, model_name):
            self.model_name = model_name

        @classmethod
        def from_pretrained(cls, model_name):
            return cls(model_name)

        def __call__(self, text, padding, truncation, max_length, return_tensors):
            return {
                "input_ids": np.array([[10, 20, 30, 40]], dtype=np.int32),
                "attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int32),
            }

    # Minimal TensorRT + pycuda stubs for import.
    class DummyLogger:
        WARNING = "warning"

        def __init__(self, severity=None):
            self.severity = severity

    class DummyRuntime:
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, engine_bytes):
            return object()

    dummy_trt = type(
        "trt",
        (),
        {
            "Logger": DummyLogger,
            "Runtime": DummyRuntime,
            "nptype": lambda dtype: np.float32,
        },
    )

    dummy_cuda_driver = type(
        "cuda",
        (),
        {
            "Stream": lambda: type("Stream", (), {"handle": 1, "synchronize": lambda self: None})(),
            "pagelocked_empty": lambda size, dtype: np.zeros(size, dtype=dtype),
            "mem_alloc": lambda nbytes: type("Mem", (), {"nbytes": nbytes})(),
            "memcpy_htod_async": lambda dst, src, stream: None,
            "memcpy_dtoh_async": lambda dst, src, stream: None,
        },
    )

    # Prepare dummy modules for sys.modules before import.
    monkeypatch.setitem(sys.modules, "tensorrt", dummy_trt)
    monkeypatch.setitem(sys.modules, "pycuda", type("pycuda", (), {}))
    monkeypatch.setitem(sys.modules, "pycuda.autoinit", type("autoinit", (), {})())
    monkeypatch.setitem(sys.modules, "pycuda.driver", dummy_cuda_driver)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type("transformers", (), {"DistilBertTokenizerFast": DummyTokenizer}),
    )

    sys.modules.pop("intent_router.runtime.trt_classifier", None)
    trt_module = importlib.import_module("intent_router.runtime.trt_classifier")

    class FakeRunner:
        def __init__(self, engine_path):
            self.engine_path = engine_path
            self.called_with = None

        def infer(self, input_ids, attention_mask):
            self.called_with = (input_ids.copy(), attention_mask.copy())
            return np.array([[0.1, 1.1]], dtype=np.float32)

    monkeypatch.setattr(trt_module, "TrtEngineRunner", FakeRunner)

    clf = trt_module.TrtIntentClassifier("engine.plan", intents=["No", "Yes"], seq_len=4)

    intent, prob = clf.predict("turn on the tv")

    assert intent == "Yes"
    assert prob == pytest.approx(0.731, rel=1e-3)
    assert isinstance(clf.runner, FakeRunner)
    assert clf.runner.engine_path == "engine.plan"
    np.testing.assert_array_equal(
        clf.runner.called_with[0], np.array([[10, 20, 30, 40]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        clf.runner.called_with[1], np.array([[1, 1, 1, 0]], dtype=np.int32)
    )

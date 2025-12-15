from __future__ import annotations
import argparse
from onnxsim import simplify
import onnx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.inp)
    simp, ok = simplify(model)
    if not ok:
        raise SystemExit("onnxsim simplify failed")
    onnx.save(simp, args.out)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export HAC(contrastive) checkpoints (.pth) to ONNX for inference.

本仓库的 HAC 权重是成对保存的：
- proj_head_<mode>.pth  (ProjectionHead)
- linear_<mode>.pth     (LinearClassifier)

该脚本会把两者合并成一个 ONNX 推理图：
input:  (batch, in_dim) float32 embedding
output: (batch, num_classes) float32 概率(默认 softmax)

示例：
  python export_hac_pth_to_onnx.py --mode both
  python export_hac_pth_to_onnx.py --mode esm --output best_check12/hac/esm/esm_infer.onnx

可选：
  python export_hac_pth_to_onnx.py --mode both --verify
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from con_train import LinearClassifier, ProjectionHead


_DEFAULT_HAC_DIR = os.path.join(_HERE, "best_check12", "hac")


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def _infer_proj_head_hparams(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int, int]:
    """Infer (in_dim, hidden, out_dim, rank, steps) from ProjectionHead state_dict."""
    if "net.0.weight" not in state:
        raise KeyError("Unexpected ProjectionHead state_dict: missing 'net.0.weight'")
    if "net.5.weight" not in state:
        raise KeyError("Unexpected ProjectionHead state_dict: missing 'net.5.weight'")

    w0 = state["net.0.weight"]  # [hidden, in_dim]
    in_dim = int(w0.shape[1])
    hidden = int(w0.shape[0])
    out_dim = int(state["net.5.weight"].shape[0])

    step0 = "net.1.delta_calculations.0.0.weight"
    if step0 not in state:
        raise KeyError("Unexpected ProjectionHead state_dict: missing Iterblock keys")
    rank = int(state[step0].shape[0])

    steps = 0
    while f"net.1.delta_calculations.{steps}.0.weight" in state:
        steps += 1
    if steps <= 0:
        raise ValueError("Failed to infer steps for Iterblock")

    return in_dim, hidden, out_dim, rank, steps


def _infer_linear_hparams(state: Dict[str, torch.Tensor]) -> Tuple[int, int, Optional[int], bool]:
    """Infer (in_dim, num_classes, hidden or None, use_layernorm) from LinearClassifier state_dict."""
    if "net.weight" in state and "net.bias" in state:
        w = state["net.weight"]
        return int(w.shape[1]), int(w.shape[0]), None, False

    has_ln = "net.0.weight" in state and state["net.0.weight"].ndim == 1
    if has_ln:
        first_linear_key = "net.1.weight"
        last_linear_key = "net.4.weight"
        use_layernorm = True
    else:
        first_linear_key = "net.0.weight"
        last_linear_key = "net.3.weight"
        use_layernorm = False

    if first_linear_key not in state or last_linear_key not in state:
        keys_preview = ", ".join(list(state.keys())[:10])
        raise KeyError(f"Unexpected LinearClassifier state_dict keys. Preview: {keys_preview}")

    w1 = state[first_linear_key]  # [hidden, in_dim]
    w2 = state[last_linear_key]  # [num_classes, hidden]

    hidden = int(w1.shape[0])
    in_dim = int(w1.shape[1])
    num_classes = int(w2.shape[0])
    return in_dim, num_classes, hidden, use_layernorm


class HACOnnxInference(nn.Module):
    def __init__(self, head: nn.Module, clf: nn.Module, *, apply_softmax: bool):
        super().__init__()
        self.head = head
        self.clf = clf
        self.apply_softmax = bool(apply_softmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(x)
        logits = self.clf(z)
        if self.apply_softmax:
            return torch.softmax(logits, dim=1)
        return logits


def _default_ckpt_dir(mode: str) -> str:
    mode = mode.lower()
    if mode not in {"esm", "prot", "both"}:
        raise ValueError("--mode must be one of: esm, prot, both")
    return os.path.join(_DEFAULT_HAC_DIR, mode)


def export_onnx(*, mode: str, ckpt_dir: str, output: str, opset: int, softmax: bool, verify: bool) -> None:
    proj_path = os.path.join(ckpt_dir, f"proj_head_{mode}.pth")
    lin_path = os.path.join(ckpt_dir, f"linear_{mode}.pth")

    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Missing ProjectionHead ckpt: {proj_path}")
    if not os.path.exists(lin_path):
        raise FileNotFoundError(f"Missing LinearClassifier ckpt: {lin_path}")

    proj_state = _strip_module_prefix(torch.load(proj_path, map_location="cpu"))
    in_dim, hidden, out_dim, rank, steps = _infer_proj_head_hparams(proj_state)

    head = ProjectionHead(
        in_dim=int(in_dim),
        hidden=int(hidden),
        out_dim=int(out_dim),
        rank=int(rank),
        steps=int(steps),
        dropout=0.1,
    )
    head.load_state_dict(proj_state, strict=True)
    head.eval()

    lin_state = _strip_module_prefix(torch.load(lin_path, map_location="cpu"))
    in_dim2, num_classes, hidden2, use_layernorm = _infer_linear_hparams(lin_state)
    if int(in_dim2) != int(out_dim):
        raise ValueError(f"Linear head input dim mismatch: proj_out={out_dim}, but linear expects {in_dim2}")

    clf = LinearClassifier(
        in_dim=int(out_dim),
        num_classes=int(num_classes),
        hidden=hidden2,
        dropout=0.1,
        use_layernorm=bool(use_layernorm),
    )
    clf.load_state_dict(lin_state, strict=True)
    clf.eval()

    model = HACOnnxInference(head, clf, apply_softmax=softmax).eval()

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    dummy = torch.randn(1, int(in_dim), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["probs" if softmax else "logits"],
        dynamic_axes={
            "input": {0: "batch"},
            ("probs" if softmax else "logits"): {0: "batch"},
        },
    )

    print(f"ONNX export done: {output}")

    if verify:
        try:
            import numpy as np
            import onnx
            import onnxruntime as ort

            onnx_model = onnx.load(output)
            onnx.checker.check_model(onnx_model)

            x = torch.randn(4, int(in_dim), dtype=torch.float32)
            with torch.no_grad():
                y_torch = model(x).cpu().numpy()

            sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
            y_onnx = sess.run(None, {"input": x.cpu().numpy().astype(np.float32)})[0]

            max_abs = float(np.max(np.abs(y_torch - y_onnx)))
            print(f"verify ok. max_abs_diff={max_abs:.6g}")
        except ImportError:
            raise SystemExit("verify 需要安装 onnx 和 onnxruntime：pip install onnx onnxruntime")


def main() -> None:
    p = argparse.ArgumentParser(description="Export HAC (proj_head + linear) .pth to ONNX for inference")
    p.add_argument("--mode", required=True, choices=["esm", "prot", "both"], help="which HAC model to export")
    p.add_argument("--ckpt-dir", default=None, help="directory containing proj_head_<mode>.pth and linear_<mode>.pth")
    p.add_argument("--output", default=None, help="output .onnx path")
    p.add_argument("--opset", type=int, default=13, help="onnx opset version")
    p.add_argument("--no-softmax", action="store_true", help="export logits instead of probabilities")
    p.add_argument("--verify", action="store_true", help="run a small torch vs onnxruntime check")

    args = p.parse_args()

    mode = str(args.mode).lower()
    ckpt_dir = str(args.ckpt_dir) if args.ckpt_dir else _default_ckpt_dir(mode)
    output = str(args.output) if args.output else os.path.join(ckpt_dir, f"{mode}_infer.onnx")

    export_onnx(
        mode=mode,
        ckpt_dir=ckpt_dir,
        output=output,
        opset=int(args.opset),
        softmax=not bool(args.no_softmax),
        verify=bool(args.verify),
    )


if __name__ == "__main__":
    main()

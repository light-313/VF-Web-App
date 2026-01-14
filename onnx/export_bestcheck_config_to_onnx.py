#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export best_check/config.json models (.pth) to ONNX.

适用场景：s_maxvote.py 的 4 模型配置（DPF + VFITER*3）。
该脚本会读取 config.json，逐个加载对应 .pth，并导出为独立的 ONNX 推理模型。

导出后的 ONNX：
- DPF:   输入 (esm_features, prot_features) -> 输出 probs/logits
- VFITER: 输入 (features) -> 输出 probs/logits

示例：
  python export_bestcheck_config_to_onnx.py --config best_check/config.json
  python export_bestcheck_config_to_onnx.py --config best_check/config.json --out-dir best_check/onnx
  python export_bestcheck_config_to_onnx.py --config best_check/config.json --verify

说明：
- 默认导出 probabilities（softmax 后）。如果你想导出 logits，用 --no-softmax。
- verify 需要安装 onnx 和 onnxruntime：pip install onnx onnxruntime
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model_type import DPF, VFITER


class _DPFInfer(nn.Module):
    def __init__(self, model: nn.Module, *, apply_softmax: bool):
        super().__init__()
        self.model = model
        self.apply_softmax = bool(apply_softmax)

    def forward(self, esm_features: torch.Tensor, prot_features: torch.Tensor) -> torch.Tensor:
        logits = self.model((esm_features, prot_features), None)
        if self.apply_softmax:
            return torch.softmax(logits, dim=1)
        return logits


class _VFITERInfer(nn.Module):
    def __init__(self, model: nn.Module, *, apply_softmax: bool):
        super().__init__()
        self.model = model
        self.apply_softmax = bool(apply_softmax)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.model(features, None)
        if self.apply_softmax:
            return torch.softmax(logits, dim=1)
        return logits


def _infer_dpf_input_dims_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Infer (esm_dim, prot_dim) from DPF state_dict."""
    esm_key = "esm_proj.0.weight"
    prot_key = "prot5_proj.0.weight"
    if esm_key not in state or prot_key not in state:
        preview = ", ".join(list(state.keys())[:12])
        raise KeyError(f"DPF state_dict missing expected keys. Preview: {preview}")
    esm_dim = int(state[esm_key].shape[1])
    prot_dim = int(state[prot_key].shape[1])
    return esm_dim, prot_dim


def _export_one_model(
    *,
    spec: Dict[str, Any],
    out_dir: str,
    opset: int,
    softmax: bool,
    verify: bool,
) -> str:
    mtype = str(spec.get("type", "")).lower()
    ftype = str(spec.get("feature_type", "")).lower()
    pth_path = str(spec.get("path", ""))
    if not pth_path:
        raise ValueError("model spec missing 'path'")

    state = torch.load(pth_path, map_location="cpu")

    if mtype in {"dpf", "improveddualpathwayfusion", "fusion"}:
        esm_dim, prot_dim = _infer_dpf_input_dims_from_state(state)
        model = DPF(
            esm_dim=int(esm_dim),
            prot5_dim=int(prot_dim),
            hidden_dim=int(spec["hidden_dim"]),
            num_layers=int(spec["num_layers"]),
            num_classes=2,
            rank=int(spec["rank"]),
            steps=int(spec["steps"]),
            dropout=float(spec["dropout"]),
        )
        model.load_state_dict(state, strict=True)
        model.eval()

        wrapper = _DPFInfer(model, apply_softmax=softmax).eval()

        out_name = f"dpf_{os.path.splitext(os.path.basename(pth_path))[0]}.onnx"
        out_path = os.path.join(out_dir, out_name)

        dummy_esm = torch.randn(1, int(esm_dim), dtype=torch.float32)
        dummy_prot = torch.randn(1, int(prot_dim), dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            (dummy_esm, dummy_prot),
            out_path,
            export_params=True,
            opset_version=int(opset),
            do_constant_folding=True,
            input_names=["esm_features", "prot_features"],
            output_names=["probs" if softmax else "logits"],
            dynamic_axes={
                "esm_features": {0: "batch"},
                "prot_features": {0: "batch"},
                ("probs" if softmax else "logits"): {0: "batch"},
            },
        )

        if verify:
            _verify_onnx_dpf(wrapper, out_path, esm_dim=int(esm_dim), prot_dim=int(prot_dim))

        return out_path

    if mtype in {"vfiter", "delta"}:
        input_dim = int(spec["input_dim"])
        model = VFITER(
            input_dim=int(input_dim),
            hidden_dim=int(spec["hidden_dim"]),
            num_layers=int(spec["num_layers"]),
            num_classes=2,
            rank=int(spec["rank"]),
            steps=int(spec["steps"]),
            dropout=float(spec["dropout"]),
        )
        model.load_state_dict(state, strict=True)
        model.eval()

        wrapper = _VFITERInfer(model, apply_softmax=softmax).eval()

        tag = ftype if ftype else "vfiter"
        out_name = f"vfiter_{tag}_{os.path.splitext(os.path.basename(pth_path))[0]}.onnx"
        out_path = os.path.join(out_dir, out_name)

        dummy = torch.randn(1, int(input_dim), dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            export_params=True,
            opset_version=int(opset),
            do_constant_folding=True,
            input_names=["features"],
            output_names=["probs" if softmax else "logits"],
            dynamic_axes={
                "features": {0: "batch"},
                ("probs" if softmax else "logits"): {0: "batch"},
            },
        )

        if verify:
            _verify_onnx_single(wrapper, out_path, input_dim=int(input_dim))

        return out_path

    raise ValueError(f"Unsupported model type: {spec.get('type')}")


def _verify_onnx_single(model: nn.Module, onnx_path: str, *, input_dim: int) -> None:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        x = torch.randn(4, int(input_dim), dtype=torch.float32)
        with torch.no_grad():
            y_t = model(x).cpu().numpy()

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        y_o = sess.run(None, {"features": x.cpu().numpy().astype(np.float32)})[0]

        max_abs = float(np.max(np.abs(y_t - y_o)))
        print(f"verify ok: {os.path.basename(onnx_path)} max_abs_diff={max_abs:.6g}")
    except ImportError:
        raise SystemExit("verify 需要安装 onnx 和 onnxruntime：pip install onnx onnxruntime")


def _verify_onnx_dpf(model: nn.Module, onnx_path: str, *, esm_dim: int, prot_dim: int) -> None:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        esm = torch.randn(4, int(esm_dim), dtype=torch.float32)
        prot = torch.randn(4, int(prot_dim), dtype=torch.float32)
        with torch.no_grad():
            y_t = model(esm, prot).cpu().numpy()

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        y_o = sess.run(
            None,
            {
                "esm_features": esm.cpu().numpy().astype(np.float32),
                "prot_features": prot.cpu().numpy().astype(np.float32),
            },
        )[0]

        max_abs = float(np.max(np.abs(y_t - y_o)))
        print(f"verify ok: {os.path.basename(onnx_path)} max_abs_diff={max_abs:.6g}")
    except ImportError:
        raise SystemExit("verify 需要安装 onnx 和 onnxruntime：pip install onnx onnxruntime")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export best_check/config.json models (.pth) to ONNX")
    ap.add_argument("--config", default=os.path.join("best_check", "config.json"), help="path to config.json")
    ap.add_argument("--out-dir", default=os.path.join("best_check", "onnx"), help="output directory")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--no-softmax", action="store_true", help="export logits instead of probabilities")
    ap.add_argument("--verify", action="store_true", help="check torch vs onnxruntime")

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    specs = cfg.get("models") or []
    if not isinstance(specs, list) or not specs:
        raise SystemExit("config.json 中 models 为空")

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_paths = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise SystemExit(f"models[{i}] 不是 dict")
        out_path = _export_one_model(
            spec=spec,
            out_dir=out_dir,
            opset=int(args.opset),
            softmax=not bool(args.no_softmax),
            verify=bool(args.verify),
        )
        out_paths.append(out_path)
        print(f"exported: {out_path}")

    print(f"done. exported {len(out_paths)} models to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark best_check 4-model ensemble: PyTorch (.pth) vs ONNXRuntime (.onnx).

适用：s_maxvote.py 对应的 best_check/config.json（DPF + VFITER*3）。

准备：
- 先运行 export_bestcheck_config_to_onnx.py 生成 best_check/onnx/*.onnx

示例：
  python bench_bestcheck_pt_vs_onnx.py --config best_check/config.json
  python bench_bestcheck_pt_vs_onnx.py --config best_check/config.json --device cuda --batch-size 256

说明：
- 默认只计“推理时间”（不包含加载 config/读取 H5/加载权重）。
- 如果要包含加载成本，加 --include-load。
- ONNX 推理依赖 onnxruntime：pip install onnxruntime (或 onnxruntime-gpu)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import s_maxvote


def _now() -> float:
    return time.perf_counter()


def _torch_sync_if_cuda(device: Union[str, torch.device, None]) -> None:
    if device is None:
        return
    if isinstance(device, torch.device):
        is_cuda = device.type == "cuda"
    else:
        is_cuda = str(device).lower().startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchResult:
    name: str
    seconds: float
    n: int

    @property
    def ms_per_sample(self) -> float:
        return (self.seconds / max(1, self.n)) * 1000.0

    @property
    def samples_per_s(self) -> float:
        return max(1e-12, self.n / max(1e-12, self.seconds))


def _print_pair(pt: BenchResult, ox: BenchResult) -> None:
    speedup = pt.seconds / max(1e-12, ox.seconds)
    print(
        "\n".join(
            [
                f"[PT ] {pt.name}: {pt.seconds:.4f}s  ({pt.ms_per_sample:.3f} ms/sample, {pt.samples_per_s:.1f} samples/s)",
                f"[ONNX] {ox.name}: {ox.seconds:.4f}s  ({ox.ms_per_sample:.3f} ms/sample, {ox.samples_per_s:.1f} samples/s)",
                f"speedup: {speedup:.2f}x",
            ]
        )
    )


def _limit(arr: np.ndarray, n: Optional[int]) -> np.ndarray:
    if n is None:
        return arr
    return arr[: int(n)]


def _bench_pt(
    *,
    cfg: Dict[str, Any],
    esm_x: torch.Tensor,
    prot_x: torch.Tensor,
    all_x: torch.Tensor,
    device: Union[str, torch.device, None],
    batch_size: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    specs = s_maxvote._build_models(cfg, esm_dim=int(esm_x.shape[1]), prot_dim=int(prot_x.shape[1]))
    for s in specs:
        s["model"].to(dev)

    def run_once() -> None:
        with torch.inference_mode():
            for s in specs:
                model = s["model"]
                ftype = str(s["feature_type"]).lower()
                if isinstance(model, s_maxvote.DPF):
                    x_in = (esm_x, prot_x)
                else:
                    if ftype == "esm2":
                        x_in = esm_x
                    elif ftype == "prot5":
                        x_in = prot_x
                    else:
                        x_in = all_x
                _ = s_maxvote._infer_prob1_batched(model, x_in, device=dev, batch_size=int(batch_size))

    for _ in range(int(warmup)):
        _torch_sync_if_cuda(dev)
        run_once()
        _torch_sync_if_cuda(dev)

    _torch_sync_if_cuda(dev)
    t0 = _now()
    for _ in range(int(iters)):
        run_once()
    _torch_sync_if_cuda(dev)
    t1 = _now()

    n = int(esm_x.shape[0]) * int(iters)
    return BenchResult(name="torch_ensemble(4models)", seconds=t1 - t0, n=n)


def _bench_onnx(
    *,
    cfg_path: str,
    onnx_dir: str,
    esm_x: torch.Tensor,
    prot_x: torch.Tensor,
    all_x: torch.Tensor,
    device: Union[str, torch.device, None],
    batch_size: int,
    warmup: int,
    iters: int,
    output_is_probs: bool,
) -> BenchResult:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    specs = cfg.get("models") or []
    if not isinstance(specs, list) or not specs:
        raise SystemExit("config.json models 为空")

    # build onnx paths according to exporter naming
    onnx_paths: List[Tuple[str, str]] = []
    for m in specs:
        mtype = str(m.get("type", "")).lower()
        ftype = str(m.get("feature_type", "")).lower()
        pth = str(m.get("path", ""))
        base = os.path.splitext(os.path.basename(pth))[0]
        if mtype in {"dpf", "fusion", "improveddualpathwayfusion"}:
            onnx_paths.append(("dpf", os.path.join(onnx_dir, f"dpf_{base}.onnx")))
        else:
            onnx_paths.append((ftype, os.path.join(onnx_dir, f"vfiter_{ftype}_{base}.onnx")))

    for _, p in onnx_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing onnx: {p} (run export_bestcheck_config_to_onnx.py first)")

    def run_once() -> None:
        for tag, p in onnx_paths:
            if tag == "dpf":
                _ = s_maxvote._infer_prob1_onnx_dpf(
                    p,
                    esm_x,
                    prot_x,
                    batch_size=int(batch_size),
                    device=device,
                    output_is_probs=bool(output_is_probs),
                )
            else:
                if tag == "esm2":
                    x = esm_x
                elif tag == "prot5":
                    x = prot_x
                else:
                    x = all_x
                _ = s_maxvote._infer_prob1_onnx_single(
                    p,
                    x,
                    batch_size=int(batch_size),
                    device=device,
                    output_is_probs=bool(output_is_probs),
                )

    # warmup
    for _ in range(int(warmup)):
        run_once()

    t0 = _now()
    for _ in range(int(iters)):
        run_once()
    t1 = _now()

    n = int(esm_x.shape[0]) * int(iters)
    return BenchResult(name="onnx_ensemble(4models)", seconds=t1 - t0, n=n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark best_check ensemble: PyTorch vs ONNX")
    ap.add_argument("--config", default=os.path.join("best_check", "config.json"))
    ap.add_argument("--onnx-dir", default=None, help="default: alongside config.json (best_check/onnx)")
    ap.add_argument("--device", default=None, help="cpu/cuda/auto; default auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--include-load", action="store_true")
    ap.add_argument("--onnx-logits", action="store_true", help="if ONNX outputs logits, will apply softmax in numpy")

    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    if args.onnx_dir is None:
        onnx_dir = os.path.join(os.path.dirname(cfg_path), "onnx")
    else:
        onnx_dir = os.path.abspath(args.onnx_dir)

    if args.include_load:
        t_load0 = _now()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    esm_h5 = cfg["test_esm_path"]
    prot_h5 = cfg["test_prot5_path"]
    seq_ids, esm_arr, prot_arr, _labels = s_maxvote._load_and_align(esm_h5, prot_h5)

    esm_arr = _limit(esm_arr, args.max_n)
    prot_arr = _limit(prot_arr, args.max_n)

    all_arr = np.concatenate([esm_arr, prot_arr], axis=1)

    esm_x = torch.from_numpy(esm_arr).float()
    prot_x = torch.from_numpy(prot_arr).float()
    all_x = torch.from_numpy(all_arr).float()

    if args.include_load:
        t_load1 = _now()
        print(f"load+data time: {t_load1 - t_load0:.4f}s (N={len(seq_ids) if args.max_n is None else int(args.max_n)})")

    device = args.device
    if device is not None and str(device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pt = _bench_pt(
        cfg=cfg,
        esm_x=esm_x,
        prot_x=prot_x,
        all_x=all_x,
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
    )

    # ONNX benchmark requires onnxruntime; s_maxvote will error with a clear message if missing
    ox = _bench_onnx(
        cfg_path=cfg_path,
        onnx_dir=onnx_dir,
        esm_x=esm_x,
        prot_x=prot_x,
        all_x=all_x,
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        output_is_probs=not bool(args.onnx_logits),
    )

    _print_pair(pt, ox)


if __name__ == "__main__":
    main()

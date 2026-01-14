#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark HAC inference: PyTorch (.pth) vs ONNXRuntime (.onnx).

适用：best_check12/hac/{esm,prot,both} 的对比学习模型（ProjectionHead + LinearClassifier）。

示例：
  python bench_hac_pt_vs_onnx.py --task both
  python bench_hac_pt_vs_onnx.py --task ensemble --device cuda --batch-size 512

说明：
- 默认只计“推理时间”（不包含加载权重/读取 H5）。
- 如果要包含加载成本，加 --include-load。
- ONNX 推理依赖 onnxruntime：pip install onnxruntime (或 onnxruntime-gpu)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from Constant import test_esm_path, test_prot_path

import m_maxvote


def _now() -> float:
    return time.perf_counter()


def _torch_sync_if_cuda(device: str) -> None:
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _try_import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore

        return ort
    except Exception:
        return None


def _ort_providers(device: str) -> List[str]:
    ort = _try_import_onnxruntime()
    if ort is None:
        return ["CPUExecutionProvider"]
    available = set(getattr(ort, "get_available_providers", lambda: [])())
    want_cuda = device.lower() in {"cuda", "cuda:0", "gpu", "auto"} and torch.cuda.is_available()
    if want_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


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


def _bench_torch_probs(
    *,
    head: torch.nn.Module,
    clf: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: str,
    warmup: int,
    iters: int,
) -> BenchResult:
    dev = m_maxvote._resolve_device(device)
    head.eval().to(dev)
    clf.eval().to(dev)

    x_t = torch.tensor(x, dtype=torch.float32)

    def run_once() -> None:
        with torch.inference_mode():
            for i in range(0, len(x_t), int(batch_size)):
                xb = x_t[i : i + int(batch_size)].to(dev)
                z = head(xb)
                logits = clf(z)
                _ = torch.softmax(logits, dim=1)

    # warmup
    for _ in range(int(warmup)):
        _torch_sync_if_cuda(device)
        run_once()
        _torch_sync_if_cuda(device)

    _torch_sync_if_cuda(device)
    t0 = _now()
    for _ in range(int(iters)):
        run_once()
    _torch_sync_if_cuda(device)
    t1 = _now()

    return BenchResult(name="torch", seconds=t1 - t0, n=int(len(x)) * int(iters))


def _bench_onnx_probs(
    *,
    onnx_path: str,
    x: np.ndarray,
    batch_size: int,
    device: str,
    warmup: int,
    iters: int,
) -> BenchResult:
    ort = _try_import_onnxruntime()
    if ort is None:
        raise SystemExit("onnxruntime 未安装：请 pip install onnxruntime 或 onnxruntime-gpu")

    sess = ort.InferenceSession(str(onnx_path), providers=_ort_providers(device))
    inp = sess.get_inputs()[0].name

    def run_once() -> None:
        for start in range(0, int(x.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(x.shape[0]))
            xb = x[start:end].astype(np.float32, copy=False)
            _ = sess.run(None, {inp: xb})[0]

    for _ in range(int(warmup)):
        run_once()

    t0 = _now()
    for _ in range(int(iters)):
        run_once()
    t1 = _now()

    return BenchResult(name="onnx", seconds=t1 - t0, n=int(len(x)) * int(iters))


def _limit(x: np.ndarray, n: Optional[int]) -> np.ndarray:
    if n is None:
        return x
    return x[: int(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark HAC: PyTorch vs ONNX")
    ap.add_argument("--task", choices=["both", "ensemble"], default="both", help="benchmark single both model or 3-model ensemble")
    ap.add_argument("--device", default="cpu", help="cpu/cuda/auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-n", type=int, default=None, help="limit samples for quicker benchmark")
    ap.add_argument("--include-load", action="store_true", help="include weight loading + H5 reading time")

    # allow override paths
    ap.add_argument("--esm-ckpt-dir", default=m_maxvote.DEFAULT_ESM_CKPT_DIR)
    ap.add_argument("--prot-ckpt-dir", default=m_maxvote.DEFAULT_PROT_CKPT_DIR)
    ap.add_argument("--both-ckpt-dir", default=m_maxvote.DEFAULT_BOTH_CKPT_DIR)
    ap.add_argument("--esm-path", default=test_esm_path)
    ap.add_argument("--prot-path", default=test_prot_path)

    # allow override onnx paths
    ap.add_argument("--esm-onnx", default=None)
    ap.add_argument("--prot-onnx", default=None)
    ap.add_argument("--both-onnx", default=None)

    args = ap.parse_args()

    if args.include_load:
        t_load0 = _now()

    if args.task == "both":
        seq_ids, x, _y = m_maxvote._load_embeddings("both", esm_path=args.esm_path, prot_path=args.prot_path)
        x = _limit(x, args.max_n)

        head, clf, _k = m_maxvote._load_model_pair(
            ckpt_dir=args.both_ckpt_dir,
            mode="both",
            input_dim=int(x.shape[1]),
            device=m_maxvote._resolve_device("cpu"),
        )
        # keep on CPU in object; we move to desired device during benchmark

        onnx_path = args.both_onnx or os.path.join(args.both_ckpt_dir, "both_infer.onnx")

        if args.include_load:
            t_load1 = _now()
            print(f"load+data time: {t_load1 - t_load0:.4f}s")

        pt = _bench_torch_probs(
            head=head,
            clf=clf,
            x=x,
            batch_size=args.batch_size,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        ox = _bench_onnx_probs(
            onnx_path=onnx_path,
            x=x,
            batch_size=args.batch_size,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        _print_pair(pt, ox)
        return

    # ensemble
    te_esm = m_maxvote._load_embeddings("esm", esm_path=args.esm_path, prot_path=args.prot_path)
    te_prot = m_maxvote._load_embeddings("prot", esm_path=args.esm_path, prot_path=args.prot_path)
    te_both = m_maxvote._load_embeddings("both", esm_path=args.esm_path, prot_path=args.prot_path)
    _ids, aligned = m_maxvote._align_by_seq_id(te_esm, te_prot, te_both)

    x_esm, _ = aligned["esm"]
    x_prot, _ = aligned["prot"]
    x_both, _ = aligned["both"]

    x_esm = _limit(x_esm, args.max_n)
    x_prot = _limit(x_prot, args.max_n)
    x_both = _limit(x_both, args.max_n)

    head_esm, clf_esm, _ = m_maxvote._load_model_pair(
        ckpt_dir=args.esm_ckpt_dir,
        mode="esm",
        input_dim=int(x_esm.shape[1]),
        device=m_maxvote._resolve_device("cpu"),
    )
    head_prot, clf_prot, _ = m_maxvote._load_model_pair(
        ckpt_dir=args.prot_ckpt_dir,
        mode="prot",
        input_dim=int(x_prot.shape[1]),
        device=m_maxvote._resolve_device("cpu"),
    )
    head_both, clf_both, _ = m_maxvote._load_model_pair(
        ckpt_dir=args.both_ckpt_dir,
        mode="both",
        input_dim=int(x_both.shape[1]),
        device=m_maxvote._resolve_device("cpu"),
    )

    esm_onnx = args.esm_onnx or os.path.join(args.esm_ckpt_dir, "esm_infer.onnx")
    prot_onnx = args.prot_onnx or os.path.join(args.prot_ckpt_dir, "prot_infer.onnx")
    both_onnx = args.both_onnx or os.path.join(args.both_ckpt_dir, "both_infer.onnx")

    if args.include_load:
        t_load1 = _now()
        print(f"load+data time: {t_load1 - t_load0:.4f}s")

    # time only the 3 forward passes (no voting) for clearer comparison
    pt_esm = _bench_torch_probs(head=head_esm, clf=clf_esm, x=x_esm, batch_size=args.batch_size, device=args.device, warmup=args.warmup, iters=args.iters)
    pt_prot = _bench_torch_probs(head=head_prot, clf=clf_prot, x=x_prot, batch_size=args.batch_size, device=args.device, warmup=args.warmup, iters=args.iters)
    pt_both = _bench_torch_probs(head=head_both, clf=clf_both, x=x_both, batch_size=args.batch_size, device=args.device, warmup=args.warmup, iters=args.iters)

    ox_esm = _bench_onnx_probs(onnx_path=esm_onnx, x=x_esm, batch_size=args.batch_size, device=args.device, warmup=args.warmup, iters=args.iters)
    ox_prot = _bench_onnx_probs(onnx_path=prot_onnx, x=x_prot, batch_size=args.batch_size, device=args.device, warmup=args.warmup, iters=args.iters)
    ox_both = _bench_onnx_probs(onnx_path=both_onnx, x=x_both, batch_size=args.batch_size, device=args.device, warmup=args.warmup, iters=args.iters)

    print("== ESM ==")
    _print_pair(pt_esm, ox_esm)
    print("== PROT ==")
    _print_pair(pt_prot, ox_prot)
    print("== BOTH ==")
    _print_pair(pt_both, ox_both)


if __name__ == "__main__":
    main()

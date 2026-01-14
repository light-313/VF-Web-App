#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Majority-vote ensemble prediction + single 'both' model prediction.

只暴露两个函数：
- predict_majority_vote_ensemble(): esm/prot/both 三模型 majority vote（并返回 soft 平均概率）
- predict_both_model(): 单 both 模型预测

尽量使用仓库当前默认路径/权重，不提供复杂 CLI 参数。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Constant import test_esm_path, test_prot_path
from embed_loader import EmbeddingLoader

# Reuse exact training-time module definitions.
from con_train import LinearClassifier, ProjectionHead


_HERE = os.path.dirname(os.path.abspath(__file__))

# Allow running as a script from any cwd (e.g. via VS Code wrappers)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 默认权重目录：优先使用仓库相对路径（更可移植）
DEFAULT_ESM_CKPT_DIR = os.path.join(_HERE, "best_check12", "hac", "esm")
DEFAULT_PROT_CKPT_DIR = os.path.join(_HERE, "best_check12", "hac", "prot")
DEFAULT_BOTH_CKPT_DIR = os.path.join(_HERE, "best_check12", "hac", "both")

# 默认推理设置
DEFAULT_BATCH_SIZE = 128
DEFAULT_DEVICE = "auto"


def _try_import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore

        return ort
    except Exception:
        return None


def _ort_providers(device: str) -> List[str]:
    """Choose ORT providers based on requested device and runtime availability."""
    ort = _try_import_onnxruntime()
    if ort is None:
        return ["CPUExecutionProvider"]
    available = set(getattr(ort, "get_available_providers", lambda: [])())
    want_cuda = device.lower() in {"cuda", "cuda:0", "gpu", "auto"} and torch.cuda.is_available()
    if want_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def _infer_proj_head_hparams(state: Dict[str, torch.Tensor], input_dim: int) -> Tuple[int, int, int, int]:
    """Infer hidden/out_dim/rank/steps from ProjectionHead state dict."""
    if "net.0.weight" not in state:
        raise KeyError("Unexpected ProjectionHead state_dict: missing 'net.0.weight'")
    if "net.5.weight" not in state:
        raise KeyError("Unexpected ProjectionHead state_dict: missing 'net.5.weight'")

    w0 = state["net.0.weight"]  # [hidden, in_dim]
    if int(w0.shape[1]) != int(input_dim):
        raise ValueError(f"Input dim mismatch: data dim={input_dim}, but head expects in_dim={int(w0.shape[1])}")

    hidden = int(w0.shape[0])
    out_dim = int(state["net.5.weight"].shape[0])

    step0 = "net.1.delta_calculations.0.0.weight"
    if step0 not in state:
        raise KeyError("Unexpected ProjectionHead state_dict: missing IterLowRankBlock keys")
    rank = int(state[step0].shape[0])

    steps = 0
    while f"net.1.delta_calculations.{steps}.0.weight" in state:
        steps += 1
    if steps <= 0:
        raise ValueError("Failed to infer steps for IterLowRankBlock")

    return hidden, out_dim, rank, steps


def _infer_linear_hparams(state: Dict[str, torch.Tensor]) -> Tuple[int, int, Optional[int], bool]:
    """Infer LinearClassifier (hidden/use_layernorm/num_classes/in_dim) from state dict."""
    # hidden is None => self.net = nn.Linear
    if "net.weight" in state and "net.bias" in state:
        w = state["net.weight"]
        return int(w.shape[1]), int(w.shape[0]), None, False

    # Sequential
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
        keys_preview = ", ".join(list(state.keys())[:8])
        raise KeyError(f"Unexpected LinearClassifier state_dict keys. Preview: {keys_preview}")

    w1 = state[first_linear_key]  # [hidden, in_dim]
    w2 = state[last_linear_key]  # [num_classes, hidden]

    hidden = int(w1.shape[0])
    in_dim = int(w1.shape[1])
    num_classes = int(w2.shape[0])
    return in_dim, num_classes, hidden, use_layernorm


def _load_embeddings(mode: str, esm_path: str, prot_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    loader = EmbeddingLoader(esm_path=esm_path, prot_path=prot_path)
    seq_ids, features, labels = loader.load_embeddings(mode=mode)
    return list(seq_ids), np.asarray(features, dtype=np.float32), np.asarray(labels)


def _align_by_seq_id(
    esm: Tuple[List[str], np.ndarray, np.ndarray],
    prot: Tuple[List[str], np.ndarray, np.ndarray],
    both: Tuple[List[str], np.ndarray, np.ndarray],
) -> Tuple[List[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Return common seq_ids (in ESM order) and aligned arrays per mode."""
    esm_ids, esm_x, esm_y = esm
    prot_ids, prot_x, prot_y = prot
    both_ids, both_x, both_y = both

    esm_map = {sid: i for i, sid in enumerate(esm_ids)}
    prot_map = {sid: i for i, sid in enumerate(prot_ids)}
    both_map = {sid: i for i, sid in enumerate(both_ids)}

    common: List[str] = [sid for sid in esm_ids if sid in prot_map and sid in both_map]
    if not common:
        raise ValueError("No common seq_id across esm/prot/both; cannot ensemble.")

    esm_idx = [esm_map[sid] for sid in common]
    prot_idx = [prot_map[sid] for sid in common]
    both_idx = [both_map[sid] for sid in common]

    esm_x2, esm_y2 = esm_x[esm_idx], esm_y[esm_idx]
    prot_x2, prot_y2 = prot_x[prot_idx], prot_y[prot_idx]
    both_x2, both_y2 = both_x[both_idx], both_y[both_idx]

    if not (np.array_equal(esm_y2, prot_y2) and np.array_equal(esm_y2, both_y2)):
        # Keep going; return ESM labels as reference.
        pass

    return common, {
        "esm": (esm_x2, esm_y2),
        "prot": (prot_x2, prot_y2),
        "both": (both_x2, both_y2),
    }


@torch.no_grad()
def _predict_probs(
    head: nn.Module,
    clf: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    head.eval().to(device)
    clf.eval().to(device)

    x_t = torch.tensor(x, dtype=torch.float32)
    probs: List[torch.Tensor] = []

    for i in range(0, len(x_t), int(batch_size)):
        xb = x_t[i : i + int(batch_size)].to(device)
        z = head(xb)
        logits = clf(z)
        probs.append(torch.softmax(logits, dim=1).detach().cpu())

    return torch.cat(probs, dim=0).numpy()


def _predict_probs_onnx(
    onnx_path: str,
    x: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    ort = _try_import_onnxruntime()
    if ort is None:
        raise ModuleNotFoundError("onnxruntime 未安装：请 pip install onnxruntime (或 onnxruntime-gpu)")

    sess = ort.InferenceSession(str(onnx_path), providers=_ort_providers(device))
    inp_name = sess.get_inputs()[0].name

    outs: List[np.ndarray] = []
    n = int(x.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        xb = x[start:end].astype(np.float32, copy=False)
        y = sess.run(None, {inp_name: xb})[0]
        outs.append(np.asarray(y, dtype=np.float32))
    return np.concatenate(outs, axis=0)


def _load_model_pair(
    ckpt_dir: str,
    mode: str,
    input_dim: int,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, int]:
    proj_path = os.path.join(ckpt_dir, f"proj_head_{mode}.pth")
    lin_path = os.path.join(ckpt_dir, f"linear_{mode}.pth")
    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Missing ProjectionHead ckpt: {proj_path}")
    if not os.path.exists(lin_path):
        raise FileNotFoundError(f"Missing LinearClassifier ckpt: {lin_path}")

    proj_state = _strip_module_prefix(torch.load(proj_path, map_location="cpu"))
    hidden, out_dim, rank, steps = _infer_proj_head_hparams(proj_state, input_dim=input_dim)
    head = ProjectionHead(
        in_dim=int(input_dim),
        hidden=int(hidden),
        out_dim=int(out_dim),
        rank=int(rank),
        steps=int(steps),
        dropout=0.1,
    )
    head.load_state_dict(proj_state, strict=True)

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

    head.to(device).eval()
    clf.to(device).eval()
    return head, clf, int(num_classes)


def _majority_vote(preds_list: List[np.ndarray], probs_list: List[np.ndarray]) -> np.ndarray:
    """Majority vote; ties broken by highest mean probability among tied classes."""
    preds = np.stack(preds_list, axis=0)  # [M, N]
    m, n = preds.shape
    if m <= 0:
        raise ValueError("preds_list is empty")
    c = int(probs_list[0].shape[1])
    mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)  # [N, C]

    out = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        votes = np.bincount(preds[:, i], minlength=c)
        top = np.flatnonzero(votes == votes.max())
        if len(top) == 1:
            out[i] = int(top[0])
        else:
            out[i] = int(top[np.argmax(mean_probs[i, top])])
    return out


def predict_majority_vote_ensemble12(
    *,
    esm_ckpt_dir: str = DEFAULT_ESM_CKPT_DIR,
    prot_ckpt_dir: str = DEFAULT_PROT_CKPT_DIR,
    both_ckpt_dir: str = DEFAULT_BOTH_CKPT_DIR,
    esm_path: str = test_esm_path,
    prot_path: str = test_prot_path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    prefer_onnx: bool = False,
    esm_onnx_path: Optional[str] = None,
    prot_onnx_path: Optional[str] = None,
    both_onnx_path: Optional[str] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Predict with (esm/prot/both) majority-vote ensemble.

    Returns: (seq_ids, ensemble_pred, ensemble_probs)
    - ensemble_pred: [N] int64
    - ensemble_probs: [N, C] float32 (simple average of 3 model probabilities)

    Notes:
    - Alignment is done by seq_id across the three modes.
    - Probabilities are computed per model then combined.
    """

    dev = _resolve_device(device)

    te_esm = _load_embeddings("esm", esm_path=esm_path, prot_path=prot_path)
    te_prot = _load_embeddings("prot", esm_path=esm_path, prot_path=prot_path)
    te_both = _load_embeddings("both", esm_path=esm_path, prot_path=prot_path)
    seq_ids, aligned = _align_by_seq_id(te_esm, te_prot, te_both)

    x_esm, _y = aligned["esm"]
    x_prot, _ = aligned["prot"]
    x_both, _ = aligned["both"]

    # ONNX priority: explicit path > default <mode>_infer.onnx (in ckpt_dir)
    esm_onnx = esm_onnx_path or os.path.join(esm_ckpt_dir, "esm_infer.onnx")
    prot_onnx = prot_onnx_path or os.path.join(prot_ckpt_dir, "prot_infer.onnx")
    both_onnx = both_onnx_path or os.path.join(both_ckpt_dir, "both_infer.onnx")

    use_onnx = bool(prefer_onnx) and all(os.path.exists(p) for p in [esm_onnx, prot_onnx, both_onnx])

    if use_onnx:
        p_esm = _predict_probs_onnx(esm_onnx, x_esm, batch_size=int(batch_size), device=device)
        p_prot = _predict_probs_onnx(prot_onnx, x_prot, batch_size=int(batch_size), device=device)
        p_both = _predict_probs_onnx(both_onnx, x_both, batch_size=int(batch_size), device=device)
        k1 = int(p_esm.shape[1])
        k2 = int(p_prot.shape[1])
        k3 = int(p_both.shape[1])
    else:
        head_esm, clf_esm, k1 = _load_model_pair(ckpt_dir=esm_ckpt_dir, mode="esm", input_dim=int(x_esm.shape[1]), device=dev)
        head_prot, clf_prot, k2 = _load_model_pair(ckpt_dir=prot_ckpt_dir, mode="prot", input_dim=int(x_prot.shape[1]), device=dev)
        head_both, clf_both, k3 = _load_model_pair(ckpt_dir=both_ckpt_dir, mode="both", input_dim=int(x_both.shape[1]), device=dev)

        p_esm = _predict_probs(head_esm, clf_esm, x_esm, batch_size=int(batch_size), device=dev)
        p_prot = _predict_probs(head_prot, clf_prot, x_prot, batch_size=int(batch_size), device=dev)
        p_both = _predict_probs(head_both, clf_both, x_both, batch_size=int(batch_size), device=dev)

    if not (k1 == k2 == k3):
        raise ValueError(f"Num classes mismatch across models: esm={k1}, prot={k2}, both={k3}")

    pred = _majority_vote(
        preds_list=[p_esm.argmax(axis=1), p_prot.argmax(axis=1), p_both.argmax(axis=1)],
        probs_list=[p_esm, p_prot, p_both],
    )

    probs = np.mean(np.stack([p_esm, p_prot, p_both], axis=0), axis=0).astype(np.float32)
    return seq_ids, pred.astype(np.int64), probs


def predict_both_model12(
    *,
    both_ckpt_dir: str = DEFAULT_BOTH_CKPT_DIR,
    esm_path: str = test_esm_path,
    prot_path: str = test_prot_path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    prefer_onnx: bool = False,
    onnx_path: Optional[str] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Predict with single 'both' contrastive model.

    Returns: (seq_ids, pred, probs)
    - pred: [N] int64
    - probs: [N, C] float32
    """

    dev = _resolve_device(device)

    seq_ids, x, _y = _load_embeddings("both", esm_path=esm_path, prot_path=prot_path)

    onnx_p = onnx_path or os.path.join(both_ckpt_dir, "both_infer.onnx")
    if bool(prefer_onnx) and os.path.exists(onnx_p):
        probs = _predict_probs_onnx(onnx_p, x, batch_size=int(batch_size), device=device).astype(np.float32)
    else:
        head, clf, _k = _load_model_pair(ckpt_dir=both_ckpt_dir, mode="both", input_dim=int(x.shape[1]), device=dev)
        probs = _predict_probs(head, clf, x, batch_size=int(batch_size), device=dev).astype(np.float32)
    pred = probs.argmax(axis=1).astype(np.int64)
    return seq_ids, pred, probs

__all__ = [
    "predict_majority_vote_ensemble12",
    "predict_both_model12",
]

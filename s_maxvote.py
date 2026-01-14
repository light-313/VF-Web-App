import argparse
import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score)

# Allow running as a script from any cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
class EmbeddingLoader:
    """
    简化版嵌入加载器，支持选择读取ESM、ProtT5或两者组合
    """
    
    def __init__(self, esm_path: Optional[str] = None, prot_path: Optional[str] = None):
        """
        初始化嵌入加载器
        
        Args:
            esm_path: ESM嵌入文件路径
            prot_path: ProtT5嵌入文件路径
        """
        self.esm_path = esm_path
        self.prot_path = prot_path
        
    def load_embeddings(self, mode: str = "esm") -> Tuple[List[str], np.ndarray, List[int]]:
        """
        加载嵌入向量
        
        Args:
            mode: 加载模式 ("esm", "prot", "both")
            
        Returns:
            tuple: (序列ID列表, 特征矩阵, 标签列表)
        """
        if mode == "esm":
            if not self.esm_path:
                raise ValueError("ESM路径未设置")
            return self._load_single_embedding(self.esm_path)
        elif mode == "prot":
            if not self.prot_path:
                raise ValueError("ProtT5路径未设置")
            return self._load_single_embedding(self.prot_path)
        elif mode == "both":
            if not self.esm_path or not self.prot_path:
                raise ValueError("ESM和ProtT5路径都需要设置")
            return self._load_combined_embeddings()
        else:
            raise ValueError("mode必须是'esm'、'prot'或'both'之一")
    
    def _load_single_embedding(self, file_path: str) -> Tuple[List[str], np.ndarray, List[int]]:
        """
        加载单个嵌入文件
        
        Args:
            file_path: 嵌入文件路径
            
        Returns:
            tuple: (序列ID列表, 特征矩阵, 标签列表)
        """
        seq_ids, _, features, labels = self._read_records(file_path)
        return seq_ids, features, labels

    def _read_records(self, file_path: str) -> Tuple[List[str], List[Optional[str]], np.ndarray, List[int]]:
        """从单个 H5 文件读取记录，保证顺序确定。返回:
        - seq_ids: HDF5 key（用于稳定对齐/拼接）
        - raw_seqs: 可选的真实序列字符串（若文件中不存在则为 None）
        - features: 特征矩阵
        - labels: 标签列表
        """
        seq_ids: List[str] = []
        raw_seqs: List[Optional[str]] = []
        features: List[np.ndarray] = []
        labels: List[int] = []

        with h5py.File(file_path, 'r') as f:
            has_sequences = 'sequences' in f
            for seq_id in sorted(f['embeddings'].keys()):
                seq_ids.append(seq_id)
                if has_sequences:
                    raw = f['sequences'][seq_id][()]
                    raw_seqs.append(raw.decode('ascii') if isinstance(raw, (bytes, bytearray)) else str(raw))
                else:
                    raw_seqs.append(None)

                features.append(f['embeddings'][seq_id][:])

                label = f['labels'][seq_id][()]
                if isinstance(label, np.integer):
                    label = int(label)
                labels.append(label)

        return seq_ids, raw_seqs, np.stack(features), labels
    
    def _load_combined_embeddings(self) -> Tuple[List[str], np.ndarray, List[int]]:
        """
        加载并合并ESM和ProtT5嵌入
        
        Returns:
            tuple: (序列ID列表, 特征矩阵, 标签列表)
        """
        # 加载 ESM / Prot 记录（顺序确定）
        esm_ids, esm_raw_seqs, esm_features, esm_labels = self._read_records(self.esm_path)
        prot_ids, prot_raw_seqs, prot_features, prot_labels = self._read_records(self.prot_path)

        prot_id_to_idx = {seq_id: idx for idx, seq_id in enumerate(prot_ids)}
        common_ids = [seq_id for seq_id in esm_ids if seq_id in prot_id_to_idx]

        pairs: List[tuple[int, int]] = []

        if common_ids:
            # 优先使用 HDF5 key (seq_id) 对齐：唯一且不会出现 raw_seq 重复覆盖的问题
            esm_id_to_idx = {seq_id: idx for idx, seq_id in enumerate(esm_ids)}
            for seq_id in common_ids:
                pairs.append((esm_id_to_idx[seq_id], prot_id_to_idx[seq_id]))
        else:
            # 如果两边 seq_id 不一致，则回退到 raw sequence 对齐（确定性 + 处理重复序列）
            if all(s is None for s in esm_raw_seqs) or all(s is None for s in prot_raw_seqs):
                raise ValueError("ESM 和 ProtT5 的 seq_id 无交集，且至少一侧缺少 sequences 字段，无法安全对齐")

            prot_queues: dict[str, deque[int]] = {}
            for idx, raw_seq in enumerate(prot_raw_seqs):
                if raw_seq is None:
                    continue
                prot_queues.setdefault(raw_seq, deque()).append(idx)

            for esm_idx, raw_seq in enumerate(esm_raw_seqs):
                if raw_seq is None:
                    continue
                q = prot_queues.get(raw_seq)
                if q:
                    prot_idx = q.popleft()
                    pairs.append((esm_idx, prot_idx))

        if not pairs:
            raise ValueError("ESM和ProtT5嵌入中没有可对齐的序列")

        # 标签一致性检查：避免对每个样本打印告警（会刷屏），改为汇总处理。
        esm_pair_labels = [int(esm_labels[e]) for e, _ in pairs]
        prot_pair_labels = [int(prot_labels[p]) for _, p in pairs]
        mismatches = [i for i, (a, b) in enumerate(zip(esm_pair_labels, prot_pair_labels)) if a != b]
        if mismatches:
            # 如果是二分类且完全相反（100% mismatch），自动翻转 prot 标签用于一致性。
            uniq = set(esm_pair_labels) | set(prot_pair_labels)
            if uniq.issubset({0, 1}) and len(mismatches) == len(pairs):
                prot_labels = [1 - int(x) for x in prot_labels]
            else:
                # 提供少量样例帮助定位数据问题
                show = mismatches[:10]
                examples = [(esm_ids[pairs[i][0]], esm_pair_labels[i], prot_pair_labels[i]) for i in show]
                raise ValueError(
                    f"ESM/ProtT5 标签不一致: {len(mismatches)}/{len(pairs)}；例如: {examples}. "
                    "请检查生成 h5 的标签映射是否一致（或是否有一侧标签被翻转）。"
                )

        combined_features: List[np.ndarray] = []
        combined_labels: List[int] = []
        out_ids: List[str] = []

        for esm_idx, prot_idx in pairs:
            out_ids.append(esm_ids[esm_idx])
            combined_features.append(np.concatenate([esm_features[esm_idx], prot_features[prot_idx]], axis=0))
            combined_labels.append(esm_labels[esm_idx])

        return out_ids, np.stack(combined_features), combined_labels

from model_type import DPF, VFITER


def _try_import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore

        return ort
    except Exception:
        return None


def _ort_providers(device: Union[str, torch.device, None]) -> List[str]:
    ort = _try_import_onnxruntime()
    if ort is None:
        return ["CPUExecutionProvider"]

    available = set(getattr(ort, "get_available_providers", lambda: [])())

    want_cuda = False
    if device is None:
        want_cuda = torch.cuda.is_available()
    elif isinstance(device, torch.device):
        want_cuda = device.type == "cuda"
    else:
        want_cuda = str(device).lower() in {"cuda", "cuda:0", "gpu", "auto"} and torch.cuda.is_available()

    if want_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _infer_prob1_onnx_single(
    onnx_path: str,
    features: torch.Tensor,
    batch_size: int,
    device: Union[str, torch.device, None],
    output_is_probs: bool = True,
) -> np.ndarray:
    ort = _try_import_onnxruntime()
    if ort is None:
        raise ModuleNotFoundError("onnxruntime 未安装：请 pip install onnxruntime (或 onnxruntime-gpu)")

    sess = ort.InferenceSession(str(onnx_path), providers=_ort_providers(device))
    inp_name = sess.get_inputs()[0].name

    n = int(features.shape[0])
    out = np.empty((n,), dtype=np.float32)
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        xb = features[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
        y = sess.run(None, {inp_name: xb})[0]
        y = np.asarray(y, dtype=np.float32)
        if not output_is_probs:
            y = _softmax_np(y, axis=1)
        out[start:end] = y[:, 1]
    return out


def _infer_prob1_onnx_dpf(
    onnx_path: str,
    esm_features: torch.Tensor,
    prot_features: torch.Tensor,
    batch_size: int,
    device: Union[str, torch.device, None],
    output_is_probs: bool = True,
) -> np.ndarray:
    ort = _try_import_onnxruntime()
    if ort is None:
        raise ModuleNotFoundError("onnxruntime 未安装：请 pip install onnxruntime (或 onnxruntime-gpu)")

    sess = ort.InferenceSession(str(onnx_path), providers=_ort_providers(device))
    inps = sess.get_inputs()
    if len(inps) != 2:
        raise ValueError(f"DPF ONNX expects 2 inputs, got {len(inps)}")
    name0 = inps[0].name
    name1 = inps[1].name

    n = int(esm_features.shape[0])
    out = np.empty((n,), dtype=np.float32)
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        e = esm_features[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
        p = prot_features[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
        y = sess.run(None, {name0: e, name1: p})[0]
        y = np.asarray(y, dtype=np.float32)
        if not output_is_probs:
            y = _softmax_np(y, axis=1)
        out[start:end] = y[:, 1]
    return out


def calculate_metrics(labels, predictions, scores=None):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sn / (precision + sn) if (precision + sn) > 0 else 0
    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    auc_score = 0
    aupr_score = 0
    if scores is not None:
        try:
            auc_score = roc_auc_score(labels, scores)
            p, r, _ = precision_recall_curve(labels, scores)
            aupr_score = auc(r, p)
        except Exception:
            pass

    return {
        "sn": sn * 100,
        "sp": sp * 100,
        "acc": acc * 100,
        "f1": f1 * 100,
        "mcc": mcc * 100,
        "auc": auc_score * 100,
        "aupr": aupr_score * 100,
    }


def _load_and_align(esm_h5: str, prot_h5: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """返回共同 seq_id 的对齐结果: (ids, esm_features, prot_features, labels[按ESM定义])."""
    loader = EmbeddingLoader(esm_path=esm_h5, prot_path=prot_h5)
    esm_ids, _, esm_features, esm_labels = loader._read_records(esm_h5)
    prot_ids, _, prot_features, prot_labels = loader._read_records(prot_h5)

    prot_idx = {sid: i for i, sid in enumerate(prot_ids)}
    pairs = [(i, prot_idx[sid]) for i, sid in enumerate(esm_ids) if sid in prot_idx]
    if not pairs:
        raise ValueError("ESM/ProtT5 没有可对齐的 seq_id")

    out_ids: List[str] = []
    out_esm: List[np.ndarray] = []
    out_prot: List[np.ndarray] = []
    out_labels: List[int] = []

    esm_pair_labels = [int(esm_labels[e]) for e, _ in pairs]
    prot_pair_labels = [int(prot_labels[p]) for _, p in pairs]
    mismatches = sum(1 for a, b in zip(esm_pair_labels, prot_pair_labels) if a != b)

    # 与 val_model.py/embed_loader.py 的策略一致：若二分类且 100% 反转，自动翻转 prot 标签（仅用于一致性检查）。
    uniq = set(esm_pair_labels) | set(prot_pair_labels)
    if mismatches and uniq.issubset({0, 1}) and mismatches == len(pairs):
        prot_labels = [1 - int(x) for x in prot_labels]

    for e, p in pairs:
        out_ids.append(esm_ids[e])
        out_esm.append(esm_features[e])
        out_prot.append(prot_features[p])
        out_labels.append(int(esm_labels[e]))

    return out_ids, np.stack(out_esm), np.stack(out_prot), np.asarray(out_labels, dtype=np.int64)


def _build_models(cfg: Dict[str, Any], esm_dim: int, prot_dim: int):
    """从 config.json 构建并加载权重。

    返回 list[dict]，每项包含: name/model/feature_type/weight/path/type。
    """
    specs: List[Dict[str, Any]] = []
    for i, m in enumerate(cfg.get("models", [])):
        mtype = str(m.get("type", "")).lower()
        ftype = str(m.get("feature_type", "")).lower()
        weight = float(m.get("weight", 1.0))
        path = m.get("path")
        if not path:
            raise ValueError(f"models[{i}] 缺少 path")

        if mtype in {"improveddualpathwayfusion", "dpf", "fusion"}:
            model = DPF(
                esm_dim=esm_dim,
                prot5_dim=prot_dim,
                hidden_dim=int(m["hidden_dim"]),
                num_layers=int(m["num_layers"]),
                num_classes=2,
                rank=int(m["rank"]),
                steps=int(m["steps"]),
                dropout=float(m["dropout"]),
            )
            name = f"fusion_{i}"
        elif mtype in {"delta", "vfiter"}:
            model = VFITER(
                input_dim=int(m["input_dim"]),
                hidden_dim=int(m["hidden_dim"]),
                num_layers=int(m["num_layers"]),
                num_classes=2,
                rank=int(m["rank"]),
                steps=int(m["steps"]),
                dropout=float(m["dropout"]),
            )
            name = f"delta_{ftype}_{i}"
        else:
            raise ValueError(f"不支持的模型 type: {m.get('type')}")

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        specs.append({"name": name, "model": model, "feature_type": ftype, "weight": weight, "path": path, "type": mtype})

    if not specs:
        raise ValueError("config.json 中 models 为空")

    return specs


def _infer_prob1_batched(
    model: torch.nn.Module,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """返回每个样本的 P(class=1)，按 batch 切片推理（不依赖 DataLoader）。"""
    model.eval()
    n = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
    out = np.empty((n,), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            lengths = torch.ones((end - start,), dtype=torch.long, device=device)
            if isinstance(x, tuple):
                xb = (x[0][start:end].to(device), x[1][start:end].to(device))
            else:
                xb = x[start:end].to(device)
            logits = model(xb, lengths)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            out[start:end] = p1.detach().cpu().numpy()
    return out


def run_majority_vote_4models(
    config: Union[str, Dict[str, Any]],
    batch_size: int = 64,
    device: Union[str, torch.device, None] = None,
    out_path: str | None = None,
    save_per_model_prob: bool = True,
    use_onnx: bool = False,
    onnx_dir: str | None = None,
    onnx_output_is_probs: bool = True,
) -> Dict[str, Any]:
    """读取 4 种模型配置并进行加权 majority vote 推理。

    - config: config.json 路径或 dict
    - batch_size: 推理 batch
    - device: "cuda"/"cpu" 或 torch.device，None 则自动
    - out_path: 若提供则保存 json
    - save_per_model_prob: 是否在 json 里保存每个模型的 prob1（体积较大）
    """
    cfg: Dict[str, Any]
    if isinstance(config, str):
        with open(config, "r") as f:
            cfg = json.load(f)
    else:
        cfg = config

    # default ONNX dir: alongside config.json (best_check/onnx)
    if onnx_dir is None and isinstance(config, str):
        onnx_dir = os.path.join(os.path.dirname(os.path.abspath(config)), "onnx")
    if onnx_dir is None:
        onnx_dir = os.path.join(_THIS_DIR, "best_check", "onnx")

    esm_h5 = cfg["test_esm_path"]
    prot_h5 = cfg["test_prot5_path"]
    seq_ids, esm_arr, prot_arr, labels = _load_and_align(esm_h5, prot_h5)

    esm_dim = int(esm_arr.shape[-1])
    prot_dim = int(prot_arr.shape[-1])
    all_arr = np.concatenate([esm_arr, prot_arr], axis=1)

    # 预先转成 CPU tensor（切片更轻），每个 batch 再搬到 device
    esm_x = torch.from_numpy(esm_arr).float()
    prot_x = torch.from_numpy(prot_arr).float()
    all_x = torch.from_numpy(all_arr).float()

    specs = _build_models(cfg, esm_dim=esm_dim, prot_dim=prot_dim)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device_t = torch.device(device)
    else:
        device_t = device

    for s in specs:
        s["model"].to(device_t)

    per_model_prob1: Dict[str, np.ndarray] = {}
    weights: List[float] = []
    names: List[str] = []

    for s in specs:
        model = s["model"]
        name = str(s["name"])
        ftype = str(s["feature_type"]).lower()
        weight = float(s["weight"])
        model_path = str(s.get("path", ""))

        if isinstance(model, DPF):
            x_in: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = (esm_x, prot_x)
        else:
            if ftype == "esm2":
                x_in = esm_x
            elif ftype == "prot5":
                x_in = prot_x
            elif ftype == "all":
                x_in = all_x
            else:
                raise ValueError(f"不支持的 feature_type: {ftype} (支持: esm2/prot5/all)")

        if use_onnx:
            # 如果 config.json 的 path 已经是 .onnx，则直接用；否则按 exporter 的命名规则在 onnx_dir 中查找。
            candidate: Optional[str] = None
            if model_path.lower().endswith(".onnx") and os.path.exists(model_path):
                candidate = model_path
            else:
                base = os.path.splitext(os.path.basename(model_path))[0]
                if isinstance(model, DPF):
                    candidate = os.path.join(str(onnx_dir), f"dpf_{base}.onnx")
                else:
                    candidate = os.path.join(str(onnx_dir), f"vfiter_{ftype}_{base}.onnx")

            if not candidate or not os.path.exists(candidate):
                raise FileNotFoundError(f"use_onnx=True 但找不到 ONNX 文件: {candidate}")

            if isinstance(model, DPF):
                per_model_prob1[name] = _infer_prob1_onnx_dpf(
                    candidate,
                    esm_x,
                    prot_x,
                    batch_size=batch_size,
                    device=device,
                    output_is_probs=bool(onnx_output_is_probs),
                )
            else:
                if not isinstance(x_in, torch.Tensor):
                    raise TypeError("VFITER expects single Tensor input")
                per_model_prob1[name] = _infer_prob1_onnx_single(
                    candidate,
                    x_in,
                    batch_size=batch_size,
                    device=device,
                    output_is_probs=bool(onnx_output_is_probs),
                )
        else:
            per_model_prob1[name] = _infer_prob1_batched(model, x_in, device=device_t, batch_size=batch_size)
        weights.append(weight)
        names.append(name)

    weights_np = np.asarray(weights, dtype=np.float32)
    votes_pos = np.zeros(len(seq_ids), dtype=np.float32)
    votes_neg = np.zeros(len(seq_ids), dtype=np.float32)
    for w, name in zip(weights_np, names):
        pred = (per_model_prob1[name] >= 0.5).astype(np.int64)
        votes_pos += w * (pred == 1)
        votes_neg += w * (pred == 0)

    final_pred = (votes_pos >= votes_neg).astype(np.int64)

    wsum = float(weights_np.sum()) if float(weights_np.sum()) > 0 else 1.0
    ensemble_prob1 = sum((w * per_model_prob1[name] for w, name in zip(weights_np, names))) / wsum

    metrics = calculate_metrics(labels, final_pred, scores=ensemble_prob1)
    result = {
        "test_esm_path": esm_h5,
        "test_prot5_path": prot_h5,
        # 便于下游脚本直接使用的紧凑输出（用户诉求：返回 seq_ids, pred, probs）
        "seq_ids": list(seq_ids),
        "pred": final_pred.astype(int).tolist(),
        "probs": ensemble_prob1.astype(float).tolist(),
        "models": [{"name": n, "weight": float(w)} for n, w in zip(names, weights_np.tolist())],
        "metrics": metrics,
        "results": [
            {
                "seq_id": sid,
                "label": int(lab),
                "pred": int(pred),
                "prob1": float(prob),
                "votes_pos": float(vp),
                "votes_neg": float(vn),
            }
            for sid, lab, pred, prob, vp, vn in zip(
                seq_ids, labels.tolist(), final_pred.tolist(), ensemble_prob1.tolist(), votes_pos.tolist(), votes_neg.tolist()
            )
        ],
    }
    if save_per_model_prob:
        result["per_model_prob1"] = {k: v.tolist() for k, v in per_model_prob1.items()}

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    ap = argparse.ArgumentParser(description="4-model ensemble majority vote inference")
    ap.add_argument("--config", default="./best_check\config.json")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--use-onnx", action="store_true", help="use ONNX models for inference (requires onnxruntime)")
    ap.add_argument("--onnx-dir", default=None, help="directory containing exported ONNX models (default: alongside config.json)")
    ap.add_argument("--onnx-logits", action="store_true", help="ONNX outputs logits (will apply softmax)")
    args = ap.parse_args()

    bs = args.batch_size or 64
    cfg_path = args.config
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    save_results = bool(cfg.get("save_results", False))
    out_path = args.out
    if out_path is None and save_results:
        out_path = os.path.join(os.path.dirname(cfg_path), "ensemble_majority_vote_results.json")

    res = run_majority_vote_4models(
        cfg_path,
        batch_size=bs,
        out_path=out_path,
        use_onnx=bool(args.use_onnx),
        onnx_dir=args.onnx_dir,
        onnx_output_is_probs=not bool(args.onnx_logits),
    )
    m = res["metrics"]
    print(
        " ".join(
            [
                f"SN:{m['sn']:.2f}%",
                f"SP:{m['sp']:.2f}%",
                f"ACC:{m['acc']:.2f}%",
                f"F1:{m['f1']:.2f}%",
                f"MCC:{m['mcc']:.2f}%",
                f"AUC:{m['auc']:.2f}%",
                f"AUPR:{m['aupr']:.2f}%",
            ]
        )
    )
    if out_path:
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

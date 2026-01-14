import os
import warnings
import torch
from model_type import *
from embed_loader import EmbeddingLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
# ==== 读取最优超参 ====
best_result_path = "/root/VF-pred/best_check/2vf/1_fusion_config.json"
with open(best_result_path, "r") as f:
    best_config = json.load(f)
best_model_path = "/root/VF-pred/best_check/2vf/1_fusion_867.pth" #9

# ==== 路径配置 ====
test_esm_path = '/root/VF-pred/fea_extraction/2esm_test.h5'
test_prot5_path = "/root/VF-pred/fea_extraction/2prot_test.h5"
feature_type_config="esm2+prot5" # prot5 esm2+prot5 esm2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_type = best_config['classifier_type'] 
is_fusion_model = classifier_type.lower() == "dpf"


def _feature_mode_from_config(feature_type: str) -> str:
    ft = (feature_type or "").lower()
    if ft == "prot5":
        return "prot"
    if ft == "esm2":
        return "esm"
    if ft in {"esm2+prot5", "prot5+esm2", "both"}:
        return "both"
    raise ValueError(f"不支持的 feature_type_config: {feature_type}")


class ArrayEmbeddingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: list[int] | np.ndarray):
        if features.ndim == 1:
            features = features[:, None]
        self.features = features
        self.labels = np.asarray(labels, dtype=np.int64)

        if len(self.features) != len(self.labels):
            raise ValueError(f"features/labels 数量不一致: {len(self.features)} vs {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def array_collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)
    lengths = torch.ones(len(batch), dtype=torch.long)
    return features, labels, lengths


class DualArrayEmbeddingDataset(Dataset):
    def __init__(self, esm_features: np.ndarray, prot5_features: np.ndarray, labels: list[int] | np.ndarray):
        if esm_features.ndim == 1:
            esm_features = esm_features[:, None]
        if prot5_features.ndim == 1:
            prot5_features = prot5_features[:, None]
        if len(esm_features) != len(prot5_features):
            raise ValueError(f"ESM/ProtT5 样本数不一致: {len(esm_features)} vs {len(prot5_features)}")

        self.esm_features = esm_features
        self.prot5_features = prot5_features
        self.labels = np.asarray(labels, dtype=np.int64)

        if len(self.esm_features) != len(self.labels):
            raise ValueError(f"features/labels 数量不一致: {len(self.esm_features)} vs {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        esm_x = torch.from_numpy(self.esm_features[idx]).float()
        prot_x = torch.from_numpy(self.prot5_features[idx]).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return (esm_x, prot_x), y


def dual_array_collate_fn(batch):
    features_tuple, labels = zip(*batch)
    esm_features, prot5_features = zip(*features_tuple)
    esm_features = torch.stack(esm_features)
    prot5_features = torch.stack(prot5_features)
    labels = torch.stack(labels)
    lengths = torch.ones(len(batch), dtype=torch.long)
    return (esm_features, prot5_features), labels, lengths


def load_aligned_dual_features(esm_h5_path: str, prot5_h5_path: str):
    """使用 EmbeddingLoader 读取并对齐 ESM/ProtT5，返回 (esm_features, prot5_features, labels)。"""
    loader = EmbeddingLoader(esm_path=esm_h5_path, prot_path=prot5_h5_path)

    esm_ids, esm_raw_seqs, esm_features, esm_labels = loader._read_records(esm_h5_path)
    prot_ids, prot_raw_seqs, prot_features, prot_labels = loader._read_records(prot5_h5_path)

    prot_id_to_idx = {seq_id: idx for idx, seq_id in enumerate(prot_ids)}
    common_ids = [seq_id for seq_id in esm_ids if seq_id in prot_id_to_idx]

    pairs: list[tuple[int, int]] = []

    if common_ids:
        esm_id_to_idx = {seq_id: idx for idx, seq_id in enumerate(esm_ids)}
        for seq_id in common_ids:
            pairs.append((esm_id_to_idx[seq_id], prot_id_to_idx[seq_id]))
    else:
        # 回退到 raw sequence 对齐（同 embed_loader.py 的确定性策略）
        if all(s is None for s in esm_raw_seqs) or all(s is None for s in prot_raw_seqs):
            raise ValueError("ESM 和 ProtT5 的 seq_id 无交集，且至少一侧缺少 sequences 字段，无法安全对齐")

        from collections import deque

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

    # 标签一致性检查（汇总）：避免逐条打印刷屏。
    esm_pair_labels = [int(esm_labels[e]) for e, _ in pairs]
    prot_pair_labels = [int(prot_labels[p]) for _, p in pairs]
    mismatches = [i for i, (a, b) in enumerate(zip(esm_pair_labels, prot_pair_labels)) if a != b]
    if mismatches:
        uniq = set(esm_pair_labels) | set(prot_pair_labels)
        if uniq.issubset({0, 1}) and len(mismatches) == len(pairs):
            # 常见问题：两边二分类标签完全翻转。翻转 prot 标签用于一致性校验。
            prot_labels = [1 - int(x) for x in prot_labels]
        else:
            show = mismatches[:10]
            examples = [(esm_ids[pairs[i][0]], esm_pair_labels[i], prot_pair_labels[i]) for i in show]
            raise ValueError(
                f"ESM/ProtT5 标签不一致: {len(mismatches)}/{len(pairs)}；例如: {examples}. "
                "请检查生成 h5 的标签映射是否一致（或是否有一侧标签被翻转）。"
            )

    out_esm: list[np.ndarray] = []
    out_prot: list[np.ndarray] = []
    out_labels: list[int] = []
    for esm_idx, prot_idx in pairs:
        out_esm.append(esm_features[esm_idx])
        out_prot.append(prot_features[prot_idx])
        out_labels.append(int(esm_labels[esm_idx]))

    return np.stack(out_esm), np.stack(out_prot), out_labels
# ==== 加载测试集 ====
if is_fusion_model:
    esm_arr, prot_arr, labels = load_aligned_dual_features(test_esm_path, test_prot5_path)
    esm_dim = esm_arr.shape[-1]
    prot5_dim = prot_arr.shape[-1]
    print(f"ESM特征维度: {esm_dim}, ProtT5特征维度: {prot5_dim}")

    test_dataset = DualArrayEmbeddingDataset(esm_arr, prot_arr, labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_config["batch_size"],
        shuffle=False,
        collate_fn=dual_array_collate_fn,
    )
    
    model = DPF(
        esm_dim=esm_dim,
        prot5_dim=prot5_dim,
        hidden_dim=best_config["hidden_dim"],
        num_layers=best_config["num_layers"],
        num_classes=2,
        rank=best_config["rank"],
        steps=best_config["steps"],
        dropout=best_config["dropout"]
    )
else:
    loader = EmbeddingLoader(esm_path=test_esm_path, prot_path=test_prot5_path)
    mode = _feature_mode_from_config(feature_type_config)
    _, features, labels = loader.load_embeddings(mode=mode)

    if features.ndim != 2:
        raise ValueError(f"EmbeddingLoader 输出 features 维度异常: {features.shape}")

    input_dim = features.shape[-1]
    print(f"输入特征维度: {input_dim}")

    test_dataset = ArrayEmbeddingDataset(features, labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_config["batch_size"],
        shuffle=False,
        collate_fn=array_collate_fn,
    )

    model = VFITER(
        input_dim=input_dim,  # 根据加载的特征自动确定维度
        hidden_dim=best_config["hidden_dim"],
        num_layers=best_config["num_layers"],
        dropout=best_config["dropout"],
        rank=best_config["rank"],
        steps=best_config["steps"],
    )


# 加载模型权重
model.to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# ==== 推理 ====
all_labels = []
all_preds = []
all_scores = []

with torch.no_grad():
    for batch in test_loader:
        x, y, lengths = batch
        
        if is_fusion_model:
            # 处理双特征输入
            esm_features, prot5_features = x
            esm_features = esm_features.to(device)
            prot5_features = prot5_features.to(device)
            x = (esm_features, prot5_features)
        else:
            # 处理单特征输入
            x = x.to(device)
        
        y = y.to(device)
        lengths = lengths.to(device)
        
        # 前向传播
        logits = model(x, lengths)
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # 收集结果
        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())

all_scores = np.array(all_scores)

# # 如果是融合模型，保存注意力权重
# if is_fusion_model and hasattr(model, "last_attention_weights"):
#     print(f"平均注意力权重: ESM={model.last_attention_weights[0].item():.4f}, ProtT5={model.last_attention_weights[1].item():.4f}")


import numpy as np
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score)


# ==== 指标计算函数 ====
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
    
    # Calculate AUC and AUPR if probability scores are provided
    auc_score = 0
    aupr_score = 0
    if scores is not None:
        try:
            auc_score = roc_auc_score(labels, scores[:, 1])
            precision, recall, _ = precision_recall_curve(labels, scores[:, 1])
            aupr_score = auc(recall, precision)
        except:
            pass
    
    return sn * 100, sp * 100, acc * 100, f1 * 100, mcc * 100, auc_score * 100, aupr_score * 100

# ==== 输出最终指标 ====
sn, sp, acc, f1, mcc, auc_score, aupr_score = calculate_metrics(all_labels, all_preds, scores=all_scores)
print(f"Sensitivity (SN): {sn:.2f}%")
print(f"Specificity (SP): {sp:.2f}%")
print(f"Accuracy (ACC): {acc:.2f}%")
print(f"F1 Score       : {f1:.2f}%")
print(f"MCC            : {mcc:.2f}%")
print(f"AUC            : {auc_score:.2f}%")
print(f"AUPR           : {aupr_score:.2f}%")

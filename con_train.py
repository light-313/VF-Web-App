import os
import random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

from embed_loader import EmbeddingLoader
from Constant import train_esm_path, train_prot_path, test_esm_path, test_prot_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 配置
# =========================
@dataclass
class Config:
    mode: str = "prot"  # 'esm' / 'prot' / 'both'

    # 数据路径（默认读 Constant.py）
    train_esm: str = train_esm_path
    train_prot: str = train_prot_path
    test_esm: str = test_esm_path
    test_prot: str = test_prot_path

    # 外层交叉验证
    outer_splits: int = 10

    # 对比学习预训练
    supervised: bool = True
    temperature: float = 0.2
    # Hardness-aware SCL: 对“高相似度负样本”加权
    use_hardness_aware_scl: bool = True
    # 负样本权重: w = 1 + hardness_lambda * sigmoid((s - thresh)/scale)
    hardness_lambda: float = 1
    hardness_threshold: float = 0.0
    hardness_scale: float = 0.1
    # 是否用相似度的 detach 来算权重（更稳定）
    hardness_detach_weights: bool = True
    proj_hidden: int = 512
    proj_out: int = 128
    proj_dropout: float = 0.1
    # itelow inside projection head
    itelow_rank: int = 4
    itelow_steps: int = 2
    pretrain_epochs: int = 15
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 0.0
    pretrain_batch_size: int = 256
    pretrain_drop_last: bool = True

    # 预训练早停（基于 train 对比损失；避免使用 outer-val 做选模）
    pretrain_early_stop_patience: int = 10
    pretrain_min_delta: float = 0.0
    pretrain_print_every: int = 1

    # embedding 维度增广
    # embedding 维度增广：用于对比学习时对原始特征做轻微扰动以增强鲁棒性
    drop_rate: float = 0.05        # 随机置零特征的概率（feature dropout）
    gauss_std: float = 0.01       # 加性高斯噪声的标准差（相对于特征标准差的比例）
    mask_span_ratio: float = 0.05 # 连续遮盖的比例（基于特征长度的 span 大小）

    # 线性评估（在 refined features 上）
    linear_hidden: Optional[int] = 256
    linear_dropout: float = 0.1
    linear_epochs: int = 50
    linear_lr: float = 1e-3
    linear_weight_decay: float = 1e-4
    linear_batch_size: int = 512
    linear_use_layernorm: bool = True

    # 线性评估早停（基于 outer-val macro-F1）
    linear_eval_every: int = 1
    linear_print_every: int = 5
    linear_early_stop_patience: int = 10
    linear_min_delta: float = 0.0

    # 性能与可复现
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True

    # projection / inference
    project_batch_size: int = 2048
    project_seed: int = 0

    # 是否尽可能开启确定性（GPU 上仍可能存在少量非确定性算子）
    deterministic: bool = False

    # 是否在“每个外层 fold 完成后”都评估一次测试集
    # 注意：test 会被重复评估 outer_splits 次（用于报告均值±方差），但不应用于选模/调参。
    eval_test_each_fold: bool = True

    # per-fold metrics summary
    save_fold_metrics: bool = True
    fold_metrics_xlsx: str = "fold_metrics_sorted.xlsx"

    out_dir: str = "nested_contrastive_results"

    # =========================
    # 记录 hardness-aware SupCon 的负样本权重 w_ij
    # =========================
    log_pair_weights: bool = True
    # 记录哪些 epoch（1-based）；None 表示记录所有 epoch
    weight_log_epochs: Optional[Tuple[int, ...]] = None
    # 每个 epoch 最多采样多少个 w_ij（负样本对）用于画图/统计（防止内存与磁盘爆炸）
    weight_log_max_samples: int = 50_000
    # 直方图 bins
    weight_hist_bins: int = 80
    # 保存目录（相对 out_dir）
    weight_log_subdir: str = "weight_logs"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """为 DataLoader 的 worker 设定确定性随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def augment_feature(
    x: torch.Tensor,
    drop_rate: float,
    gauss_std: float,
    mask_span_ratio: float,
) -> torch.Tensor:
    x = x.clone()
    if drop_rate > 0:
        mask = torch.rand_like(x) < drop_rate
        x[mask] = 0
    if gauss_std > 0:
        std = x.std().clamp_min(1e-6)
        x = x + torch.randn_like(x) * (gauss_std * std)
    L = x.shape[-1]
    span = int(L * mask_span_ratio)
    if span > 0:
        start = random.randint(0, max(0, L - span))
        x[start : start + span] = 0
    return x


class ContrastiveWrapper(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, cfg: Config):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.cfg = cfg

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        feat = self.x[idx]
        label = self.y[idx]
        v1 = augment_feature(feat, self.cfg.drop_rate, self.cfg.gauss_std, self.cfg.mask_span_ratio)
        v2 = augment_feature(feat, self.cfg.drop_rate, self.cfg.gauss_std, self.cfg.mask_span_ratio)
        return v1, v2, label
from model_type import *


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, rank: int, steps: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Iterblock(hidden, rank=int(rank), steps=int(steps), dropout=float(dropout)),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: Optional[int], dropout: float, use_layernorm: bool):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(in_dim, num_classes)
        else:
            layers: List[nn.Module] = []
            if use_layernorm:
                layers.append(nn.LayerNorm(in_dim))
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, num_classes),
                ]
            )
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t())
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    sim = sim / temperature
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets])
    return F.cross_entropy(sim, targets)


def supervised_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    labels2 = labels.repeat(2)
    sim = torch.matmul(z, z.t()) / temperature
    B2 = z.size(0)
    eye = torch.eye(B2, device=z.device).bool()
    sim.masked_fill_(eye, -9e15)
    pos_mask = (labels2.unsqueeze(0) == labels2.unsqueeze(1)) & (~eye)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    numerator = (pos_mask * log_prob).sum(dim=1)
    denom = pos_mask.sum(dim=1).clamp_min(1)
    return -(numerator / denom).mean()


def hardness_aware_supervised_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    hardness_lambda: float,
    hardness_threshold: float,
    hardness_scale: float,
    detach_weights: bool,
    return_neg_weights: bool = False,
) -> torch.Tensor:
    """Hardness-aware SupCon.

    核心：对负样本 j 的贡献按其相似度 s_ij 加权。
    若负样本相似度高（更“难”），则权重更大，迫使模型去拉开这些易混淆负样本。
    """
    z = torch.cat([z1, z2], dim=0)
    labels2 = labels.repeat(2)
    sim = torch.matmul(z, z.t()) / temperature

    B2 = z.size(0)
    eye = torch.eye(B2, device=z.device).bool()
    sim.masked_fill_(eye, -9e15)

    pos_mask = (labels2.unsqueeze(0) == labels2.unsqueeze(1)) & (~eye)
    neg_mask = (~pos_mask) & (~eye)

    # weights for negatives: w = 1 + lambda * sigmoid((s - thr)/scale)
    s_for_w = sim.detach() if detach_weights else sim
    scale = float(hardness_scale) if float(hardness_scale) > 0 else 1e-6
    w_neg = 1.0 + float(hardness_lambda) * torch.sigmoid((s_for_w - float(hardness_threshold)) / scale)
    w = torch.ones_like(sim)
    w = torch.where(neg_mask, w_neg, w)
    w = torch.where(eye, torch.zeros_like(w), w)

    # weighted denominator: log sum_j w_ij * exp(sim_ij)
    log_w = torch.log(w.clamp_min(1e-12))
    log_denom = torch.logsumexp(sim + log_w, dim=1, keepdim=True)

    log_prob = sim - log_denom
    numerator = (pos_mask * log_prob).sum(dim=1)
    denom = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(numerator / denom).mean()

    if not return_neg_weights:
        return loss

    # 仅返回负样本对的权重向量（不含对角、正样本对）
    neg_w_vec = w[neg_mask]
    return loss, neg_w_vec


def _sample_to_quota(rng: np.random.Generator, values: np.ndarray, quota: int) -> np.ndarray:
    """从 values 中随机采样至 quota（不放回）；若 values 更短则全取。"""
    if quota <= 0:
        return np.empty((0,), dtype=np.float32)
    if values.size <= quota:
        return values.astype(np.float32, copy=False)
    idx = rng.choice(values.size, size=quota, replace=False)
    return values[idx].astype(np.float32, copy=False)


def _plot_weight_hist(epoch_a: int, w_a: np.ndarray, epoch_b: int, w_b: np.ndarray, out_path: str, bins: int) -> None:
    """画 Epoch A vs Epoch B 的 w_ij 分布直方图（叠加）。优先输出 PNG；若缺 matplotlib 则输出 CSV 直方图。"""
    out_path = str(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    # 公共 bins：覆盖两者范围
    all_min = float(min(w_a.min(initial=0.0), w_b.min(initial=0.0)))
    all_max = float(max(w_a.max(initial=1.0), w_b.max(initial=1.0)))
    if all_max <= all_min:
        all_max = all_min + 1e-6

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        plt.hist(w_a, bins=bins, range=(all_min, all_max), alpha=0.55, density=True, label=f"Epoch {epoch_a}")
        plt.hist(w_b, bins=bins, range=(all_min, all_max), alpha=0.55, density=True, label=f"Epoch {epoch_b}")
        plt.xlabel("negative-pair weight w_ij")
        plt.ylabel("density")
        plt.title("Hardness-aware SCL: negative weights distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    except Exception:
        # matplotlib 不可用：输出直方图数据到 CSV
        hist_a, bin_edges = np.histogram(w_a, bins=bins, range=(all_min, all_max), density=True)
        hist_b, _ = np.histogram(w_b, bins=bins, range=(all_min, all_max), density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("bin_center,density_epoch_a,density_epoch_b\n")
            for c, ha, hb in zip(centers, hist_a, hist_b):
                f.write(f"{c},{ha},{hb}\n")
        return


def _save_weight_epoch_stats_csv(rows: List[dict], out_path: str) -> None:
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def _plot_weight_evolution(
    hist_matrix: np.ndarray,
    bin_edges: np.ndarray,
    epochs: List[int],
    stats_rows: List[dict],
    out_png: str,
) -> None:
    """高级一点的可视化：epoch-直方图热力图 + 统计量曲线。

    hist_matrix: [E, B]，每行是该 epoch 的 density 直方图。
    bin_edges: [B+1]
    epochs: len=E
    """
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if hist_matrix.size == 0:
            return

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        # 用 log1p 提升对尾部的可读性
        img = np.log1p(hist_matrix)

        fig = plt.figure(figsize=(10, 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.25)

        ax0 = fig.add_subplot(gs[0, 0])
        im = ax0.imshow(
            img,
            aspect="auto",
            origin="lower",
            extent=[float(bin_centers.min()), float(bin_centers.max()), float(epochs[0]), float(epochs[-1])],
            interpolation="nearest",
        )
        ax0.set_ylabel("epoch")
        ax0.set_title("Hardness-aware SCL: negative weight distribution over epochs (log1p density)")
        cbar = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.02)
        cbar.set_label("log1p(density)")

        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        # stats curves: mean / p50 / p90
        ep = [int(r["epoch"]) for r in stats_rows]
        mean = [float(r["mean"]) for r in stats_rows]
        p50 = [float(r["p50"]) for r in stats_rows]
        p90 = [float(r["p90"]) for r in stats_rows]
        ax1.plot(mean, ep, label="mean", linewidth=1.6)
        ax1.plot(p50, ep, label="p50", linewidth=1.6)
        ax1.plot(p90, ep, label="p90", linewidth=1.6)
        ax1.set_xlabel("negative-pair weight w_ij")
        ax1.set_ylabel("epoch")
        ax1.grid(True, linewidth=0.3, alpha=0.5)
        ax1.legend(loc="lower right", frameon=False)

        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
    except Exception:
        # matplotlib 不可用：跳过 PNG（上层会保存 CSV）
        return


def load_train_embeddings(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    loader = EmbeddingLoader(esm_path=cfg.train_esm, prot_path=cfg.train_prot)
    seq_ids, features, labels = loader.load_embeddings(mode=cfg.mode)
    x = np.asarray(features)
    y = np.asarray(labels)

    return x, y, np.asarray(seq_ids, dtype=str)


def load_test_embeddings(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    loader = EmbeddingLoader(esm_path=cfg.test_esm, prot_path=cfg.test_prot)
    _seq_ids, features, labels = loader.load_embeddings(mode=cfg.mode)
    return np.asarray(features), np.asarray(labels)


def train_projection_head(cfg: Config, x_train: np.ndarray, y_train: np.ndarray, *, fold_tag: Optional[str] = None) -> ProjectionHead:
    set_seed(cfg.seed)

    in_dim = int(x_train.shape[1])
    head = ProjectionHead(
        in_dim=in_dim,
        hidden=cfg.proj_hidden,
        out_dim=cfg.proj_out,
        rank=cfg.itelow_rank,
        steps=cfg.itelow_steps,
        dropout=cfg.proj_dropout,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.pretrain_weight_decay)

    ds = ContrastiveWrapper(x_train, y_train, cfg)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    dl = DataLoader(
        ds,
        batch_size=cfg.pretrain_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=cfg.pretrain_drop_last,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    epoch_iter = tqdm(range(1, cfg.pretrain_epochs + 1), desc="Pretrain", leave=False)

    # weight logging: record all epochs by default (streaming save per epoch)
    record_all_epochs = cfg.weight_log_epochs is None
    weight_epochs = None if record_all_epochs else set(int(e) for e in (cfg.weight_log_epochs or ()))

    # For advanced plots: keep per-epoch histogram rows + summary stats.
    hist_rows: List[np.ndarray] = []
    hist_epochs: List[int] = []
    stats_rows: List[dict] = []

    # Use a stable histogram range based on definition: w in [1, 1+lambda]
    w_min = 1.0
    w_max = 1.0 + max(0.0, float(cfg.hardness_lambda))
    if w_max <= w_min:
        w_max = w_min + 1e-6
    bin_edges = np.linspace(w_min, w_max, int(cfg.weight_hist_bins) + 1, dtype=np.float64)

    for ep in epoch_iter:
        head.train()
        total = 0.0
        n = 0

        # 为每个 epoch 单独设定 rng，保证可复现且 epoch 间不相关
        rng = np.random.default_rng(int(cfg.seed) + int(ep) * 10007)

        # per-epoch weight buffer
        ep_weight_buf: List[np.ndarray] = []
        ep_weight_count = 0

        for v1, v2, lab in dl:
            v1 = v1.to(device)
            v2 = v2.to(device)
            lab = lab.to(device)

            z1 = head(v1)
            z2 = head(v2)
            if cfg.supervised:
                if cfg.use_hardness_aware_scl:
                    want_log = bool(cfg.log_pair_weights) and (record_all_epochs or (weight_epochs is not None and int(ep) in weight_epochs))
                    if want_log:
                        loss, neg_w = hardness_aware_supervised_contrastive_loss(
                            z1,
                            z2,
                            lab,
                            cfg.temperature,
                            cfg.hardness_lambda,
                            cfg.hardness_threshold,
                            cfg.hardness_scale,
                            cfg.hardness_detach_weights,
                            return_neg_weights=True,
                        )

                        # 采样到配额，避免存满所有 pair
                        remaining = int(cfg.weight_log_max_samples) - int(ep_weight_count)
                        if remaining > 0:
                            neg_w_np = neg_w.detach().float().cpu().numpy()
                            samp = _sample_to_quota(rng, neg_w_np, remaining)
                            if samp.size > 0:
                                ep_weight_buf.append(samp)
                                ep_weight_count += int(samp.size)
                    else:
                        loss = hardness_aware_supervised_contrastive_loss(
                            z1,
                            z2,
                            lab,
                            cfg.temperature,
                            cfg.hardness_lambda,
                            cfg.hardness_threshold,
                            cfg.hardness_scale,
                            cfg.hardness_detach_weights,
                        )
                else:
                    loss = supervised_contrastive_loss(z1, z2, lab, cfg.temperature)
            else:
                loss = nt_xent_loss(z1, z2, cfg.temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * v1.size(0)
            n += int(v1.size(0))

        avg = total / max(1, n)

        epoch_iter.set_postfix({"loss": f"{avg:.4f}", "best": f"{best_loss:.4f}", "best_ep": int(best_epoch), "bad": int(bad)})

        if avg < best_loss - float(cfg.pretrain_min_delta):
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= int(cfg.pretrain_early_stop_patience):
                msg = f"    Early stopping pretrain (best_loss={best_loss:.4f} at ep={best_epoch})"
                tqdm.write(msg)
                break

        # end of epoch: save weights + accumulate hist/stats
        if cfg.log_pair_weights and cfg.supervised and cfg.use_hardness_aware_scl and ep_weight_buf:
            log_dir = os.path.join(cfg.out_dir, cfg.weight_log_subdir)
            os.makedirs(log_dir, exist_ok=True)
            tag = cfg.mode if fold_tag is None else f"{cfg.mode}_{fold_tag}"

            arr = np.concatenate(ep_weight_buf, axis=0).astype(np.float32, copy=False)
            out_npy = os.path.join(log_dir, f"neg_w_{tag}_epoch_{int(ep):04d}.npy")
            np.save(out_npy, arr)

            # stats
            stats_rows.append(
                {
                    "epoch": int(ep),
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "p50": float(np.quantile(arr, 0.50)),
                    "p90": float(np.quantile(arr, 0.90)),
                    "p99": float(np.quantile(arr, 0.99)),
                    "max": float(np.max(arr)),
                    "file": os.path.basename(out_npy),
                }
            )

            # histogram row (density)
            hist, _ = np.histogram(arr.astype(np.float64), bins=bin_edges, density=True)
            hist_rows.append(hist.astype(np.float32, copy=False))
            hist_epochs.append(int(ep))

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    # end of training: export stats + advanced plots
    if cfg.log_pair_weights and cfg.supervised and cfg.use_hardness_aware_scl and stats_rows:
        log_dir = os.path.join(cfg.out_dir, cfg.weight_log_subdir)
        os.makedirs(log_dir, exist_ok=True)
        tag = cfg.mode if fold_tag is None else f"{cfg.mode}_{fold_tag}"

        # stats table
        stats_csv = os.path.join(log_dir, f"neg_w_stats_{tag}.csv")
        _save_weight_epoch_stats_csv(stats_rows, stats_csv)

        # heatmap + curves
        if hist_rows and hist_epochs:
            hist_mat = np.stack(hist_rows, axis=0)
            out_png = os.path.join(log_dir, f"neg_w_evolution_{tag}.png")
            _plot_weight_evolution(hist_mat, bin_edges, hist_epochs, stats_rows, out_png)
            # fallback: save raw histogram matrix
            hist_csv = os.path.join(log_dir, f"neg_w_hist_matrix_{tag}.csv")
            try:
                with open(hist_csv, "w", encoding="utf-8") as f:
                    f.write("epoch," + ",".join([f"bin_{i}" for i in range(hist_mat.shape[1])]) + "\n")
                    for e, row in zip(hist_epochs, hist_mat):
                        f.write(str(int(e)) + "," + ",".join(str(float(x)) for x in row.tolist()) + "\n")
            except Exception:
                pass

        # still provide a direct comparison plot: epoch 1 vs last recorded epoch
        try:
            last_ep = int(stats_rows[-1]["epoch"]) if stats_rows else None
            if last_ep is not None and last_ep != 1:
                tag2 = tag
                w1_path = os.path.join(log_dir, f"neg_w_{tag2}_epoch_{1:04d}.npy")
                wL_path = os.path.join(log_dir, f"neg_w_{tag2}_epoch_{last_ep:04d}.npy")
                if os.path.exists(w1_path) and os.path.exists(wL_path):
                    w1 = np.load(w1_path)
                    wL = np.load(wL_path)
                    out_cmp = os.path.join(log_dir, f"neg_w_hist_{tag2}_epoch_0001_vs_{last_ep:04d}.png")
                    _plot_weight_hist(1, w1, last_ep, wL, out_cmp, bins=int(cfg.weight_hist_bins))
        except Exception:
            pass

    return head


@torch.no_grad()
def project_features(cfg: Config, head: ProjectionHead, features: np.ndarray) -> np.ndarray:
    x_t = torch.tensor(features, dtype=torch.float32)
    g = torch.Generator()
    g.manual_seed(int(cfg.project_seed))
    dl = DataLoader(
        TensorDataset(x_t),
        batch_size=int(cfg.project_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory) and torch.cuda.is_available(),
        persistent_workers=(int(cfg.num_workers) > 0),
        worker_init_fn=seed_worker if int(cfg.num_workers) > 0 else None,
        generator=g,
    )

    out: List[torch.Tensor] = []
    for (xb,) in dl:
        z = head(xb.to(device))
        out.append(z.cpu())
    return torch.cat(out, dim=0).numpy()


def train_linear_eval(cfg: Config, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Tuple[nn.Module, float, float, float]:
    set_seed(cfg.seed)
    num_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1
    model = LinearClassifier(
        in_dim=int(x_train.shape[1]),
        num_classes=num_classes,
        hidden=cfg.linear_hidden,
        dropout=cfg.linear_dropout,
        use_layernorm=cfg.linear_use_layernorm,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.linear_lr, weight_decay=cfg.linear_weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.linear_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=g,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.linear_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad = 0

    epoch_iter = tqdm(range(1, cfg.linear_epochs + 1), desc="Linear", leave=False)

    last_batch_loss = float("inf")

    for ep in epoch_iter:
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = criterion(model(xb), yb)
            last_batch_loss = float(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % max(1, cfg.linear_eval_every) == 0 or ep == cfg.linear_epochs:
            f1, acc, mcc = evaluate_classifier(model, val_dl)

            epoch_iter.set_postfix({"loss": f"{last_batch_loss:.4f}", "val_f1": f"{f1:.4f}", "best_f1": f"{best_f1:.4f}", "best_ep": int(best_epoch), "bad": int(bad)})

            if f1 > best_f1 + float(cfg.linear_min_delta):
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = ep
                bad = 0
            else:
                bad += 1
                if bad >= int(cfg.linear_early_stop_patience):
                    msg = f"    Early stopping linear eval (best_val_macro_f1={best_f1:.4f} at ep={best_epoch})"
                    tqdm.write(msg)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    f1, acc, mcc = evaluate_classifier(model, val_dl)
    return model, f1, acc, mcc


@torch.no_grad()
def evaluate_classifier(model: nn.Module, dl: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    y_true = []
    y_pred = []
    for xb, yb in dl:
        logits = model(xb.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(yb.numpy())
    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
    return float(macro_f1), float(acc), float(mcc)


def run_nested_cv(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    if cfg.deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Device: {device}")
    print(f"Config: {cfg}")
    x_all, y_all, _seq_ids = load_train_embeddings(cfg)
    print(f"Train: n={len(y_all)} dim={x_all.shape[1]} classes={len(np.unique(y_all))}")

    skf = StratifiedKFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.seed)

    fold_val_metrics: List[Tuple[float, float, float]] = []
    fold_test_metrics: List[Tuple[float, float, float]] = []
    fold_records: List[dict] = []

    x_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    if cfg.eval_test_each_fold:
        x_test, y_test = load_test_embeddings(cfg)
        print(f"Test: n={len(y_test)} dim={x_test.shape[1]} classes={len(np.unique(y_test))}")

    fold_iter = tqdm(skf.split(x_all, y_all), total=cfg.outer_splits, desc="Outer CV")

    for fold_id, (tr_idx, va_idx) in enumerate(fold_iter, start=1):
        print(f"\n=== Outer Fold {fold_id}/{cfg.outer_splits} ===")
        # 外层：只在 outer-train 上预训练
        x_tr = x_all[tr_idx]
        y_tr = y_all[tr_idx]
        x_va = x_all[va_idx]
        y_va = y_all[va_idx]

        head = train_projection_head(cfg, x_tr, y_tr, fold_tag=f"fold_{fold_id:02d}")

        # 投影得到 refined features（train/val 都用该折训练出来的 head）
        z_tr = project_features(cfg, head, x_tr)
        z_va = project_features(cfg, head, x_va)

        clf, val_f1, val_acc, val_mcc = train_linear_eval(cfg, z_tr, y_tr, z_va, y_va)
        fold_val_metrics.append((val_f1, val_acc, val_mcc))
        tqdm.write(f"[Fold {fold_id}] val_macro_f1={val_f1:.4f} acc={val_acc:.4f} mcc={val_mcc:.4f}")

        test_f1: Optional[float] = None
        test_acc: Optional[float] = None
        test_mcc: Optional[float] = None

        # 每折在固定测试集上评估（用该折训练得到的 head + classifier）
        if cfg.eval_test_each_fold and x_test is not None and y_test is not None:
            z_te = project_features(cfg, head, x_test)
            te_dl = DataLoader(
                TensorDataset(torch.tensor(z_te, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
                batch_size=cfg.linear_batch_size,
                shuffle=False,
            )
            test_f1, test_acc, test_mcc = evaluate_classifier(clf, te_dl)
            fold_test_metrics.append((float(test_f1), float(test_acc), float(test_mcc)))
            tqdm.write(f"[Fold {fold_id}] test_macro_f1={float(test_f1):.4f} acc={float(test_acc):.4f} mcc={float(test_mcc):.4f}")

        # 保存每折模型（可复现）
        fold_dir = os.path.join(cfg.out_dir, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(head.state_dict(), os.path.join(fold_dir, f"proj_head_{cfg.mode}.pth"))
        torch.save(clf.state_dict(), os.path.join(fold_dir, f"linear_{cfg.mode}.pth"))

        record = {
            "fold": int(fold_id),
            "val_macro_f1": float(val_f1),
            "val_acc": float(val_acc),
            "val_mcc": float(val_mcc),
            "test_macro_f1": None if test_f1 is None else float(test_f1),
            "test_acc": None if test_acc is None else float(test_acc),
            "test_mcc": None if test_mcc is None else float(test_mcc),
        }
        fold_records.append(record)
        if cfg.save_fold_metrics:
            with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

    arr_val = np.asarray(fold_val_metrics, dtype=float)
    mean_val = arr_val.mean(axis=0)
    std_val = arr_val.std(axis=0)
    print("\n=== Nested CV Summary (outer val) ===")
    print(f"macro_f1: {mean_val[0]:.4f} ± {std_val[0]:.4f}")
    print(f"acc     : {mean_val[1]:.4f} ± {std_val[1]:.4f}")
    print(f"mcc     : {mean_val[2]:.4f} ± {std_val[2]:.4f}")

    if fold_test_metrics:
        arr_te = np.asarray(fold_test_metrics, dtype=float)
        mean_te = arr_te.mean(axis=0)
        std_te = arr_te.std(axis=0)
        print("\n=== Summary (test evaluated each fold) ===")
        print(f"macro_f1: {mean_te[0]:.4f} ± {std_te[0]:.4f}")
        print(f"acc     : {mean_te[1]:.4f} ± {std_te[1]:.4f}")
        print(f"mcc     : {mean_te[2]:.4f} ± {std_te[2]:.4f}")

    if cfg.save_fold_metrics and fold_records:
        # Sort by highest F1: prefer test F1 if available, otherwise val F1.
        has_test = any(r.get("test_macro_f1") is not None for r in fold_records)
        sort_key = "test_macro_f1" if has_test else "val_macro_f1"
        sorted_records = sorted(
            fold_records,
            key=lambda r: (-1e9 if r.get(sort_key) is None else float(r.get(sort_key))),
            reverse=True,
        )
        out_path = os.path.join(cfg.out_dir, cfg.fold_metrics_xlsx)
        try:
            import pandas as pd

            df = pd.DataFrame(sorted_records)
            df.to_excel(out_path, index=False)
            tqdm.write(f"Saved fold metrics: {out_path} (sorted by {sort_key})")
        except Exception as e:
            # Fallback to CSV if Excel writer engine is missing.
            csv_path = os.path.splitext(out_path)[0] + ".csv"
            try:
                import pandas as pd

                pd.DataFrame(sorted_records).to_csv(csv_path, index=False)
                tqdm.write(f"Excel export failed ({type(e).__name__}); saved CSV instead: {csv_path}")
            except Exception:
                tqdm.write(f"Failed to export fold metrics table: {type(e).__name__}: {e}")

if __name__ == "__main__":
    cfg = Config()

    # 你可以在这里直接改参数
    cfg.mode = "prot"
    cfg.outer_splits = 10
    cfg.supervised = True

    cfg.pretrain_epochs = 200
    cfg.linear_epochs = 500

    # Windows 下建议 0~4
    cfg.num_workers = 8

    run_nested_cv(cfg)

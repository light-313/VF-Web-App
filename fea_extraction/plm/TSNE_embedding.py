import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np

from train import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# 假设已经提取了 H5 数据集的所有特征
# 获取所有特征并转成 NumPy 数组 (假设 feat_list 存储了所有的特征数据)
feat_list = []
labels = []
h5_path="/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"
# 读取所有样本的特征
dataset = H5Dataset(
    # "/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"
    # "/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"
    # "/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"

    h5_path=h5_path,
    feature_type="esm2",
    prot5_path="/root/autodl-tmp/test_prot_features_modified.h5",
    csv_path="/root/VF-pred/raw_data/test_seqsim_features.csv"
)
# from trad_train import *
# dataset = SequenceDataset(
#     # "/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"
#     # "/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"
#     # "/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5"

#     h5_path=h5_path,
#     feature_type="esm2",
#     csv_path="/root/VF-pred/raw_data/test_seqsim_features.csv"
# )
for i in range(len(dataset)):
    feat, label = dataset[i]
    feat_list.append(feat.numpy())  # 将 PyTorch 张量转为 NumPy 数组
    labels.append(label.item())  # 提取标签值并转换为单一的标量

# 将特征数据转换为 NumPy 数组
feat_array = np.array(feat_list)

# 使用 PCA 降维到 50 维
pca = PCA(n_components=50)
feat_pca = pca.fit_transform(feat_array)

# 使用 t-SNE 将 50 维的特征降到 2 维
tsne = TSNE(n_components=2, random_state=42)
feat_tsne = tsne.fit_transform(feat_pca)

# 绘制 t-SNE 可视化图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(feat_tsne[:, 0], feat_tsne[:, 1], c=labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of ESM2 Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig("tsne_visualization.png")

import umap

# 使用 UMAP 降维
umap_model = umap.UMAP(n_components=2, random_state=42)
feat_umap = umap_model.fit_transform(feat_pca)

# 绘制 UMAP 可视化图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(feat_umap[:, 0], feat_umap[:, 1], c=labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title('UMAP Visualization of ESM2 Features')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.savefig("umap_visualization.png")

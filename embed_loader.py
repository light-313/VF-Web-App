import os
import h5py
import torch
import numpy as np
from typing import Optional, List, Tuple
from collections import deque

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

        combined_features: List[np.ndarray] = []
        combined_labels: List[int] = []
        out_ids: List[str] = []

        for esm_idx, prot_idx in pairs:
            out_ids.append(esm_ids[esm_idx])
            combined_features.append(np.concatenate([esm_features[esm_idx], prot_features[prot_idx]], axis=0))
            if esm_labels[esm_idx] != prot_labels[prot_idx]:
                print(f"警告: 序列 {esm_ids[esm_idx]} 在ESM和ProtT5中的标签不一致")
            combined_labels.append(esm_labels[esm_idx])

        return out_ids, np.stack(combined_features), combined_labels

def main():
    """
    示例用法
    """
    # 示例路径（请根据实际情况修改）
    esm_embedding_path = r"C:\Users\light.huang\Desktop\VF-M\VF-pred\data\esm2_test.h5"
    prot_embedding_path = r"C:\Users\light.huang\Desktop\VF-M\VF-pred\data\prot_test.h5"
    
    # 创建加载器
    loader = EmbeddingLoader(esm_path=esm_embedding_path, prot_path=prot_embedding_path)
    
    # 选择加载模式
    mode = "both"  # 可选: "esm", "prot", "both"
    
    try:
        sequence_ids, features, labels = loader.load_embeddings(mode=mode)
        print(f"成功加载 {len(sequence_ids)} 个序列的嵌入")
        print(f"特征维度: {features.shape}")
    except Exception as e:
        print(f"加载嵌入时出错: {e}")

if __name__ == "__main__":
    main()
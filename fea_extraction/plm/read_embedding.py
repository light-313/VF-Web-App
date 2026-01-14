import h5py
import numpy as np

def load_prot5_h5(h5_path):
    """读取ProtT5保存的h5文件，返回embedding、序列和标签字典"""
    embeddings = {}
    sequences = {}
    labels = {}
    with h5py.File(h5_path, 'r') as hf:
        emb_group = hf['embeddings']
        seq_group = hf['sequences']
        label_group = hf['labels']
        for seq_id in emb_group.keys():
            embeddings[seq_id] = np.array(emb_group[seq_id])
            sequences[seq_id] = seq_group[seq_id][()].decode('ascii')
            labels[seq_id] = int(label_group[seq_id][()])
    print(f"共加载{len(embeddings)}条序列")

    return embeddings, sequences, labels

if __name__ == "__main__":
    h5_path = "/root/autodl-tmp/kangjuntai_train_prot5.h5"
    embeddings, sequences, labels = load_prot5_h5(h5_path)
    # 示例：打印前3条
    for i, seq_id in enumerate(list(embeddings.keys())[:3]):
        print(f"ID: {seq_id}")
        print(f"Embedding shape: {embeddings[seq_id].shape}")
        print(f"Sequence: {sequences[seq_id]}")
        print(f"Label: {labels[seq_id]}")
        print("-" * 40)
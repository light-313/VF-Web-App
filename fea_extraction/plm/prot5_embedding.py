import os
import torch
import h5py
import time
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer

# 1. 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_fasta(file_paths, max_length=10000):
    seqs, labels, label_map = [], [], {}
    for path in file_paths:
        try:
            for record in SeqIO.parse(path, "fasta"):
                seq = str(record.seq).upper() # 统一大写
                if len(seq) > max_length: continue
                # 假设 description 格式为: >ID|label
                parts = record.description.split("|")
                label = parts[1].strip() if len(parts) > 1 else "unknown"
                
                if label not in label_map:
                    label_map[label] = len(label_map)
                seqs.append(seq)
                labels.append(label_map[label])
        except Exception as e:
            print(f"Error reading {path}: {e}")
    print(f"Read {len(seqs)} sequences, {len(label_map)} classes.")
    return seqs, labels, label_map

def get_embeddings(model, tokenizer, seq_dict, max_residues=4000, max_batch=100):
    protein_embs = {}
    sorted_keys = sorted(seq_dict.keys(), key=lambda k: len(seq_dict[k]), reverse=True)
    
    pbar = tqdm(total=len(sorted_keys), desc="Encoding")
    batch_ids, batch_seqs = [], []
    current_res = 0

    def process_batch(b_ids, b_seqs):
        # 添加防御性判断，防止空列表报错
        if not b_ids:
            return
            
        processed_seqs = [" ".join(list(s)) for s in b_seqs]
        # 针对 ProstT5 的特殊处理：模型期望输入以 '<AA2cp>' (蛋白质到二结构) 等前缀开始，
        # 但如果是纯嵌入提取，通常直接输入空格分隔的序列即可。
        inputs = tokenizer.batch_encode_plus(
            processed_seqs, 
            add_special_tokens=True, 
            padding=True, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            last_hidden_states = outputs.last_hidden_state
        
        for i, (idx, original_seq) in enumerate(zip(b_ids, b_seqs)):
            seq_len = len(original_seq)
            # 这里的 seq_len 对应空格分隔前的长度，注意 ProstT5 会自动加上 </s> token
            # 我们只需要原始序列长度对应的特征
            emb = last_hidden_states[i, :seq_len].mean(dim=0).cpu().numpy()
            protein_embs[idx] = emb
        pbar.update(len(b_ids))

    for k in sorted_keys:
        seq = seq_dict[k]
        # 修改逻辑：如果加入当前序列会超过阈值，先处理之前的批次
        if (batch_ids and (current_res + len(seq) > max_residues)) or len(batch_ids) >= max_batch:
            process_batch(batch_ids, batch_seqs)
            batch_ids, batch_seqs = [], []
            current_res = 0
        
        batch_ids.append(k)
        batch_seqs.append(seq)
        current_res += len(seq)

    # 处理最后一批
    if batch_ids:
        process_batch(batch_ids, batch_seqs)
    
    pbar.close()
    return protein_embs

def main(fasta_paths, output_path, model_path):
    # 1. 加载模型 - 确保路径正确
    print(f"Loading model from {model_path}...")
    # ProstT5 是基于 T5 架构，通常加载 Encoder 即可
    model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    
    # 2. 数据准备
    seqs_list, labels_list, _ = read_fasta(fasta_paths)
    sequences = {f"seq_{i}": s for i, s in enumerate(seqs_list)}
    labels = {f"seq_{i}": l for i, l in enumerate(labels_list)}
    
    # 3. 提取特征
    embeddings = get_embeddings(model, tokenizer, sequences)
    
# 4. 保存 H5
    print(f"Saving to {output_path}")
    with h5py.File(output_path, 'w') as hf:
        for k in tqdm(embeddings.keys(), desc="Saving"):
            g = hf.create_group(k)
            g.create_dataset('embeddings', data=embeddings[k])
            
            # 修复 NumPy 2.0 的 AttributeError
            # 方法 A: 使用 np.bytes_
            # 方法 B: 直接编码为 utf-8 bytes (推荐)
            curr_seq = sequences[k].encode('utf-8')
            g.create_dataset('sequence', data=curr_seq)
            
            g.create_dataset('label', data=labels[k])

if __name__ == "__main__":
    # 配置
    CONF = {
        "fasta_path": ["/root/VF-pred/raw_data/test.fasta"],
        "output_path": "/root/autodl-fs/2prot5_test.h5",
        "model_path": "/root/autodl-tmp/prot5_model" # 确保此目录下有 config.json 和 pytorch_model.bin
    }
    
    main(CONF["fasta_path"], CONF["output_path"], CONF["model_path"])
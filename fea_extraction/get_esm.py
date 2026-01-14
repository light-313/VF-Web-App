import torch
from tqdm import tqdm
from Bio import SeqIO
import h5py
import os
import warnings
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 与 esm2_embedding.py 保持一致：tokenizer 最大长度与 ESM2 模型维度
max_length = 102400  # tokenizer 最大长度（按需可再调小）
feature_dim = 1280  # ESM2_t33_650M_UR50D hidden size

class ESM2FeatureExtractor:
    def __init__(self, model_path, device=None, layer_num: int = -1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = os.path.basename(model_path).split('.')[0]
        # 与 esm2_embedding.py 的 pooling_method="mean" 逻辑一致：默认用最后一层
        self.layer_num = layer_num
        self._load_model(model_path)

    def _load_model(self, model_path):
        print(f"加载模型: {model_path} ...")
        try:
            if os.path.exists(model_path) and os.path.isdir(model_path):
                model = AutoModel.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # 默认使用 facebook/esm2_t33_650M_UR50D
                hf_model_name = "facebook/esm2_t33_650M_UR50D"
                model = AutoModel.from_pretrained(hf_model_name)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
            print(f"模型加载完成: {self.model_name} (输出层: {self.layer_num}, 设备: {self.device})")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    @staticmethod
    def read_fasta(file_paths, max_sequence_length_filter=500000):
        """
        Reads FASTA files, extracting sequence IDs and sequences.
        Filters out sequences longer than max_sequence_length_filter.
        Does NOT extract or process labels.

        Args:
            file_paths (list): List of paths to FASTA files.
            max_sequence_length_filter (int): Maximum length of sequences to include.
                                                Sequences longer than this will be skipped.
                                                Note: This is different from the tokenizer's max_length.

        Returns:
            tuple: A tuple containing:
                - list: List of sequence IDs (str).
                - list: List of sequences (str).
        """
        sequence_ids: list[str] = []
        sequences: list[str] = []
        labels: list[int] = []
        label_map: dict[str, int] = {}
        read_count = 0
        skipped_count = 0

        for path in file_paths:
            try:
                with open(path, "r") as f:
                    for record in SeqIO.parse(f, "fasta"):
                        read_count += 1
                        seq = str(record.seq)
                        if len(seq) > max_sequence_length_filter:
                            skipped_count += 1
                            continue  # Skip sequences that are too long for the filter

                        # Use record.id as the sequence identifier
                        sequence_ids.append(record.id)
                        sequences.append(seq)
                        # 与 esm2_embedding.py 保持一致：从 description 里取 label，并映射为 int
                        parts = record.description.split("|")
                        if len(parts) < 2:
                            raise ValueError(f"FASTA header 缺少 '|' label 字段: {record.description}")
                        raw_label = parts[1].strip()
                        if raw_label not in label_map:
                            label_map[raw_label] = len(label_map)
                        labels.append(label_map[raw_label])

            except FileNotFoundError:
                print(f"错误: 文件未找到 - {path}")
            except Exception as e:
                print(f"读取文件 {path} 时出错: {e}")

        print(f"从 {len(file_paths)} 个文件读取 {read_count} 条记录.")
        print(f"跳过 {skipped_count} 条长度超过 {max_sequence_length_filter} 的序列.")
        print(f"处理 {len(sequence_ids)} 条序列.")

        return sequence_ids, sequences, labels

    def extract_features(self, sequence_ids, sequences, labels, output_path, batch_size=1, save_format="h5"):
        assert len(sequence_ids) == len(sequences) == len(labels), "长度不一致"
        results = {}

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="提取特征"):
                batch_sequences = sequences[i:i+batch_size]
                batch_ids = sequence_ids[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**inputs, output_hidden_states=True)
                reps = out.hidden_states[self.layer_num]  # [batch, seq_len, hidden_dim]

                for j in range(len(batch_sequences)):
                    current_id = batch_ids[j]
                    # 如果 ID 重复，添加后缀避免覆盖
                    if current_id in results:
                        current_id = f"{current_id}_{i+j}"
                    seq = batch_sequences[j]
                    lbl = batch_labels[j]
                    attn_mask = inputs["attention_mask"][j]
                    seq_len = int(attn_mask.sum().item())

                    # 与 esm2_embedding.py pooling_method="mean" 一致：
                    # 去掉开头 CLS 与末尾 EOS（以及 padding）后做 mean pooling
                    if seq_len > 2:
                        feat = reps[j, 1:seq_len - 1].mean(dim=0).cpu()
                    else:
                        feat = reps[j, 0].cpu()

                    # 保护性处理：确保是固定 1280 维
                    if feat.ndim != 1:
                        feat = feat.view(-1)
                    if feat.shape[0] != feature_dim:
                        if feat.shape[0] > feature_dim:
                            feat = feat[:feature_dim]
                        else:
                            feat = torch.nn.functional.pad(feat, (0, feature_dim - feat.shape[0]))

                    results[current_id] = {"features": feat, "sequence": seq, "label": lbl}

        print(f"提取完成，共 {len(results)} 条序列")
        self._save_features(results, output_path, save_format)

    def _save_features(self, results, output_path, save_format):
        if save_format == "pt":
            torch.save(results, output_path)
        elif save_format == "h5":
            with h5py.File(output_path, 'w') as f:
                emb_grp = f.create_group("embeddings")
                seq_grp = f.create_group("sequences")
                lbl_grp = f.create_group("labels")
                for sid, data in results.items():
                    emb_grp.create_dataset(sid, data=data["features"].numpy())
                    seq_grp.create_dataset(sid, data=data["sequence"].encode('ascii'))
                    lbl_grp.create_dataset(sid, data=int(data["label"]))
                f.attrs["total_sequences"] = len(results)
                f.attrs["feature_dim"] = feature_dim
                f.attrs["pooling_method"] = "mean"
                f.attrs["model_name"] = self.model_name
            print(f"HDF5文件保存完成: {output_path}")
        else:
            raise ValueError("只支持 'pt' 或 'h5' 保存格式")


# ---------------- Example ----------------
if __name__ == "__main__":
    fasta_files = ["/root/VF-pred/raw_data/test.fasta"]
    output_path = "/root/VF-pred/fea_extraction/2esm_test.h5"
    model_path = "/root/autodl-fs/esm"

    extractor = ESM2FeatureExtractor(model_path=model_path, device="cuda")
    seq_ids, seqs, labels = extractor.read_fasta(fasta_files)
    extractor.extract_features(seq_ids, seqs, labels, output_path, batch_size=1, save_format="h5")

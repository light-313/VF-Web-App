import torch
from tqdm import tqdm
from Bio import SeqIO
import h5py
import os
import warnings
from transformers import T5EncoderModel, T5Tokenizer

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

max_length = 1024  # tokenizer 最大长度
model_path='/root/autodl-fs/prot5_model'
class ProtT5FeatureExtractor:
    def __init__(self, model_name="Rostlab/ProstT5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._load_model(model_name)

    def _load_model(self, model_name):
        print(f"加载 ProtT5 模型: {model_name} ...")
        try:
            self.model = T5EncoderModel.from_pretrained(model_path).to(self.device).eval()
            self.tokenizer  = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
            self.model.eval()
            print(f"模型加载完成: {model_name}（设备: {self.device}）")
        except Exception as e:
            print(f"加载模型出错: {e}")
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
    

    def extract_features(self, sequence_ids, sequences, labels, output_path, batch_size=4, save_format="h5"):
        """提取 ProtT5 特征并保存，保存格式与 ESM 完全一致"""
        assert len(sequence_ids) == len(sequences) == len(labels), "ID、序列、标签长度不一致"

        results = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="提取特征"):
                batch_sequences = sequences[i:i+batch_size]
                batch_ids = sequence_ids[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]

                # ProtT5 需要空格分隔氨基酸，并添加 </s> token
                batch_sequences_token = [" ".join(list(seq)) + " </s>" for seq in batch_sequences]
                inputs = self.tokenizer(batch_sequences_token, padding=True, truncation=True,
                                        max_length=max_length, return_tensors="pt").to(self.device)
                out = self.model(**inputs)
                last_hidden = out.last_hidden_state  # [batch, seq_len, hidden_dim]

                for j in range(len(batch_sequences)):
                    current_id = batch_ids[j]
                    # 如果 ID 重复，添加后缀避免覆盖
                    if current_id in results:
                        current_id = f"{current_id}_{i+j}"
                    seq_len = inputs["attention_mask"][j].sum().item()
                    # mean pooling
                    if seq_len > 0:
                        feat = last_hidden[j, :seq_len].mean(dim=0).cpu()
                    else:
                        feat = last_hidden[j, 0].cpu()
                    

                    results[current_id] = {
                        "features": feat,
                        "sequence": batch_sequences[j],  # 保存原始序列
                        "label": batch_labels[j]
                    }

        print(f"提取完成，共 {len(results)} 条序列特征")
        self._save_features(results, output_path, save_format)

    def _save_features(self, results, output_path, save_format):
        """保存 HDF5 或 PT 文件"""
        if save_format == "pt":
            torch.save(results, output_path)
        elif save_format == "h5":
            try:
                with h5py.File(output_path, 'w') as f:
                    embeddings_group = f.create_group("embeddings")
                    sequences_group = f.create_group("sequences")
                    labels_group = f.create_group("labels")

                    for sid, data in results.items():
                        embeddings_group.create_dataset(sid, data=data["features"].numpy())
                        sequences_group.create_dataset(sid, data=data["sequence"].encode('ascii'))
                        labels_group.create_dataset(sid, data=int(data["label"]))
                        

                    f.attrs["total_sequences"] = len(results)
                    f.attrs["feature_dim"] = list(results.values())[0]["features"].shape[0] if results else 0
                    f.attrs["pooling_method"] = "mean"
                    f.attrs["model_name"] = self.model_name

                    print(f"HDF5 文件已保存:")
                    print(f"  - embeddings: {len(results)} 条")
                    print(f"  - sequences: {len(results)} 条")
                    print(f"  - labels: {len(results)} 条")
            except Exception as e:
                print(f"保存 HDF5 文件 {output_path} 时出错: {e}")
                raise
        else:
            raise ValueError(f"不支持的保存格式: {save_format}, 只支持 'pt' 或 'h5'")



# ---------------- Example ----------------
if __name__ == "__main__":
    input_fasta_files = ["/root/VF-pred/raw_data/test.fasta"]
    output_path = "./test.h5"

    extractor = ProtT5FeatureExtractor()
    seq_ids, seqs, labels = extractor.read_fasta(input_fasta_files)
    extractor.extract_features(seq_ids, seqs, labels, output_path, batch_size=1, save_format="h5")
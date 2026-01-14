import torch
from tqdm import tqdm
from Bio import SeqIO
import h5py
import os
import warnings
from torch.nn.utils.rnn import pad_sequence
from tape import ProteinBertModel, TAPETokenizer, UniRepModel

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# You can define max_length here
max_length = 8000

class ProteinFeatureExtractor:
    def __init__(self, model_name, device=None):
        """
        Initializes the feature extractor.
        
        Args:
            model_name (str): The name of the TAPE model to load (e.g., 'babbler-1900').
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._load_model(model_name)

    def _load_model(self, model_name):
        """Loads the TAPE UniRep model and tokenizer."""
        print(f"正在加载模型: {model_name}...")
        
        self.model = ProteinBertModel.from_pretrained('bert-base').to(self.device)
        self.tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
        self.model.eval()
        
        print(f"模型已加载: {self.model_name} (设备: {self.device})")

    @staticmethod
    def read_fasta(file_paths, max_seq_length=10000):
        """Reads FASTA files and truncates sequences exceeding max length."""
        seqs, labels, label_map = [], [], {}
        for path in file_paths:
            try:
                with open(path, "r") as f:
                    for record in SeqIO.parse(f, "fasta"):
                        seq = str(record.seq)
                        if len(seq) > max_seq_length:
                            print(f"Truncating sequence {record.id} from {len(seq)} to {max_seq_length}")
                            seq = seq[:max_seq_length]  # Truncate sequence to max_length
                        
                        label = record.description.split("|")[1].strip()
                        if label not in label_map:
                            label_map[label] = len(label_map)
                        
                        seqs.append(seq)
                        labels.append(label_map[label])
            except Exception as e:
                print(f"读取文件 {path} 时出错: {e}")
        
        print(f"读取 {len(seqs)} 条序列，共 {len(label_map)} 类")
        return seqs, labels

    def extract_features(self, sequences, labels, output_path, batch_size=8, save_format="pt"):
        """Extracts features for sequences and saves them."""
        results = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="提取特征"):
                try:
                    batch_seqs = sequences[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]

                    token_ids = [torch.tensor(self.tokenizer.encode(s)) for s in batch_seqs]
                    
                    # ------------------- FIX IS HERE -------------------
                    # The padding index for TAPETokenizer is 0.
                    # Replaced self.tokenizer.padding_idx with 0.
                    inputs = pad_sequence(token_ids, batch_first=True, padding_value=0).to(self.device)
                    # ---------------------------------------------------

                    _, reps = self.model(inputs)

                    for j, seq in enumerate(batch_seqs):
                        print(len(seq))
                        feature = reps[j].cpu()
                        print(feature.shape)
                        results[f"seq_{i+j}"] = {
                            "features": feature,
                            "label": batch_labels[j],
                            "sequence": seq
                        }

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[显存不足] 当前批中最大序列长度: {max(len(s) for s in batch_seqs)}，请减少 batch_size 或缩短序列。")
                        torch.cuda.empty_cache()
                    else:
                        raise e
        
        feature_dim = list(results.values())[0]["features"].shape if results else "N/A"
        print(f"提取特征完成，共 {len(results)} 条序列")
        print(f"特征维度: {feature_dim}")

        self._save_features(results, output_path, save_format)
        print(f"特征保存至 {output_path} (格式: {save_format})")

    def _save_features(self, results, output_path, save_format):
        """Saves features to a file in the specified format."""
        if save_format == "pt":
            torch.save(results, output_path)
        elif save_format == "h5":
            with h5py.File(output_path, 'w') as f:
                for sid, data in results.items():
                    grp = f.create_group(sid)
                    grp.create_dataset("features", data=data["features"].numpy())
                    grp.attrs["label"] = data["label"]
                    grp.attrs["sequence"] = data["sequence"]
        else:
            raise ValueError(f"不支持的保存格式: {save_format}")

# Example Usage
if __name__ == "__main__":
    import torch


    model_name = 'bert-base'
    input_fasta_files = ["/root/VF-pred/raw_data/train_ba.fasta"]
    output_dir = "/root/autodl-tmp"
    output_path = os.path.join(output_dir, f"train_{model_name}.h5")

    os.makedirs(output_dir, exist_ok=True)
    
    extractor = ProteinFeatureExtractor(model_name=model_name, device="cuda")

    seqs, labels = extractor.read_fasta(input_fasta_files, max_seq_length=max_length)

    extractor.extract_features(
        sequences=seqs,
        labels=labels,
        output_path=output_path,
        batch_size=1,
        save_format="h5"
    )
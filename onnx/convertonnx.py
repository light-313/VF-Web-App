import os
import json
import torch
import numpy as np
import warnings
from model_type import * # 确保 model_type.py 在同一目录
# 如果不需要加载真实数据来获取维度，可以注释掉下面这行并手动指定 dim
from embed_loader import EmbeddingLoader 

# 过滤警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 沿用你的路径配置
BEST_RESULT_PATH = r"C:\Users\WMM2288\Desktop\VF-pred\best_check\2vf\1_fusion_config.json"
BEST_MODEL_PATH = r"C:\Users\WMM2288\Desktop\VF-pred\best_check\2vf\1_fusion_867.pth"
OUTPUT_ONNX_PATH = "best_model.onnx"

# 用于获取特征维度的 H5 路径 (只用于读取维度，不用于推理)
TEST_ESM_PATH = r'C:\Users\WMM2288\Desktop\VF-pred\data\2test_esm.h5'
TEST_PROT5_PATH = r"C:\Users\WMM2288\Desktop\VF-pred\data\2prot.h5"
FEATURE_TYPE_CONFIG = "esm2+prot5"

# 如果不想加载庞大的 H5 文件，可以在这里直接硬编码维度 (设为 None 则自动从文件读取)
# 例如 ESM-2 通常是 1280 (3B) 或 480 (150M), ProtT5 通常是 1024
MANUAL_DIMS = {
    "esm_dim": None,    # 例如 1280
    "prot5_dim": None,  # 例如 1024
    "input_dim": None   # 单流模型时的维度
}
# ===========================================

def get_dims(is_fusion, mode):
    """获取特征维度的辅助函数"""
    dims = {}
    
    if is_fusion:
        if MANUAL_DIMS["esm_dim"] and MANUAL_DIMS["prot5_dim"]:
            print("使用手动配置的维度...")
            return MANUAL_DIMS["esm_dim"], MANUAL_DIMS["prot5_dim"]
        
        print("正在从 H5 文件读取特征维度...")
        # 为了速度，我们只读取第一条记录
        loader = EmbeddingLoader(esm_path=TEST_ESM_PATH, prot_path=TEST_PROT5_PATH)
        # Hack: 直接读取内部方法获取维度，避免加载整个数据集
        import h5py
        with h5py.File(TEST_ESM_PATH, 'r') as f:
            esm_dim = f['features'][0].shape[-1]
        with h5py.File(TEST_PROT5_PATH, 'r') as f:
            prot_dim = f['features'][0].shape[-1]
        return esm_dim, prot_dim
    
    else:
        if MANUAL_DIMS["input_dim"]:
            return MANUAL_DIMS["input_dim"]
            
        print("正在从 H5 文件读取特征维度...")
        target_path = TEST_PROT5_PATH if mode == 'prot' else TEST_ESM_PATH
        import h5py
        with h5py.File(target_path, 'r') as f:
            input_dim = f['features'][0].shape[-1]
        return input_dim

def main():
    device = torch.device("cpu") # 导出通常建议在 CPU 上进行，兼容性最好
    print(f"正在读取配置: {BEST_RESULT_PATH}")
    
    with open(BEST_RESULT_PATH, "r") as f:
        best_config = json.load(f)

    classifier_type = best_config['classifier_type'] 
    is_fusion_model = classifier_type.lower() == "dpf"
    print(f"检测到模型类型: {classifier_type} (融合模型: {is_fusion_model})")

    # 1. 初始化模型
    model = None
    dummy_input = None
    input_names = []
    dynamic_axes = {}

    if is_fusion_model:
        esm_dim, prot5_dim = get_dims(is_fusion=True, mode=None)
        print(f"维度信息: ESM={esm_dim}, ProtT5={prot5_dim}")
        
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
        
        # 准备 DPF 的虚拟输入
        # 注意：你的 DPF forward 接收 (x, lengths)，其中 x 是 (esm, prot) 元组
        batch_size = 1
        dummy_esm = torch.randn(batch_size, esm_dim)
        dummy_prot = torch.randn(batch_size, prot5_dim)
        dummy_lengths = torch.ones(batch_size, dtype=torch.long)
        
        # 构造输入元组：args 必须与 forward 参数签名匹配
        # model.forward((esm, prot), lengths)
        dummy_input = ((dummy_esm, dummy_prot), dummy_lengths)
        
        # ONNX 输入节点名称 (PyTorch 会自动展平元组)
        input_names = ['esm_features', 'prot_features', 'lengths']
        dynamic_axes = {
            'esm_features': {0: 'batch_size'},
            'prot_features': {0: 'batch_size'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    else:
        # 单流模型逻辑
        mode = "prot" if "prot" in FEATURE_TYPE_CONFIG else "esm" # 简化判断
        input_dim = get_dims(is_fusion=False, mode=mode)
        print(f"维度信息: Input={input_dim}")

        model = VFITER(
            input_dim=input_dim,
            hidden_dim=best_config["hidden_dim"],
            num_layers=best_config["num_layers"],
            dropout=best_config["dropout"],
            rank=best_config["rank"],
            steps=best_config["steps"],
        )

        # 准备 VFITER 的虚拟输入
        batch_size = 1
        dummy_feat = torch.randn(batch_size, input_dim)
        dummy_lengths = torch.ones(batch_size, dtype=torch.long)
        
        # model.forward(x, lengths)
        dummy_input = (dummy_feat, dummy_lengths)
        
        input_names = ['features', 'lengths']
        dynamic_axes = {
            'features': {0: 'batch_size'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    # 2. 加载权重
    print(f"正在加载权重: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 3. 导出 ONNX
    print("开始导出 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=13,  # 推荐 13 以获得更好的 Transformer 支持
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"导出成功！文件保存至: {OUTPUT_ONNX_PATH}")

    # 4. 简单验证
    try:
        import onnx
        onnx_model = onnx.load(OUTPUT_ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型结构检查通过。")
    except ImportError:
        print("未安装 onnx 库，跳过验证步骤。")

if __name__ == "__main__":
    main()
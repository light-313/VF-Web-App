import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import datetime
import time
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
# 从trad_train.py导入必要函数
from trad_train import (seed_everything, read_fasta, extract_features_parallel, 
                        SequenceDataset, calculate_metrics, device,
                        create_model, create_dataloader)

def trainable(config):
    """用于Ray Tune的训练函数，支持所有模型类型"""
    # 从共享变量获取数据
    model_type = trainable.model_type
    train_dataset = trainable.train_dataset
    val_dataset = trainable.val_dataset
    input_dim = trainable.input_dim
    
    # 处理XGBoost模型
    if model_type.lower() == "xgboost":
        from xgboost import XGBClassifier
        X_train, y_train = train_dataset
        X_val, y_val = val_dataset
        
        # 创建模型
        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            gamma=config["gamma"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            tree_method='hist',  # 加速训练
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        print(f"XGBoost模型验证集F1: {metrics['f1']:.4f}")
        
    # 处理神经网络模型
    else:
        # 创建数据加载器
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # 创建模型
        hidden_dim = int(config["hidden_dim"])
        dropout = config["dropout"]
        num_layers = 2
        
        model = create_model(model_type, input_dim, hidden_dim, num_layers, dropout, num_classes=2)
        model.to(device)
        
        # 设置优化器
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])
        
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        patience = 5
        best_val_f1 = 0.0
        no_improve = 0
        best_metrics = None
        
        for epoch in range(30):  # 最多训练30轮
            # 训练阶段
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 验证阶段
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 计算指标
            metrics = calculate_metrics(all_labels, all_preds)
            current_f1 = metrics["f1"]
            
            # 早停
            if current_f1 > best_val_f1:
                print(f"第 {epoch + 1} 轮验证集F1提升: {current_f1:.4f} -> {best_val_f1:.4f}")
                best_val_f1 = current_f1
                best_metrics = metrics
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        metrics = best_metrics if best_metrics is not None else metrics
    
    # 报告结果
    session.report({
        "f1": metrics["f1"], 
        "acc": metrics["acc"], 
        "sn": metrics["sn"], 
        "sp": metrics["sp"], 
        "mcc": metrics["mcc"]
    })

def tune_hyperparameters(model_type, feature_combination, train_sequences, train_labels, 
                        test_sequences=None, test_labels=None, num_samples=20):
    """使用Ray Tune进行超参数调优"""
    seed_everything(42)
    start_time = time.time()
    
    # 提取特征
    print(f"使用特征组合 {feature_combination} 提取特征...")
    X_train = extract_features_parallel(train_sequences, feature_combination)
    if test_sequences is not None:
        X_test = extract_features_parallel(test_sequences, feature_combination)
    
    # 确保数据类型正确
    X_train = X_train.astype(np.float32)
    y_train = np.array(train_labels)
    
    # 划分训练集和验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 设置超参数搜索空间
    if model_type.lower() == "xgboost":
        config = {
            "n_estimators": tune.choice([50, 100, 200, 300, 500]),
            "max_depth": tune.randint(3, 10),
            "learning_rate": tune.loguniform(1e-3, 0.3),
            "subsample": tune.uniform(0.6, 1.0),
            "colsample_bytree": tune.uniform(0.6, 1.0),
            "min_child_weight": tune.randint(1, 10),
            "gamma": tune.uniform(0, 0.5),
            "reg_alpha": tune.loguniform(1e-5, 1.0),
            "reg_lambda": tune.loguniform(1e-5, 1.0)
        }
        
        # 设置XGBoost的训练数据
        trainable.train_dataset = (X_train_split, y_train_split)
        trainable.val_dataset = (X_val, y_val)
        
    else:
        config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([64, 128, 256, 512]),
            "hidden_dim": tune.choice([64, 128, 256, 512]),
            "dropout": tune.uniform(0.1, 0.5),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "optimizer": tune.choice(["adam", "adamw", "sgd"])
        }
        
        # 为BiLSTM和MLP添加层数参数

        
        # 创建数据集
        trainable.train_dataset = SequenceDataset(X_train_split, y_train_split)
        trainable.val_dataset = SequenceDataset(X_val, y_val)
    
    # 设置共享变量
    trainable.model_type = model_type
    trainable.input_dim = X_train.shape[1]
    
    # 配置搜索算法和调度器
    search_alg = OptunaSearch(metric="f1", mode="max")
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=30,
        grace_period=10,
        reduction_factor=3
    )
    # 设置输出目录
    experiment_dir = f"/root/autodl-tmp/tune_results/{model_type}_{'_'.join(feature_combination)}"
    os.makedirs(experiment_dir, exist_ok=True)
    # 执行超参数搜索
    print(f"开始 {model_type} 模型的超参数调优，使用特征: {feature_combination}...")
    
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 1, "gpu": 0.2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        verbose=2,
        progress_reporter=tune.CLIReporter(
            parameter_columns=list(config.keys()),
            metric_columns=["f1", "acc", "sn", "sp", "mcc", "training_iteration"]
        ),
        storage_path=experiment_dir,
        name=f"tune_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    )
# 直接从result对象获取DataFrame，不使用Analysis类
    df_results = result.results_df
    
    # 添加模型和特征信息列
    df_results["model_type"] = model_type
    df_results["features"] = str(feature_combination)
    
    # 保存所有试验结果到CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_csv_path = f"{experiment_dir}/all_trials_{timestamp}.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"所有试验结果已保存到: {results_csv_path}")
    
    
    # 获取最佳超参数
    best_trial = result.get_best_trial("f1", "max", "last")
    best_config = best_trial.config
    best_metrics = {
        "f1": best_trial.last_result["f1"],
        "acc": best_trial.last_result["acc"],
        "sn": best_trial.last_result["sn"],
        "sp": best_trial.last_result["sp"],
        "mcc": best_trial.last_result["mcc"]
    }
    
    print(f"\n最佳超参数: {best_config}")
    print(f"验证集F1: {best_metrics['f1']:.4f}, 准确率: {best_metrics['acc']:.4f}")
    
    # 保存调参结果
    output_dir = "/root/autodl-tmp/tune_results"
    os.makedirs(output_dir, exist_ok=True)
    feature_str = "_".join(feature_combination)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"{output_dir}/{model_type}_{feature_str}_params_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump({
            "model_type": model_type,
            "feature_combination": feature_combination,
            "best_config": best_config,
            "validation_metrics": best_metrics,
            "timestamp": timestamp
        }, f, indent=4)
    
    print(f"调参结果已保存到 {output_path}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    return best_config, best_metrics

def train_with_best_params(model_type, feature_combination, best_config, 
                          train_sequences, train_labels, test_sequences=None, test_labels=None):
    """使用最佳超参数训练模型并评估"""
    from trad_train import cross_validate_model
    
    print(f"\n使用最佳超参数训练 {model_type} 模型，特征: {feature_combination}")
    
    # 交叉验证训练
    cv_results = cross_validate_model(
        model_type=model_type,
        feature_combination=feature_combination,
        train_sequences=train_sequences,
        train_labels=train_labels,
        test_sequences=test_sequences,
        test_labels=test_labels,
        num_classes=2,
        k_folds=5,
        batch_size=int(best_config.get("batch_size", 128))
    )
    
    # 保存交叉验证结果
    output_dir = "/root/autodl-tmp/final_models"
    os.makedirs(output_dir, exist_ok=True)
    feature_str = "_".join(feature_combination)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"{output_dir}/{model_type}_{feature_str}_cv_results_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump({
            "model_type": model_type,
            "feature_combination": feature_combination,
            "best_config": best_config,
            "cv_results": {
                "avg_val": {k: float(v) for k, v in cv_results['avg_val'].items()},
                "std_val": {k: float(v) for k, v in cv_results['std_val'].items()},
                "avg_test": {k: float(v) for k, v in cv_results['avg_test'].items()},
                "std_test": {k: float(v) for k, v in cv_results['std_test'].items()},
            },
            "timestamp": timestamp
        }, f, indent=4)
    
    print(f"交叉验证结果已保存到 {output_path}")
    
    # 打印结果摘要
    print("\n交叉验证结果摘要:")
    print(f"验证集F1: {cv_results['avg_val']['f1']:.4f} ± {cv_results['std_val']['f1']:.4f}")
    print(f"测试集F1: {cv_results['avg_test']['f1']:.4f} ± {cv_results['std_test']['f1']:.4f}")
    print(f"验证集准确率: {cv_results['avg_val']['acc']:.4f} ± {cv_results['std_val']['acc']:.4f}")
    print(f"测试集准确率: {cv_results['avg_test']['acc']:.4f} ± {cv_results['std_test']['acc']:.4f}")
    
    return cv_results

if __name__ == "__main__":
    # 设置随机种子
    seed_everything(42)
    
    # 读取数据
    print("读取FASTA文件...")
    train_fasta_file = ("/root/VF-pred/raw_data/train_ba.fasta",)
    test_fasta_file = ("/root/VF-pred/raw_data/test.fasta",)
    
    train_sequences, train_labels, _ = read_fasta(train_fasta_file)
    test_sequences, test_labels, _ = read_fasta(test_fasta_file)
    print(f"训练序列数: {len(train_sequences)}, 测试序列数: {len(test_sequences)}")

    # 要调参的模型和特征组合
    models_to_tune = [
        # {"model": "bilstm", "features": ["kmer", "qso", "seqsim"]},
        # {"model": "cnn", "features": ["seqsim",'dpc']},
        # {"model": "xgboost", "features": ["dde", "dpc", "seqsim"]},
        # {"model": "xgboost", "features": ["aac", "dpc", "seqsim"]},
        # {"model": "cnn", "features": ["seqsim",'paac']},
        # {"model": "bilstm", "features": ["seqsim"]},
        
        # 不包含seqsim的组合
        # {"model": "bilstm", "features": ["kmer", "paac", "qso"]}
        # {"model": "bilstm", "features": ["kmer",  "qso"]},
        # {"model": "cnn", "features": ["kmer",'dde','aac']},
        # {"model": "cnn", "features": ["kmer",'dpc']},
        # {"model": "cnn", "features": ["kmer",'paac','aac']},
        # # {"model": "xgboost", "features": ["kmer", "qso"]},
        # # {"model": "xgboost", "features": ["kmer", "aac"]},
        
        
        # {"model": "bilstm", "features": ["kmer",'qsq']},
        # {"model": "bilstm", "features": ["kmer",'dde']},
        # {"model": "bilstm", "features": ["kmer",'paac','aac']},
        {"model": "cnn", "features": ["kmer",'qso','aac']},
        {"model": "cnn", "features": ["dde",'qso','aac']},
        {"model": "cnn", "features": ["kmer",'qso','dpc']},
        {"model": "xgboost", "features": ["kmer", "paac",'dpc']},
        {"model": "xgboost", "features": ["kmer", "qso"]},
        {"model": "xgboost", "features": ["kmer", "aac"]},
        
        
        
        
        

        
        
        
        
        
        
    ]
    
    # 存储所有结果
    all_results = {}
    
    # 为每个模型和特征组合进行调参
    for config in models_to_tune:
        model_type = config["model"]
        feature_combination = config["features"]
        
        print(f"\n{'='*50}")
        print(f"开始处理 {model_type} 模型，特征: {feature_combination}")
        print(f"{'='*50}")
        
        try:
            # 超参数调优
            best_config, best_metrics = tune_hyperparameters(
                model_type=model_type,
                feature_combination=feature_combination,
                train_sequences=train_sequences,
                train_labels=train_labels,
                test_sequences=test_sequences,
                test_labels=test_labels,
                num_samples=20  # 调整次数
            )
            
            # 使用最佳超参数进行交叉验证训练
            cv_results = train_with_best_params(
                model_type=model_type,
                feature_combination=feature_combination,
                best_config=best_config,
                train_sequences=train_sequences,
                train_labels=train_labels,
                test_sequences=test_sequences,
                test_labels=test_labels
            )
            
            # 记录结果
            key = f"{model_type}_{'_'.join(feature_combination)}"
            all_results[key] = {
                "model_type": model_type,
                "features": feature_combination,
                "best_config": best_config,
                "tune_metrics": best_metrics,
                "cv_metrics": {
                    "val_f1": float(cv_results['avg_val']['f1']),
                    "test_f1": float(cv_results['avg_test']['f1']),
                    "val_acc": float(cv_results['avg_val']['acc']),
                    "test_acc": float(cv_results['avg_test']['acc'])
                }
            }
            
        except Exception as e:
            print(f"处理 {model_type} 模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存总结果
    summary_path = "/root/autodl-tmp/tune_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n所有调参和训练结果已保存到 {summary_path}")
    
    # 输出最佳结果
    print("\n===== 所有模型最佳结果摘要 =====")
    for key, result in all_results.items():
        print(f"\n{result['model_type']} 使用特征 {result['features']}:")
        print(f"  验证集F1: {result['cv_metrics']['val_f1']:.4f}")
        print(f"  测试集F1: {result['cv_metrics']['test_f1']:.4f}")
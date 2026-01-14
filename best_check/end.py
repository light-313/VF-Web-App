import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors

# 1. 准备数据
data = {
    'method': [
        'majority_vote', 'confidence_weighted', 'super_ensemble', 'simple_avg', 
        'max_vote', 'product', 'svm_stacking', 'rank_avg', 'dempster_shafer', 
        'mlp_stacking', 'gradient_boosted', 'xgboost_stacking', 'catboost_stacking', 
        'confidence_intervals', 'bayesian_stacking'
    ],
    'accuracy': [
        0.909774, 0.908365, 0.909305, 0.908835, 0.909305, 0.908835, 0.909305, 
        0.911184, 0.908835, 0.911184, 0.898026, 0.896617, 0.887688, 0.877820, 0.872650
    ],
    'macro_f1': [
        0.876097, 0.875672, 0.875659, 0.875260, 0.873948, 0.873194, 0.872987, 
        0.872968, 0.872471, 0.872413, 0.861466, 0.843560, 0.826997, 0.821470, 0.816760
    ],
    'mcc': [
        0.895958, 0.894304, 0.895421, 0.894856, 0.895378, 0.894905, 0.895334, 
        0.897642, 0.894823, 0.897609, 0.882274, 0.881036, 0.871865, 0.859629, 0.853864
    ]
}

df = pd.DataFrame(data)

# 2. 数据预处理
# 格式化方法名称，去掉下划线，首字母大写
df['method_label'] = df['method'].apply(lambda x: x.replace('_', ' ').title())
# 按 Macro F1 降序排序
df = df.sort_values('macro_f1', ascending=True).reset_index(drop=True)

# 3. 设置绘图风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 优先使用 Arial
sns.set_context("talk") # 调整字体大小适合论文
fig = plt.figure(figsize=(16, 10))

# 创建 GridSpec 布局：左边占 60%，右边占 40%
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.15)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# --- 左图：Lollipop Chart (展示 Macro F1) ---
# 颜色映射：根据 F1 分数生成渐变色
norm = plt.Normalize(df['macro_f1'].min(), df['macro_f1'].max())
colors = cm.viridis(norm(df['macro_f1']))

# 绘制水平线 (Stems)
ax1.hlines(y=df.index, xmin=0.8, xmax=df['macro_f1'], color='gray', alpha=0.5, linewidth=1)
# 绘制点 (Markers)
ax1.scatter(df['macro_f1'], df.index, color=colors, s=150, zorder=3, edgecolors='black', linewidth=0.5)

# 高亮前三名
for i in range(len(df)-3, len(df)):
    ax1.text(df['macro_f1'][i] + 0.002, i, f"{df['macro_f1'][i]:.4f}", 
             va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    # 给前三名加个红圈强调
    ax1.scatter(df['macro_f1'][i], i, s=250, facecolors='none', edgecolors='#d62728', linewidth=2, zorder=4)

# 装饰左图
ax1.set_xlim(0.81, 0.89)  # 设置 X 轴范围聚焦差异
ax1.set_yticks(df.index)
ax1.set_yticklabels(df['method_label'], fontsize=12, fontweight='medium')
ax1.set_xlabel('Macro-F1 Score', fontsize=14, labelpad=10)
ax1.set_title('Evaluation of Ensemble Strategies (Sorted by Macro-F1)', fontsize=16, fontweight='bold', pad=20, loc='left')
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

# 添加箭头指示 Top 1
top_idx = len(df) - 1
ax1.annotate('Best for Imbalanced Data', 
             xy=(df['macro_f1'][top_idx], top_idx), 
             xytext=(df['macro_f1'][top_idx]-0.02, top_idx-1.5),
             arrowprops=dict(facecolor='#333', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, color='#333', style='italic')

# --- 右图：Heatmap (展示 Accuracy, F1, MCC) ---
# 准备热力图数据
heatmap_data = df[['accuracy', 'macro_f1', 'mcc']].copy()
# 为了让热力图颜色对比更明显，我们按列进行 Min-Max 归一化（仅用于颜色显示）
heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

# 绘制热力图
sns.heatmap(heatmap_norm, ax=ax2, cmap="GnBu", annot=heatmap_data, fmt=".4f", 
            cbar=False, linewidths=1, linecolor='white', annot_kws={"size": 10})

# 装饰右图
ax2.set_title('Performance Metrics Heatmap', fontsize=16, fontweight='bold', pad=20)
ax2.set_yticks([]) # 隐藏 Y 轴刻度，因为和左图对齐
ax2.set_xticklabels(['Accuracy', 'Macro-F1', 'MCC'], fontsize=12)
# 调整 X 轴标签位置到顶部
ax2.xaxis.tick_top()

# 添加边框
for _, spine in ax2.spines.items():
    spine.set_visible(True)
    spine.set_color('#ddd')

# 4. 保存与展示
plt.tight_layout()
plt.savefig('ensemble_strategy_evaluation.png', dpi=300)
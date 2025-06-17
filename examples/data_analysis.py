import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 设置支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置中文字体支持")
except:
    print("警告：无法设置中文字体，图表中的中文可能无法正确显示")

# 1. 数据加载与预处理
# 注意：请根据您的实际文件路径修改下面的路径
df = pd.read_csv('/root/autodl-tmp/pykt_self_version/examples/saved_model/assist2009_dkt_qid_saved_model_3407_0_0.5_256_0.001_0_1_0/batch_evaluation_results.csv')  # 请修改为您的实际文件路径

# 检查并处理缺失值
print("缺失值统计:\n", df.isnull().sum())
df = df.dropna()

# 🔍 新增：滤除total_questions<1000的学生
df = df[df['total_questions'] >= 1000]
print(f"滤除后剩余样本数: {len(df)}")

# 2. 相关系数计算
targets = ['windowauclate_mean', 'windowacclate_mean']
features = ['total_questions', 'overall_accuracy', 'accuracy_range', 'accuracy_variance']

# 计算相关系数矩阵
corr_matrix = df[targets + features].corr(method='pearson')

# 3. 统计显著性检验
significant_corrs = {}
for target in targets:
    for feature in features:
        corr, p_value = pearsonr(df[target], df[feature])
        significant_corrs[(target, feature)] = {
            'correlation': corr,
            'p_value': p_value,
            'significance': '显著' if p_value < 0.05 else '不显著'
        }

# 4. 可视化分析 - 第一个图形（3个子图）
plt.figure(figsize=(15, 10))

# 4.1 热力图可视化
plt.subplot(2, 2, 1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('全变量相关系数矩阵热力图', fontsize=14)

# 4.2 结果变量1与特征的相关系数条形图
plt.subplot(2, 2, 2)
sns.barplot(x=corr_matrix.loc[targets[0], features].values, 
            y=features, palette='viridis')
plt.title(f'{targets[0]}与特征相关系数', fontsize=14)
plt.xlim(-1, 1)
plt.axvline(0, color='black', linestyle='--')

# 4.3 结果变量2与特征的相关系数条形图
plt.subplot(2, 2, 3)
sns.barplot(x=corr_matrix.loc[targets[1], features].values, 
            y=features, palette='magma')
plt.title(f'{targets[1]}与特征相关系数', fontsize=14)
plt.xlim(-1, 1)
plt.axvline(0, color='black', linestyle='--')

# 4.4 替代方案：选择一个关键特征的散点图
plt.subplot(2, 2, 4)
# 选择相关性最强的特征进行展示
strongest_feature = 'overall_accuracy'  # 您可以根据实际情况修改
plt.scatter(df[strongest_feature], df[targets[0]], alpha=0.6, label=targets[0])
plt.scatter(df[strongest_feature], df[targets[1]], alpha=0.6, label=targets[1])
plt.xlabel(strongest_feature)
plt.ylabel('目标变量值')
plt.title(f'{strongest_feature}与目标变量关系', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 单独创建散点图矩阵（修正后的pairplot）
plt.figure(figsize=(12, 8))
# 创建一个包含关键变量的数据子集进行pairplot
plot_data = df[targets + features].copy()
g = sns.pairplot(plot_data, 
                x_vars=features, 
                y_vars=targets, 
                kind='reg', 
                plot_kws={'line_kws': {'color': 'red', 'alpha': 0.8}, 'scatter_kws': {'alpha': 0.6}})
g.fig.suptitle('特征与结果变量关系散点图矩阵', y=1.02, fontsize=16)
plt.savefig('pairplot_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 结果输出
print("\n=== 相关系数矩阵 ===")
print(corr_matrix)

print("\n=== 显著性检验结果 ===")
for key, value in significant_corrs.items():
    strength = "极强相关" if abs(value['correlation']) >= 0.8 else \
               "强相关" if abs(value['correlation']) >= 0.6 else \
               "中等相关" if abs(value['correlation']) >= 0.4 else \
               "弱相关" if abs(value['correlation']) >= 0.2 else "极弱/无相关"
               
    print(f"{key[0]} vs {key[1]}: r={value['correlation']:.3f} ({strength}), p={value['p_value']:.4f} ({value['significance']})")

# 7. 可选：创建更详细的相关性分析图
plt.figure(figsize=(10, 6))
# 创建相关系数的条形图比较
corr_data = []
for target in targets:
    for feature in features:
        corr_data.append({
            'Target': target,
            'Feature': feature,
            'Correlation': corr_matrix.loc[target, feature],
            'Significant': significant_corrs[(target, feature)]['significance']
        })

corr_df = pd.DataFrame(corr_data)
pivot_corr = corr_df.pivot(index='Feature', columns='Target', values='Correlation')

sns.heatmap(pivot_corr, annot=True, cmap='RdBu_r', center=0, fmt='.3f', 
            cbar_kws={'label': '相关系数'})
plt.title('目标变量与特征变量相关系数对比', fontsize=14)
plt.ylabel('特征变量')
plt.xlabel('目标变量')
plt.tight_layout()
plt.savefig('correlation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
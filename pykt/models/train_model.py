import os, sys
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
from imblearn.over_sampling import SMOTE
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
from pykt.config import FOCAL_LOSS,freeze_epoch
from pykt.config import MULTI_LEVEL_TRAIN,SMOTE_METHOD
import pandas as pd
import time  # 导入时间模块
import pykt.models.glo
import traceback
from pykt.models.TCN_ABQR import BGRL
from itertools import zip_longest
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ii = 0
TCN_ABQR = 0
def optimized_embedding_smote(minority_embeddings, minority_labels, minority_cshfts,
                               target_count, random_state=42):
    """
    针对高维嵌入优化的SMOTE实现
    """
    np.random.seed(random_state)
    
    # 1. 数据预处理 - 标准化嵌入
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(minority_embeddings)
    
    # 2. 计算数据统计特性
    n_samples, n_features = normalized_embeddings.shape
    feature_std = np.std(normalized_embeddings, axis=0)
    pairwise_distances = pdist(normalized_embeddings)
    median_distance = np.median(pairwise_distances)
    
    print(f"数据维度: {n_features}, 样本数: {n_samples}")
    print(f"中位数距离: {median_distance:.4f}")
    
    # 3. 自适应参数设置
    # 根据数据维度和分布调整参数
    if n_features > 100:  # 高维数据
        k_neighbors = min(15, n_samples // 3)  # 增加邻居数
        alpha_range = (0.1, 0.9)              # 扩大插值范围
        noise_factor = 0.3                     # 增加噪声
        diversity_threshold = 0.7              # 降低多样性阈值
    else:  # 低维数据
        k_neighbors = min(8, n_samples // 4)
        alpha_range = (0.2, 0.8)
        noise_factor = 0.15
        diversity_threshold = 0.5
    
    print(f"自适应参数: k={k_neighbors}, alpha={alpha_range}, noise={noise_factor}")
    
    # 4. 构建邻居模型
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree')
    nbrs.fit(normalized_embeddings)
    
    # 5. 生成合成样本
    n_synthetic = target_count - n_samples
    synthetic_embeddings = []
    synthetic_labels = []
    synthetic_cshfts = []
    
    for i in range(n_synthetic):
        # 随机选择种子样本
        seed_idx = np.random.randint(0, n_samples)
        seed_embedding = normalized_embeddings[seed_idx]
        
        # 找到邻居
        distances, indices = nbrs.kneighbors([seed_embedding])
        neighbor_indices = indices[0][1:]  # 排除自己
        
        # 智能邻居选择：根据距离分布选择
        neighbor_distances = distances[0][1:]
        if len(neighbor_distances) > 3:
            # 选择中等距离的邻居，避免最近和最远
            sorted_indices = np.argsort(neighbor_distances)
            mid_range = sorted_indices[len(sorted_indices)//4:3*len(sorted_indices)//4]
            if len(mid_range) > 0:
                selected_neighbor = np.random.choice(neighbor_indices[mid_range])
            else:
                selected_neighbor = np.random.choice(neighbor_indices)
        else:
            selected_neighbor = np.random.choice(neighbor_indices)
        
        neighbor_embedding = normalized_embeddings[selected_neighbor]
        
        # 扩展插值策略
        alpha = np.random.uniform(*alpha_range)
        
        # 基础插值
        synthetic_embedding = seed_embedding + alpha * (neighbor_embedding - seed_embedding)
        
        # 添加方向性噪声（沿着主成分方向）
        direction = neighbor_embedding - seed_embedding
        direction_noise = np.random.normal(0, noise_factor, size=direction.shape)
        synthetic_embedding += direction_noise * np.linalg.norm(direction)
        
        # 添加随机噪声
        random_noise = np.random.normal(0, noise_factor * feature_std)
        synthetic_embedding += random_noise
        
        # 确保不偏离分布太远（使用较松的约束）
        for dim in range(len(synthetic_embedding)):
            dim_mean = np.mean(normalized_embeddings[:, dim])
            dim_std = feature_std[dim]
            if abs(synthetic_embedding[dim] - dim_mean) > 4 * dim_std:  # 放宽到4个标准差
                synthetic_embedding[dim] = np.random.normal(dim_mean, 2 * dim_std)
        
        synthetic_embeddings.append(synthetic_embedding)
        synthetic_labels.append(minority_labels[seed_idx])
        synthetic_cshfts.append(minority_cshfts[seed_idx])
    
    # 6. 反标准化
    synthetic_embeddings = np.array(synthetic_embeddings)
    synthetic_embeddings = scaler.inverse_transform(synthetic_embeddings)
    
    # 7. 合并数据
    augmented_embeddings = np.vstack([minority_embeddings, synthetic_embeddings])
    augmented_labels = np.hstack([minority_labels, np.array(synthetic_labels)])
    augmented_cshfts = np.hstack([minority_cshfts, np.array(synthetic_cshfts)])
    
    print(f"生成合成样本: {len(synthetic_embeddings)}")
    
    return augmented_embeddings, augmented_labels, augmented_cshfts
def custom_oversampling_improved(minority_embeddings, minority_labels, minority_cshfts, 
                                target_count, k_neighbors=5, random_state=42, 
                                evaluate_quality=True, 
                                # 新增参数解决距离过小问题
                                alpha_range=(0.2, 0.8),  # 扩大插值范围
                                noise_factor=0.1,        # 添加噪声
                                min_distance_ratio=0.3,  # 最小距离比率
                                diversity_boost=True):   # 多样性增强
    """
    改进的自定义过采样函数，解决合成样本过于接近原始样本的问题
    
    参数:
    - alpha_range: 插值系数范围，默认(0.2, 0.8)避免过于接近端点
    - noise_factor: 噪声因子，添加随机噪声增加多样性
    - min_distance_ratio: 最小距离比率阈值
    - diversity_boost: 是否启用多样性增强
    """
    np.random.seed(random_state)
    
    n_samples = len(minority_embeddings)
    n_synthetic = target_count - n_samples
    
    print(f"原始少数类样本数: {n_samples}")
    print(f"需要生成的合成样本数: {n_synthetic}")
    print(f"改进参数: alpha_range={alpha_range}, noise_factor={noise_factor}")
    
    if n_synthetic <= 0:
        print("不需要生成额外样本")
        return minority_embeddings, minority_labels, minority_cshfts
    
    # 构建最近邻模型
    k_neighbors = min(k_neighbors, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree', metric='euclidean')
    nbrs.fit(minority_embeddings)
    
    # 计算原始数据的统计信息
    feature_std = np.std(minority_embeddings, axis=0)
    feature_mean = np.mean(minority_embeddings, axis=0)
    
    # 存储生成的样本
    synthetic_embeddings = []
    synthetic_labels = []
    synthetic_cshfts = []
    
    for i in range(n_synthetic):
        # 随机选择一个种子样本
        seed_idx = np.random.randint(0, n_samples)
        seed_embedding = minority_embeddings[seed_idx]
        seed_label = minority_labels[seed_idx]
        seed_cshft = minority_cshfts[seed_idx]
        
        # 找到种子样本的最近邻（排除自己）
        distances, indices = nbrs.kneighbors([seed_embedding])
        neighbor_indices = indices[0][1:]
        
        # 改进1: 选择较远的邻居增加多样性
        if diversity_boost and len(neighbor_indices) > 2:
            # 选择距离中等的邻居，避免最近的
            mid_idx = len(neighbor_indices) // 2
            neighbor_idx = neighbor_indices[mid_idx:]
            neighbor_idx = np.random.choice(neighbor_idx)
        else:
            neighbor_idx = np.random.choice(neighbor_indices)
        
        neighbor_embedding = minority_embeddings[neighbor_idx]
        neighbor_cshft = minority_cshfts[neighbor_idx]
        
        # 改进2: 使用受限的插值范围
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        synthetic_embedding = seed_embedding + alpha * (neighbor_embedding - seed_embedding)
        
        # 改进3: 添加适量的高斯噪声
        if noise_factor > 0:
            noise = np.random.normal(0, noise_factor * feature_std, size=synthetic_embedding.shape)
            synthetic_embedding = synthetic_embedding + noise
        
        # 改进4: 确保生成的样本不会偏离分布太远
        # 使用软约束将样本拉回合理范围
        for dim in range(len(synthetic_embedding)):
            if abs(synthetic_embedding[dim] - feature_mean[dim]) > 3 * feature_std[dim]:
                synthetic_embedding[dim] = feature_mean[dim] + np.random.normal(0, feature_std[dim])
        
        synthetic_label = seed_label
        synthetic_cshft = np.random.choice([seed_cshft, neighbor_cshft])
        
        synthetic_embeddings.append(synthetic_embedding)
        synthetic_labels.append(synthetic_label)
        synthetic_cshfts.append(synthetic_cshft)
        
        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{n_synthetic} 个合成样本")
    
    # 合并原始样本和合成样本
    synthetic_embeddings = np.array(synthetic_embeddings)
    synthetic_labels = np.array(synthetic_labels)
    synthetic_cshfts = np.array(synthetic_cshfts)
    
    augmented_embeddings = np.vstack([minority_embeddings, synthetic_embeddings])
    augmented_labels = np.hstack([minority_labels, synthetic_labels])
    augmented_cshfts = np.hstack([minority_cshfts, synthetic_cshfts])
    
    print(f"最终增强后样本数: {len(augmented_embeddings)}")
    print(f"标签分布: {Counter(augmented_labels)}")
    
    # 质量评估和自适应调整
    if evaluate_quality and len(synthetic_embeddings) > 0:
        print("\n" + "="*50)
        print("开始评估合成数据质量...")
        print("="*50)
        
        try:
            # 创建评估器
            evaluator = SMOTEEvaluator(
                X_original=minority_embeddings,
                y_original=minority_labels, 
                X_synthetic=synthetic_embeddings,
                y_synthetic=synthetic_labels
            )
            
            # 进行距离分析
            dist_results = evaluator.distance_analysis()
            distance_ratio = dist_results['distance_ratio']
            
            print(f"当前距离比率: {distance_ratio:.4f}")
            
            # 自适应调整：如果距离比率仍然太小，进行二次调整
            if distance_ratio < min_distance_ratio:
                print(f"距离比率 {distance_ratio:.4f} 小于最小阈值 {min_distance_ratio}，进行二次调整...")
                
                # 对合成样本添加更多噪声
                adjustment_factor = min_distance_ratio / distance_ratio
                additional_noise = np.random.normal(0, 
                    noise_factor * adjustment_factor * feature_std, 
                    size=synthetic_embeddings.shape)
                
                synthetic_embeddings_adjusted = synthetic_embeddings + additional_noise
                
                # 重新合并
                augmented_embeddings = np.vstack([minority_embeddings, synthetic_embeddings_adjusted])
                
                print("二次调整完成")
            
            # 进行综合评估
            evaluation_results = evaluator.comprehensive_evaluation()
            
            print("="*50)
            print("合成数据质量评估完成！")
            print("="*50)
            
            return augmented_embeddings, augmented_labels, augmented_cshfts, evaluation_results
            
        except Exception as e:
            print(f"质量评估失败: {e}")
            return augmented_embeddings, augmented_labels, augmented_cshfts, None
    
    return augmented_embeddings, augmented_labels, augmented_cshfts


def adaptive_smote_generation(minority_embeddings, minority_labels, minority_cshfts,
                             target_count, max_iterations=3, random_state=42):
    """
    自适应SMOTE生成，自动调整参数直到达到满意的质量
    """
    # 参数候选列表
    param_configs = [
        # 保守配置
        {'alpha_range': (0.3, 0.7), 'noise_factor': 0.05, 'diversity_boost': False},
        # 中等配置  
        {'alpha_range': (0.2, 0.8), 'noise_factor': 0.1, 'diversity_boost': True},
        # 激进配置
        {'alpha_range': (0.1, 0.9), 'noise_factor': 0.15, 'diversity_boost': True},
    ]
    
    best_result = None
    best_score = 0
    
    for iteration, config in enumerate(param_configs):
        print(f"\n尝试配置 {iteration + 1}: {config}")
        
        result = custom_oversampling_improved(
            minority_embeddings, minority_labels, minority_cshfts,
            target_count, random_state=random_state + iteration,
            **config
        )
        
        if len(result) == 4:  # 包含评估结果
            augmented_embeddings, augmented_labels, augmented_cshfts, evaluation_results = result
            
            if evaluation_results:
                quality_score = evaluation_results['quality_score']
                distance_ratio = evaluation_results['distance_results']['distance_ratio']
                
                print(f"配置 {iteration + 1} 质量评分: {quality_score}/100")
                print(f"配置 {iteration + 1} 距离比率: {distance_ratio:.4f}")
                
                # 评估标准：质量分数高且距离比率合理
                if quality_score > best_score and distance_ratio >= 0.3:
                    best_score = quality_score
                    best_result = result
                    print(f"找到更好的配置！")
                
                # 如果达到满意的结果，提前停止
                if quality_score >= 85 and 0.3 <= distance_ratio <= 1.5:
                    print(f"达到满意的质量标准，停止尝试")
                    return result
    
    if best_result:
        print(f"\n选择最佳配置，质量评分: {best_score}/100")
        return best_result
    else:
        print(f"\n所有配置都不理想，使用最后一个配置的结果")
        return result


# 高级SMOTE变体实现
def borderline_smote_generation(minority_embeddings, minority_labels, minority_cshfts,
                               majority_embeddings, target_count, k_neighbors=5, random_state=42):
    """
    Borderline-SMOTE: 专注于边界样本的过采样
    """
    from sklearn.neighbors import NearestNeighbors
    
    np.random.seed(random_state)
    n_samples = len(minority_embeddings)
    
    # 合并多数类和少数类数据来识别边界样本
    all_data = np.vstack([minority_embeddings, majority_embeddings])
    all_labels = np.hstack([np.ones(len(minority_embeddings)), np.zeros(len(majority_embeddings))])
    
    # 为每个少数类样本找到k个最近邻
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
    nbrs.fit(all_data)
    
    # 识别边界样本（邻居中多数类样本占比在50%-100%之间）
    border_indices = []
    for i in range(n_samples):
        distances, indices = nbrs.kneighbors([minority_embeddings[i]])
        neighbor_labels = all_labels[indices[0]]
        majority_ratio = np.sum(neighbor_labels == 0) / k_neighbors
        
        if 0.5 <= majority_ratio < 1.0:  # 边界样本
            border_indices.append(i)
    
    print(f"识别出 {len(border_indices)} 个边界样本，占比 {len(border_indices)/n_samples:.2%}")
    
    # 如果边界样本太少，放宽条件
    if len(border_indices) < n_samples * 0.1:
        border_indices = list(range(n_samples))
        print("边界样本太少，使用所有样本")
    
    # 从边界样本生成合成样本
    n_synthetic = target_count - n_samples
    synthetic_embeddings = []
    synthetic_labels = []
    synthetic_cshfts = []
    
    minority_nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree')
    minority_nbrs.fit(minority_embeddings)
    
    for i in range(n_synthetic):
        # 随机选择一个边界样本
        border_idx = np.random.choice(border_indices)
        seed_embedding = minority_embeddings[border_idx]
        seed_label = minority_labels[border_idx]
        seed_cshft = minority_cshfts[border_idx]
        
        # 在少数类样本中找邻居
        distances, indices = minority_nbrs.kneighbors([seed_embedding])
        neighbor_indices = indices[0][1:]  # 排除自己
        
        neighbor_idx = np.random.choice(neighbor_indices)
        neighbor_embedding = minority_embeddings[neighbor_idx]
        neighbor_cshft = minority_cshfts[neighbor_idx]
        
        # 生成合成样本
        alpha = np.random.uniform(0.1, 0.9)
        synthetic_embedding = seed_embedding + alpha * (neighbor_embedding - seed_embedding)
        
        synthetic_embeddings.append(synthetic_embedding)
        synthetic_labels.append(seed_label)
        synthetic_cshfts.append(np.random.choice([seed_cshft, neighbor_cshft]))
    
    synthetic_embeddings = np.array(synthetic_embeddings)
    synthetic_labels = np.array(synthetic_labels)
    synthetic_cshfts = np.array(synthetic_cshfts)
    
    augmented_embeddings = np.vstack([minority_embeddings, synthetic_embeddings])
    augmented_labels = np.hstack([minority_labels, synthetic_labels])
    augmented_cshfts = np.hstack([minority_cshfts, synthetic_cshfts])
    
    return augmented_embeddings, augmented_labels, augmented_cshfts

def create_robust_oversampled_dataloader(embeddings1, labels1, cshft1, 
                                       embeddings2, labels2, cshft2,
                                       embeddings3, labels3, cshft3,
                                       batch_size=32, device='cuda', 
                                       method='borderline', # 'adaptive', 'improved', 'borderline'
                                       k_neighbors=5, random_state=42):
    """
    使用鲁棒的过采样方法创建数据加载器
    """
    print(f"使用 {method} 过采样方法")
    
    target_count = len(embeddings2)
    
    if method == 'adaptive':
        # 自适应方法
        print("对Level1进行自适应过采样...")
        result1 = adaptive_smote_generation(
            embeddings1, labels1, cshft1, target_count, random_state=random_state
        )
        augmented_embeddings1 = result1[0]
        augmented_labels1 = result1[1] 
        augmented_cshfts1 = result1[2]
        
        print("对Level3进行自适应过采样...")
        result3 = adaptive_smote_generation(
            embeddings3, labels3, cshft3, target_count, random_state=random_state + 1
        )
        augmented_embeddings3 = result3[0]
        augmented_labels3 = result3[1]
        augmented_cshfts3 = result3[2]
        
    elif method == 'improved':
        # 改进方法
        print("对Level1进行改进过采样...")
        augmented_embeddings1, augmented_labels1, augmented_cshfts1 = custom_oversampling_improved(
            embeddings1, labels1, cshft1, target_count,
            alpha_range=(0.2, 0.8), noise_factor=0.1, diversity_boost=True,
            k_neighbors=k_neighbors, random_state=random_state
        )
        
        print("对Level3进行改进过采样...")
        augmented_embeddings3, augmented_labels3, augmented_cshfts3 = custom_oversampling_improved(
            embeddings3, labels3, cshft3, target_count,
            alpha_range=(0.2, 0.8), noise_factor=0.1, diversity_boost=True,
            k_neighbors=k_neighbors, random_state=random_state + 1
        )
    elif method == 'opt':
        # 改进方法
        print("对Level1进行改进过采样...")
        augmented_embeddings1, augmented_labels1, augmented_cshfts1 = optimized_embedding_smote(
            embeddings1, labels1, cshft1, target_count
        )
        
        print("对Level3进行改进过采样...")
        augmented_embeddings3, augmented_labels3, augmented_cshfts3 = optimized_embedding_smote(
            embeddings3, labels3, cshft3, target_count
        )    
    elif method == 'borderline':
        # 边界SMOTE方法
        print("对Level1进行边界过采样...")
        augmented_embeddings1, augmented_labels1, augmented_cshfts1 = borderline_smote_generation(
            embeddings1, labels1, cshft1, embeddings2,
            target_count, k_neighbors=k_neighbors, random_state=random_state
        )
        
        print("对Level3进行边界过采样...")
        augmented_embeddings3, augmented_labels3, augmented_cshfts3 = borderline_smote_generation(
            embeddings3, labels3, cshft3, embeddings2,
            target_count, k_neighbors=k_neighbors, random_state=random_state + 1
        )
    
    # 合并数据
    final_embeddings = np.vstack([augmented_embeddings1, augmented_embeddings3])
    final_labels = np.hstack([augmented_labels1, augmented_labels3])
    final_cshfts = np.hstack([augmented_cshfts1, augmented_cshfts3])
    
    print(f"最终过采样后数据分布: {Counter(final_labels)}")
    
    # 转换为torch张量
    embeddings_tensor = torch.FloatTensor(final_embeddings).to(device)
    labels_tensor = torch.LongTensor(final_labels).to(device)
    cshfts_tensor = torch.LongTensor(final_cshfts).to(device)
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(embeddings_tensor, labels_tensor, cshfts_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
class SMOTEEvaluator:
    def __init__(self, X_original, y_original, X_synthetic, y_synthetic):
        """
        初始化SMOTE评估器
        
        Parameters:
        X_original: 原始特征数据
        y_original: 原始标签数据  
        X_synthetic: 合成特征数据
        y_synthetic: 合成标签数据
        """
        self.X_original = X_original
        self.y_original = y_original
        self.X_synthetic = X_synthetic
        self.y_synthetic = y_synthetic
        
    def statistical_comparison(self):
        """统计分布比较"""
        print("=== 统计分布比较 ===")
        results = {}
        
        for i in range(self.X_original.shape[1]):
            feature_name = f"Feature_{i+1}"
            
            # 原始数据统计
            orig_mean = np.mean(self.X_original[:, i])
            orig_std = np.std(self.X_original[:, i])
            
            # 合成数据统计
            synth_mean = np.mean(self.X_synthetic[:, i])
            synth_std = np.std(self.X_synthetic[:, i])
            
            # KS检验 - 检验两个分布是否相同
            ks_stat, ks_p = stats.ks_2samp(self.X_original[:, i], self.X_synthetic[:, i])
            
            # Mann-Whitney U检验
            mw_stat, mw_p = stats.mannwhitneyu(self.X_original[:, i], self.X_synthetic[:, i])
            
            results[feature_name] = {
                'original_mean': orig_mean,
                'synthetic_mean': synth_mean,
                'mean_diff': abs(orig_mean - synth_mean),
                'original_std': orig_std,
                'synthetic_std': synth_std,
                'std_diff': abs(orig_std - synth_std),
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'mw_p_value': mw_p
            }
            
        return pd.DataFrame(results).T
    
    def distance_analysis(self):
        """距离分析 - 计算合成样本到最近原始样本的距离"""
        print("=== 距离分析 ===")
        
        # 使用KNN找到每个合成样本最近的原始样本
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(self.X_original)
        
        distances, indices = knn.kneighbors(self.X_synthetic)
        distances = distances.flatten()
        
        # 计算原始数据内部的距离作为参考
        orig_distances = pdist(self.X_original)
        
        results = {
            'synthetic_to_original_distances': distances,
            'mean_distance_to_original': np.mean(distances),
            'std_distance_to_original': np.std(distances),
            'original_internal_mean_distance': np.mean(orig_distances),
            'original_internal_std_distance': np.std(orig_distances),
            'distance_ratio': np.mean(distances) / np.mean(orig_distances)
        }
        
        return results
    
    def diversity_analysis(self):
        """多样性分析"""
        print("=== 多样性分析 ===")
        
        # 计算合成数据的内部距离
        synth_distances = pdist(self.X_synthetic)
        orig_distances = pdist(self.X_original)
        
        # 计算轮廓系数
        combined_X = np.vstack([self.X_original, self.X_synthetic])
        combined_labels = np.hstack([np.zeros(len(self.X_original)), 
                                   np.ones(len(self.X_synthetic))])
        
        silhouette = silhouette_score(combined_X, combined_labels)
        
        results = {
            'synthetic_internal_diversity': np.mean(synth_distances),
            'original_internal_diversity': np.mean(orig_distances),
            'diversity_ratio': np.mean(synth_distances) / np.mean(orig_distances),
            'silhouette_score': silhouette  # 越接近0越好，表示两个数据集越相似
        }
        
        return results
    
    def visualize_comparison(self, method='tsne', figsize=(15, 5)):
        """可视化比较"""
        # 合并数据进行降维
        combined_X = np.vstack([self.X_original, self.X_synthetic])
        combined_labels = ['Original'] * len(self.X_original) + ['Synthetic'] * len(self.X_synthetic)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. 降维可视化
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_X)//4))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            
        reduced_data = reducer.fit_transform(combined_X)
        
        # 绘制降维结果
        for i, label in enumerate(['Original', 'Synthetic']):
            mask = np.array(combined_labels) == label
            axes[0].scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                          label=label, alpha=0.6, s=50)
        axes[0].set_title(f'{method.upper()} Visualization')
        axes[0].legend()
        
        # 2. 特征分布比较（前两个特征）
        if self.X_original.shape[1] >= 2:
            axes[1].scatter(self.X_original[:, 0], self.X_original[:, 1], 
                          label='Original', alpha=0.6, s=50)
            axes[1].scatter(self.X_synthetic[:, 0], self.X_synthetic[:, 1], 
                          label='Synthetic', alpha=0.6, s=50)
            axes[1].set_xlabel('Feature 1')
            axes[1].set_ylabel('Feature 2')
            axes[1].set_title('Feature Space (First 2 Features)')
            axes[1].legend()
        
        # 3. 距离分布比较
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(self.X_original)
        distances, _ = knn.kneighbors(self.X_synthetic)
        
        axes[2].hist(distances.flatten(), bins=30, alpha=0.7, density=True, 
                    label='Distance to Nearest Original')
        axes[2].set_xlabel('Distance')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Distance Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_evaluation(self):
        """综合评估"""
        print("=== SMOTE数据质量综合评估报告 ===\n")
        
        # 1. 统计比较
        stat_results = self.statistical_comparison()
        print("1. 统计分布差异:")
        print(f"   - 平均KS统计量: {stat_results['ks_statistic'].mean():.4f}")
        print(f"   - 平均均值差异: {stat_results['mean_diff'].mean():.4f}")
        print(f"   - 平均标准差差异: {stat_results['std_diff'].mean():.4f}")
        
        # 2. 距离分析
        dist_results = self.distance_analysis()
        print(f"\n2. 距离分析:")
        print(f"   - 合成样本到原始样本平均距离: {dist_results['mean_distance_to_original']:.4f}")
        print(f"   - 距离比率 (合成到原始/原始内部): {dist_results['distance_ratio']:.4f}")
        
        # 3. 多样性分析
        div_results = self.diversity_analysis()
        print(f"\n3. 多样性分析:")
        print(f"   - 多样性比率: {div_results['diversity_ratio']:.4f}")
        print(f"   - 轮廓系数: {div_results['silhouette_score']:.4f}")
        
        # 4. 整体评估
        print(f"\n4. 整体评估:")
        
        # 基于多个指标给出评估
        quality_score = 0
        issues = []
        
        # KS检验评估
        if stat_results['ks_statistic'].mean() < 0.1:
            quality_score += 25
            print("   ✓ 统计分布相似性: 良好")
        elif stat_results['ks_statistic'].mean() < 0.2:
            quality_score += 15
            print("   ⚠ 统计分布相似性: 中等")
        else:
            quality_score += 5
            print("   ✗ 统计分布相似性: 较差")
            issues.append("统计分布差异较大")
        
        # 距离比率评估
        if 0.5 <= dist_results['distance_ratio'] <= 1.5:
            quality_score += 25
            print("   ✓ 距离分布: 合理")
        elif 0.3 <= dist_results['distance_ratio'] <= 2.0:
            quality_score += 15
            print("   ⚠ 距离分布: 可接受")
        else:
            quality_score += 5
            print("   ✗ 距离分布: 异常")
            if dist_results['distance_ratio'] < 0.3:
                issues.append("合成样本过于接近原始样本")
            else:
                issues.append("合成样本距离原始样本过远")
        
        # 多样性评估
        if 0.8 <= div_results['diversity_ratio'] <= 1.2:
            quality_score += 25
            print("   ✓ 数据多样性: 良好")
        elif 0.6 <= div_results['diversity_ratio'] <= 1.4:
            quality_score += 15
            print("   ⚠ 数据多样性: 中等")
        else:
            quality_score += 5
            print("   ✗ 数据多样性: 异常")
            issues.append("数据多样性异常")
        
        # 轮廓系数评估
        if abs(div_results['silhouette_score']) < 0.2:
            quality_score += 25
            print("   ✓ 数据相似性: 高")
        elif abs(div_results['silhouette_score']) < 0.4:
            quality_score += 15
            print("   ⚠ 数据相似性: 中等")
        else:
            quality_score += 5
            print("   ✗ 数据相似性: 低")
            issues.append("合成数据与原始数据相似性较低")
        
        print(f"\n5. 总体质量评分: {quality_score}/100")
        
        if quality_score >= 80:
            print("   评级: 优秀 - SMOTE生成的数据质量很高")
        elif quality_score >= 60:
            print("   评级: 良好 - SMOTE生成的数据质量可接受")
        elif quality_score >= 40:
            print("   评级: 一般 - 建议调整SMOTE参数")
        else:
            print("   评级: 较差 - 强烈建议重新配置SMOTE")
        
        if issues:
            print(f"\n6. 主要问题:")
            for issue in issues:
                print(f"   - {issue}")
        
        return {
            'statistical_results': stat_results,
            'distance_results': dist_results,
            'diversity_results': div_results,
            'quality_score': quality_score,
            'issues': issues
        }



# 在你的代码中添加以下使用示例

# 1. 在 custom_oversampling 函数中添加评估
def custom_oversampling(minority_embeddings, minority_labels, minority_cshfts, 
                                       target_count, k_neighbors=5, random_state=42, 
                                       evaluate_quality=True):
    """
    带质量评估的自定义过采样函数
    """
    np.random.seed(random_state)
    
    n_samples = len(minority_embeddings)
    n_synthetic = target_count - n_samples
    
    print(f"原始少数类样本数: {n_samples}")
    print(f"需要生成的合成样本数: {n_synthetic}")
    
    if n_synthetic <= 0:
        print("不需要生成额外样本")
        return minority_embeddings, minority_labels, minority_cshfts
    
    # 构建最近邻模型
    k_neighbors = min(k_neighbors, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree', metric='euclidean')
    nbrs.fit(minority_embeddings)
    
    # 存储生成的样本
    synthetic_embeddings = []
    synthetic_labels = []
    synthetic_cshfts = []
    
    for i in range(n_synthetic):
        # 随机选择一个种子样本
        seed_idx = np.random.randint(0, n_samples)
        seed_embedding = minority_embeddings[seed_idx]
        seed_label = minority_labels[seed_idx]
        seed_cshft = minority_cshfts[seed_idx]
        
        # 找到种子样本的最近邻（排除自己）
        distances, indices = nbrs.kneighbors([seed_embedding])
        neighbor_indices = indices[0][1:]
        
        # 随机选择一个邻居
        neighbor_idx = np.random.choice(neighbor_indices)
        neighbor_embedding = minority_embeddings[neighbor_idx]
        neighbor_cshft = minority_cshfts[neighbor_idx]
        
        # 在种子样本和邻居之间生成新样本
        alpha = np.random.uniform(0.3, 0.7)
        synthetic_embedding = seed_embedding + alpha * (neighbor_embedding - seed_embedding)
        
        synthetic_label = seed_label
        synthetic_cshft = np.random.choice([seed_cshft, neighbor_cshft])
        
        synthetic_embeddings.append(synthetic_embedding)
        synthetic_labels.append(synthetic_label)
        synthetic_cshfts.append(synthetic_cshft)
        
        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{n_synthetic} 个合成样本")
    
    # 合并原始样本和合成样本
    synthetic_embeddings = np.array(synthetic_embeddings)
    synthetic_labels = np.array(synthetic_labels)
    synthetic_cshfts = np.array(synthetic_cshfts)
    
    augmented_embeddings = np.vstack([minority_embeddings, synthetic_embeddings])
    augmented_labels = np.hstack([minority_labels, synthetic_labels])
    augmented_cshfts = np.hstack([minority_cshfts, synthetic_cshfts])
    
    print(f"最终增强后样本数: {len(augmented_embeddings)}")
    print(f"标签分布: {Counter(augmented_labels)}")
    
    # ============ 添加质量评估 ============
    if evaluate_quality and len(synthetic_embeddings) > 0:
        print("\n" + "="*50)
        print("开始评估合成数据质量...")
        print("="*50)
        
        try:
            # 创建评估器
            evaluator = SMOTEEvaluator(
                X_original=minority_embeddings,
                y_original=minority_labels, 
                X_synthetic=synthetic_embeddings,
                y_synthetic=synthetic_labels
            )
            
            # 进行综合评估
            evaluation_results = evaluator.comprehensive_evaluation()
            
            # 可视化比较（可选）
            # evaluator.visualize_comparison(method='tsne')
            
            print("="*50)
            print("合成数据质量评估完成！")
            print("="*50)
            
            return augmented_embeddings, augmented_labels, augmented_cshfts
            
        except Exception as e:
            print(f"质量评估失败: {e}")
            return augmented_embeddings, augmented_labels, augmented_cshfts, None
    
    return augmented_embeddings, augmented_labels, augmented_cshfts

def create_custom_oversampled_dataloader(embeddings1, labels1, cshft1, 
                                       embeddings2, labels2, cshft2,
                                       embeddings3, labels3, cshft3,
                                       batch_size=32, device='cuda', 
                                       k_neighbors=5, random_state=42):
    """
    使用自定义过采样创建数据加载器
    embeddings2, labels2, cshft2 是多数类（作为目标大小）
    embeddings1, labels1, cshft1 和 embeddings3, labels3, cshft3 是少数类（需要过采样）
    只将过采样后的少数类数据放入dataloader
    """
    
    print(f"Level1数据形状: embeddings={embeddings1.shape}, labels={labels1.shape}, cshfts={cshft1.shape}")
    print(f"Level2数据形状: embeddings={embeddings2.shape}, labels={labels2.shape}, cshfts={cshft2.shape}")  
    print(f"Level3数据形状: embeddings={embeddings3.shape}, labels={labels3.shape}, cshfts={cshft3.shape}")
    
    print(f"Level1标签分布: {Counter(labels1)}")
    print(f"Level2标签分布: {Counter(labels2)}")
    print(f"Level3标签分布: {Counter(labels3)}")
    
    # 确定目标大小（多数类level2的大小）
    target_count = len(embeddings2)
    print(f"目标过采样大小: {target_count}")
    
    try:
        # 分别对level1和level3进行过采样，使其达到level2的大小
        print("对Level1进行过采样...")
        augmented_embeddings1, augmented_labels1, augmented_cshfts1 = custom_oversampling(
            embeddings1, labels1, cshft1,
            target_count=target_count,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        
        print("对Level3进行过采样...")
        augmented_embeddings3, augmented_labels3, augmented_cshfts3 = custom_oversampling(
            embeddings3, labels3, cshft3,
            target_count=target_count,
            k_neighbors=k_neighbors,
            random_state=random_state + 1  # 使用不同的随机种子
        )
        
        # 只合并过采样后的少数类数据（level1和level3），不包括多数类level2
        final_embeddings = np.vstack([augmented_embeddings1, augmented_embeddings3])
        final_labels = np.hstack([augmented_labels1, augmented_labels3])
        final_cshfts = np.hstack([augmented_cshfts1, augmented_cshfts3])
        
        print(f"最终过采样后数据分布: {Counter(final_labels)}")
        print(f"最终数据形状: embeddings={final_embeddings.shape}, labels={final_labels.shape}, cshfts={final_cshfts.shape}")
        print(f"Level1过采样后大小: {len(augmented_embeddings1)}")
        print(f"Level3过采样后大小: {len(augmented_embeddings3)}")
        print(f"总计数据大小: {len(final_embeddings)}")
        
        # 转换为torch张量
        embeddings_tensor = torch.FloatTensor(final_embeddings).to(device)
        labels_tensor = torch.LongTensor(final_labels).to(device)
        cshfts_tensor = torch.LongTensor(final_cshfts).to(device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(embeddings_tensor, labels_tensor, cshfts_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print("自定义过采样数据加载器创建成功！")
        return dataloader
        
    except Exception as e:
        print(f"自定义过采样失败: {e}")
        import traceback
        print(f"完整错误信息: {traceback.format_exc()}")
        
        # 如果失败，返回原始的少数类数据（level1和level3）
        print("使用原始少数类数据创建数据加载器...")
        combined_embeddings = np.vstack([embeddings1, embeddings3])
        combined_labels = np.hstack([labels1, labels3])
        combined_cshfts = np.hstack([cshft1, cshft3])
        
        embeddings_tensor = torch.FloatTensor(combined_embeddings).to(device)
        labels_tensor = torch.LongTensor(combined_labels).to(device)
        cshfts_tensor = torch.LongTensor(combined_cshfts).to(device)
        
        dataset = TensorDataset(embeddings_tensor, labels_tensor, cshfts_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print("原始少数类数据加载器创建成功！")
        return dataloader
    
def train_model_multilevel_custom(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, 
                                 level1_loader=None,level2_loader=None, level3_loader=None, freeze_epoch=10,
                                 test_loader=None, test_window_loader=None, save_model=False, 
                                 data_config=None, fold=None, emb_sizess=128, model_config={}):
    """使用自定义过采样的多级训练函数"""
    max_auc, best_epoch = 0, -1
    train_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查是否启用多级训练
    if MULTI_LEVEL_TRAIN != 1:
        print("使用原始训练方式")
        return train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                          test_loader, test_window_loader, save_model, data_config, fold,
                          emb_sizess, model_config, level1_loader,level2_loader, level3_loader)
    
    print("使用自定义过采样的多级训练方式")
    
    # 检查必要的数据加载器
    if level1_loader is None or level3_loader is None or level2_loader is None:
        print("警告: level1_loader 或 level3_loader 为空，使用原始训练方式")
        return train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                          test_loader, test_window_loader, save_model, data_config, fold,
                          emb_sizess, model_config, level1_loader, level3_loader)
    
    custom_dataloader = None
    model_frozen = False
    
    # 初始化最佳模型指标
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = 0, 0
    
    # 计时相关变量
    import time
    total_forward_time = 0.0
    total_epoch_time = 0.0
    total_forward_count = 0
    
    for i in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        loss_mean = []
        epoch_forward_time = 0.0
        epoch_forward_count = 0
        
        # 第一阶段：正常训练到指定epoch
        if i <= freeze_epoch:
            model.train()
            for data in train_loader:
                train_step += 1
                
                # 计时
                forward_start = time.time()
                
                # DKT前向传播
                loss = model_forward(model, data)
                
                forward_end = time.time()
                forward_duration = forward_end - forward_start
                epoch_forward_time += forward_duration
                epoch_forward_count += 1
                total_forward_time += forward_duration
                total_forward_count += 1
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                loss_mean.append(loss.detach().cpu().numpy())
        
        # 在指定epoch后进行自定义过采样
        elif i == freeze_epoch + 1:
            print(f"在第{i}个epoch进行自定义过采样...")
            
            # 冻结模型权重，收集嵌入
            print("冻结模型权重，收集嵌入向量...")
            for param in model.parameters():
                param.requires_grad = False
            model_frozen = True
            
            # 从level1和level3数据中收集嵌入
            print("从level1数据收集嵌入...")
            embeddings1, labels1, cshft1 = collect_embeddings_from_loader(model, level1_loader, device)
            print("从level2数据收集嵌入...")
            embeddings2, labels2, cshft2 = collect_embeddings_from_loader(model, level2_loader, device)
            print("从level3数据收集嵌入...")
            embeddings3, labels3, cshft3 = collect_embeddings_from_loader(model, level3_loader, device)
            
            # 使用自定义过采样创建新的数据加载器
            print("使用自定义过采样创建增强数据...")
            # custom_dataloader = create_custom_oversampled_dataloader(
            #     embeddings1, labels1, cshft1,embeddings2,labels2, cshft2, embeddings3, labels3, cshft3,
            #     batch_size=32, device=device, k_neighbors=10, random_state=42
            # )
            custom_dataloader = create_robust_oversampled_dataloader(
                embeddings1, labels1, cshft1,embeddings2,labels2, cshft2, embeddings3, labels3, cshft3,
                batch_size=32, device=device, k_neighbors=10, random_state=42,method=SMOTE_METHOD
            )
            # 解冻模型权重
            print("解冻模型权重...")
            for param in model.parameters():
                param.requires_grad = True
            model_frozen = False
            
            # 继续训练
            model.train()
            
            # 在原始数据上训练
            for data in train_loader:
                train_step += 1
                
                forward_start = time.time()
                loss = model_forward(model, data)
                forward_end = time.time()
                forward_duration = forward_end - forward_start
                epoch_forward_time += forward_duration
                epoch_forward_count += 1
                total_forward_time += forward_duration
                total_forward_count += 1
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                loss_mean.append(loss.detach().cpu().numpy())
            
            # 在自定义过采样数据上训练
            if custom_dataloader is not None:
                print("开始训练smote")
                for custom_data in custom_dataloader:
                    train_step += 1
                    
                    forward_start = time.time()
                    loss = model_forward_with_smote_data(model, custom_data, device)
                    forward_end = time.time()
                    forward_duration = forward_end - forward_start
                    epoch_forward_time += forward_duration
                    epoch_forward_count += 1
                    total_forward_time += forward_duration
                    total_forward_count += 1
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    loss_mean.append(loss.detach().cpu().numpy())
        
        # 第三阶段：继续在原始数据和自定义过采样数据上训练
        else:
            model.train()
            
            # 在原始数据上训练
            for data in train_loader:
                train_step += 1
                
                forward_start = time.time()
                loss = model_forward(model, data)
                forward_end = time.time()
                forward_duration = forward_end - forward_start
                epoch_forward_time += forward_duration
                epoch_forward_count += 1
                total_forward_time += forward_duration
                total_forward_count += 1
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                loss_mean.append(loss.detach().cpu().numpy())
            
            # 在自定义过采样数据上训练（如果存在）
            if custom_dataloader is not None:
                for custom_data in custom_dataloader:
                    train_step += 1
                    
                    forward_start = time.time()
                    loss = model_forward_with_smote_data(model, custom_data, device)
                    forward_end = time.time()
                    forward_duration = forward_end - forward_start
                    epoch_forward_time += forward_duration
                    epoch_forward_count += 1
                    total_forward_time += forward_duration
                    total_forward_count += 1
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    loss_mean.append(loss.detach().cpu().numpy())
        
        # 计算平均损失
        loss_mean = np.mean(loss_mean)
        
        # 验证模型
        auc, acc = evaluate(model, valid_loader, model.model_name)
        
        # 保存最佳模型 (完整的保存逻辑)
        if auc > max_auc + 1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + "_model.ckpt"))
                print(f"保存最佳模型到: {os.path.join(ckpt_path, model.emb_type + '_model.ckpt')}")
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader is not None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader is not None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        
        # 计算时间统计
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_epoch_time += epoch_duration
        avg_forward_time_epoch = epoch_forward_time / epoch_forward_count if epoch_forward_count > 0 else 0.0
        
        # 打印训练信息 (与原版相同的格式)
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")
        print(f"            Avg Forward Time this Epoch: {avg_forward_time_epoch:.6f} seconds")
        
        if i == freeze_epoch + 1:
            print(f"            已完成自定义过采样，继续训练...")
        
        # 早停机制
        if i - best_epoch >= 10:
            break
    
    # 确保模型权重被解冻
    if model_frozen:
        for param in model.parameters():
            param.requires_grad = True
    
    # 计算总体时间统计
    overall_avg_forward_time = total_forward_time / total_forward_count if total_forward_count > 0 else 0.0
    avg_epoch_time = total_epoch_time / i if i > 0 else 0.0
    
    print("\n自定义过采样多级训练完成！")
    print(f"每个 forward 的总体平均时间: {overall_avg_forward_time:.6f} 秒")
    print(f"每个 epoch 的平均时间: {avg_epoch_time:.2f} 秒")
    
    # 保存最终结果 (与原版相同)
    results = {
        "Training Method": "Multi-Level with Custom Oversampling",
        "Freeze Epoch": freeze_epoch,
        "Best Epoch": best_epoch,
        "Best AUC": max_auc,
        "Valid AUC": validauc,
        "Valid Accuracy": validacc,
        "Test AUC": testauc,
        "Test Accuracy": testacc,
        "Window Test AUC": window_testauc,
        "Window Test Accuracy": window_testacc,
        "Overall Avg Forward Time": overall_avg_forward_time,
        "Avg Epoch Time": avg_epoch_time
    }
    save_results_to_file(results, ckpt_path)
    
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch


if TCN_ABQR:
    pre_load_gcn = "/share/disk/hzb/dataset/assistment2009/ques_skill_gcn_adj.pt"
    matrix = torch.load(pre_load_gcn).to(device)
    if not matrix.is_sparse:
        matrix = matrix.to_sparse()
def save_results_to_file(results, folder_path, file_name="results.txt"):
    """保存结果到文件中"""
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"结果已保存到文件: {file_path}")


import torch
import torch.nn as nn
import torch.nn.functional as F
def collect_embeddings_from_loader(model, data_loader, device):
    """从数据加载器中收集嵌入向量"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_cshft = []
    with torch.no_grad():
        for data in data_loader:
            # 解析数据
            dcur = data
            q, c, r = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device)
            qshft, cshft, rshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device)
            m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)
            
            # DKT前向传播获取嵌入
            if model.model_name == "dkt":
                y, xemb = model(c.long(), r.long())
                
                # 收集有效的嵌入向量和对应的标签
                valid_mask = sm.bool()  # 使用shifted mask来确定有效位置
                
                # 展平处理
                xemb_flat = xemb.view(-1, xemb.size(-1))  # [batch_size * seq_len, emb_size]
                rshft_flat = rshft.view(-1)  # [batch_size * seq_len]
                
                cshft_flat = cshft.view(-1)
                valid_mask_flat = valid_mask.view(-1)  # [batch_size * seq_len]
                
                # 只取有效位置的数据
                valid_embeddings = xemb_flat[valid_mask_flat]
                valid_labels = rshft_flat[valid_mask_flat]
                valid_cshft = cshft_flat[valid_mask_flat]
                
                all_embeddings.append(valid_embeddings.cpu().numpy())
                all_labels.append(valid_labels.cpu().numpy())
                all_cshft.append(valid_cshft.cpu().numpy())
    
    # 合并所有批次的数据
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)
    all_cshft = np.hstack(all_cshft)
    
    return all_embeddings, all_labels , all_cshft

def save_evaluation_results(evaluation_results, filename):
    """保存SMOTE评估结果到文件"""
    if evaluation_results is None:
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SMOTE数据质量评估报告\n")
            f.write("="*50 + "\n\n")
            
            # 写入质量评分
            quality_score = evaluation_results.get('quality_score', 0)
            f.write(f"总体质量评分: {quality_score}/100\n\n")
            
            # 写入统计结果
            stat_results = evaluation_results.get('statistical_results')
            if stat_results is not None:
                f.write("统计分布比较:\n")
                f.write(f"平均KS统计量: {stat_results['ks_statistic'].mean():.4f}\n")
                f.write(f"平均均值差异: {stat_results['mean_diff'].mean():.4f}\n")
                f.write(f"平均标准差差异: {stat_results['std_diff'].mean():.4f}\n\n")
            
            # 写入距离分析
            dist_results = evaluation_results.get('distance_results')
            if dist_results is not None:
                f.write("距离分析:\n")
                f.write(f"合成样本到原始样本平均距离: {dist_results['mean_distance_to_original']:.4f}\n")
                f.write(f"距离比率: {dist_results['distance_ratio']:.4f}\n\n")
            
            # 写入多样性分析
            div_results = evaluation_results.get('diversity_results')
            if div_results is not None:
                f.write("多样性分析:\n")
                f.write(f"多样性比率: {div_results['diversity_ratio']:.4f}\n")
                f.write(f"轮廓系数: {div_results['silhouette_score']:.4f}\n\n")
            
            # 写入问题列表
            issues = evaluation_results.get('issues', [])
            if issues:
                f.write("主要问题:\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
        
        print(f"评估结果已保存到: {filename}")
        
    except Exception as e:
        print(f"保存评估结果失败: {e}")
        


def model_forward_with_smote_data(model, smote_data, device):
    """处理SMOTE生成的数据进行训练"""
    embeddings, labels ,cshfts = smote_data
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # 从嵌入直接到LSTM层（跳过embedding层）
    # h, _ = model.lstm_layer(embeddings.unsqueeze(1))  # 添加序列维度
    # h = model.dropout_layer(h)
    # y = model.out_layer(h)
    y,_ = model(if_strait_input = 1,strait_input = embeddings)
    
    print(f"y{y.shape}")
    print(f"labels{labels.shape}")
    y = (y * one_hot(cshfts.long(), model.num_c)).sum(-1)
    # y = torch.sigmoid(y).squeeze(1)  # 移除序列维度
    print(f"y_sig{y.shape}")
    # 计算损失
    loss = torch.nn.functional.binary_cross_entropy(y, labels.float())
    
    return loss


def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name
    
    # 定义 Focal Loss 函数
    def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        y_pred: 模型输出的概率 [N]
        y_true: 真实标签 [N]
        alpha: 类别平衡权重
        gamma: 聚焦参数
        reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        
        # 计算 p_t = p if y=1 else 1-p
        p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
        
        # 计算调制因子 (1 - p_t)^gamma
        focal_term = (1 - p_t) ** gamma
        
        # 应用类别平衡权重 alpha_t
        alpha_t = torch.where(y_true == 1, alpha, 1 - alpha)
        loss = alpha_t * focal_term * bce_loss
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss
    
    # 根据全局变量选择损失函数
    if FOCAL_LOSS:
        loss_fn = focal_loss
        print("Using Focal Loss!")
    else:
        loss_fn = F.binary_cross_entropy
        print("Using Binary Cross-Entropy Loss!")

    if model_name in ["atdkt", "simplekt", "stablekt", "bakt_time", "sparsekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss1 = loss_fn(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
            
    elif model_name in ["rekt"]:
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double())

    elif model_name in ["rkt","dimkt","LSTM_Template","CTNKT","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double())
        
    elif model_name in ["TCN_ABQR"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double())
        
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        
        # 对两个损失都使用 Focal Loss
        loss = loss_fn(y_next.double(), r_next.double())
        loss_r = loss_fn(y_curr.double(), r_curr.double())
        
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
        
    elif model_name in ["Transformer_Template","akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","dtransformer"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double()) + preloss[0]
        
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        if FOCAL_LOSS:
            # 对于 LPKT，使用自定义的 reduction
            criterion = lambda y_pred, y_true: focal_loss(y_pred, y_true, reduction='sum')
        else:
            criterion = nn.BCELoss(reduction='none')
            
        loss = criterion(y, t).sum() if not FOCAL_LOSS else criterion(y, t)
    
    return loss


def model_forward(model, data, opt=None, rel=None,model_config={},data_label=0):
    # print(f"model_config5: {model_config}")
    global ii
    print(f"model forward{ii}")
    ii = ii+1
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "stablekt", "sparsekt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        

        y,_ = model(c.long(), r.long())
        # print(f"y shape{y.shape}")
        # print(f"cshft shape{cshft.long().shape}")

        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"y true shape{y.shape}")
        # print(f"y true {y}")
        
        ys.append(y) # first: yshft
        
    elif model_name in ["LSTM_Template"]:
        # print(f"cfloat:{c.float().shape}")
        y = model(c.long(), r.long())
        # print(f"y shape{y.shape}")
        # print(f"cshft shape{cshft.long().shape}")
        # print(f"model.num_c{model.num_c}")
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"selete shape{y.shape}")
        # print(f"y{y}")
        
        ys.append(y) # first: yshft
        
    elif model_name in ["CTNKT"]:
        # print(f"cfloat:{c.float().shape}")
        y = model(c.long(), r.long())
        # print(f"y shape{y.shape}")
        # print(f"cshft shape{cshft.long().shape}")
        # print(f"model.num_c{model.num_c}")
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"selete shape{y.shape}")
        # print(f"y{y}")
        
        ys.append(y) # first: yshft
    elif model_name in ["TCN_ABQR"]:
        # print(f"cc{cc.shape}")
        print(f"model_config: {model_config}")
        #扰动数据设计
        d = model_config['emb_size']
        print(f"d.shape: {d}")
        perturb_shape = (matrix.shape[0],d)
        step_size = 3e-2
        step_m = 3
        grad_clip = 15.0
        mm = 0.99
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
        pro_max = model_config['num_q']
        pro_embed = nn.Parameter(torch.ones((pro_max, d)))
        nn.init.xavier_uniform_(pro_embed)
        perturb.requires_grad_()
        print(f"perturb.shape: {perturb.shape}")
        pro_embed_trained = auto_load_BGRL(pro_embed,matrix,perturb,model_config,mm=mm)
        print(f"pro_embed_trained: {pro_embed_trained.shape}")
        # forward =  lambda perturb: model(q.long(), r.long(), c.long(), qshft.long(), cshft.long())
        y =  model(q.long(), r.long(), c.long(), qshft.long(), cshft.long(),pro_embed=pro_embed_trained)

        
        # print(f"bbb emb{emb_size}")
        
        # y, contrast_loss = forward(perturb)
        # y, contrast_loss = forward(perturb)
        
        
        # print("+"*100)
        # print(y.shape)
        # print("+"*100)
        # y = (y * one_hot(qshft.long(), model.num_q)).sum(-1)
        ys.append(y)
        # loss = cal_loss(model, ys, r, rshft, sm, preloss)
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
        # loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        # print("+"*100)
        # print(loss)
        # print("+"*100)
        # loss /= step_m
        # print("+"*100)
        # print(f"loss: {loss}")
        # print("+"*100)
        
        opt.zero_grad()

        # for _ in range(step_m - 1):
        #     loss.backward()
        #     # print(f"perturb.shape2: {perturb.shape}")
        #     # perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        #     # perturb.data = perturb_data.data
        #     # perturb.grad[:] = 0
        #     # perturb.grad.zero_()

        #     # predict, _, contrast_loss, c_reg_loss = forward(perturb)  # akt
        #     # y, contrast_loss = forward(perturb)
        #     pro_embed_trained = auto_load_BGRL(pro_embed,matrix,perturb,model_config,mm=mm)
        #     y =  model(q.long(), r.long(), c.long(), qshft.long(), cshft.long(),pro_embed=pro_embed_trained)
        #     ys.clear()
        #     # ys.append(y)
        #     ys.append(y[:,:])
        #     # loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        #     loss = cal_loss(model, ys, r, rshft, sm, preloss)
        #     loss /= step_m
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        # model.gcl.update_target_network(mm)  # EMA更新对抗学习的参数
        # auto_load_BGRL(pro_embed,matrix,perturb,model_config,mm=mm,update=True)
        # print("传播一次了")
        return loss
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["Transformer_Template","akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None,emb_sizess=128,model_config={},level1_loader=None,level2_loader=None,level3_loader=None):
    
    if MULTI_LEVEL_TRAIN == 1:
        # 使用多级训练
        return train_model_multilevel_custom(
    model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
    level1_loader,level2_loader, level3_loader, freeze_epoch, 
    test_loader, test_window_loader, save_model, data_config, fold, emb_sizess, model_config
)
    else:
        max_auc, best_epoch = 0, -1
        train_step = 0
        # print(f"model_config3: {model_config}")
        rel = None
        if model.model_name == "rkt":
            dpath = data_config["dpath"]
            dataset_name = dpath.split("/")[-1]
            tmp_folds = set(data_config["folds"]) - {fold}
            folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
            if dataset_name in ["algebra2005", "bridge2algebra2006"]:
                fname = "phi_dict" + folds_str + ".pkl"
                rel = pd.read_pickle(os.path.join(dpath, fname))
            else:
                fname = "phi_array" + folds_str + ".pkl" 
                rel = pd.read_pickle(os.path.join(dpath, fname))

        if model.model_name=='lpkt':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
        
        total_forward_time = 0.0  # 累计所有 forward 的时间
        total_epoch_time = 0.0    # 累计所有 epoch 的时间
        total_forward_count = 0    # 累计所有 forward 的次数

        for i in range(1, num_epochs + 1):
            epoch_start_time = time.time()  # 记录 epoch 开始时间
            loss_mean = []
            epoch_forward_time = 0.0       # 当前 epoch 的 forward 时间
            epoch_forward_count = 0         # 当前 epoch 的 forward 次数
            global ii
            ii = 0

                
                
            for data in train_loader:
                train_step += 1
                if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                    model.model.train()
                else:
                    model.train()
                
                # 记录 forward 开始时间
                forward_start = time.time()
                
                if model.model_name=='rkt':
                    loss = model_forward(model, data, rel)
                elif model.model_name=='TCN_ABQR':
                    print("entered_here!")
                    print(f"model_config4: {model_config}")
                    loss = model_forward(model, data,opt, rel,model_config=model_config)
                else:
                    loss = model_forward(model, data)
                
                # 记录 forward 结束时间
                forward_end = time.time()
                forward_duration = forward_end - forward_start

                # 更新时间统计
                epoch_forward_time += forward_duration
                epoch_forward_count += 1
                total_forward_time += forward_duration
                total_forward_count += 1
                if model.model_name != "TCN_ABQR":
                    opt.zero_grad()
                    loss.backward()  # 计算梯度
                if model.model_name == "rkt":
                    clip_grad_norm_(model.parameters(), model.grad_clip)
                if model.model_name == "dtransformer":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()  # 更新模型参数

                loss_mean.append(loss.detach().cpu().numpy())
                if model.model_name == "gkt" and train_step % 10 == 0:
                    text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                    debug_print(text=text, fuc_name="train_model")
            
            if model.model_name == 'lpkt':
                scheduler.step()  # 更新学习率

            loss_mean = np.mean(loss_mean)
            
            if model.model_name == 'rkt':
                auc, acc = evaluate(model, valid_loader, model.model_name, rel)
            else:
                auc, acc = evaluate(model, valid_loader, model.model_name)

            if auc > max_auc + 1e-3:
                if save_model:
                    torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + "_model.ckpt"))
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                if not save_model:
                    if test_loader is not None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")
                        testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                    if test_window_loader is not None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                        window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
                validauc, validacc = auc, acc
            
            epoch_end_time = time.time()  # 记录 epoch 结束时间
            epoch_duration = epoch_end_time - epoch_start_time
            total_epoch_time += epoch_duration

            # 计算当前 epoch 的平均 forward 时间
            avg_forward_time_epoch = epoch_forward_time / epoch_forward_count if epoch_forward_count > 0 else 0.0

            print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
            print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")
            print(f"            Avg Forward Time this Epoch: {avg_forward_time_epoch:.6f} seconds")

            # 检查是否提前停止
            if i - best_epoch >= 10:
                break

        # 训练结束后，计算所有 forward 的平均时间和所有 epoch 的平均时间
        overall_avg_forward_time = total_forward_time / total_forward_count if total_forward_count > 0 else 0.0
        avg_epoch_time = total_epoch_time / i if i > 0 else 0.0

        print("\n训练结束！")
        print(f"每个 forward 的总体平均时间: {overall_avg_forward_time:.6f} 秒")
        print(f"每个 epoch 的平均时间: {avg_epoch_time:.2f} 秒")

        # 保存最终结果
        results = {
            "Best Epoch": best_epoch,
            "Best AUC": max_auc,
            "Valid AUC": validauc,
            "Valid Accuracy": validacc,
            "Test AUC": testauc,
            "Test Accuracy": testacc,
            "Window Test AUC": window_testauc,
            "Window Test Accuracy": window_testacc,
            "Overall Avg Forward Time": overall_avg_forward_time,
            "Avg Epoch Time": avg_epoch_time
        }
        save_results_to_file(results, ckpt_path)

        return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch


def auto_load_BGRL(pro_embed, Q_matrix, perb,model_config,mm,force_retrain=False):
    pro_embed = pro_embed.to(device)
    Q_matrix = Q_matrix.to(device)
    print("tranning embedding")
    dataset_path = model_config["dataset_path"]
    pro_embed_path = os.path.join(model_config['dataset_path'],"pro_embed.pt")
    d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2 = (model_config[k] for k in ['emb_size', 'p', 'drop_feat1', 'drop_feat2', 'drop_edge1', 'drop_edge2'])
    print(f"d.shape2: {d}")
    model = BGRL(d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2).to(device)
    # print(f"not os.path.exists(pro_embed_path): {not os.path.exists(pro_embed_path)}")
    if not os.path.exists(pro_embed_path) or force_retrain: #不存在，则训练
        print("不存在嵌入，开始训练")
        
        opt = torch.optim.Adam(model.parameters(),lr=1e-4)
        try:
            # pro_embed, contrast_loss = model(pro_embed, Q_matrix, perb)
            model.train()
            for epoch in range(100):
                _, contrast_loss = model(pro_embed, Q_matrix, perb)
                opt.zero_grad()
                contrast_loss.backward()
                # contrast_loss = 0.1 * contrast_loss
                opt.step()
                model.update_target_network(mm)
                if epoch %10 == 0:
                    print(f"Epoch {epoch:03d} | Loss: {contrast_loss.item():.4f}")
            torch.save(model.state_dict(),os.path.join(model_config['dataset_path'],"Graph.pt"))
            with torch.no_grad():
                pro_embed,_ = model(pro_embed, Q_matrix, perb)
                torch.save(pro_embed,pro_embed_path)
            print(f"模型训练完成保存到{pro_embed_path}")
        except Exception as e:
            # 打印完整错误堆栈
            print(f"\n❌ ABQR嵌入矩阵训练失败，详细错误信息：")
            print(traceback.format_exc())
            
            # 补充关键变量状态检查
            print("\n关键变量状态检查：")
            print(f"model_config 是否存在: {'dataset_path' in model_config}")
            if 'dataset_path' in model_config:
                print(f"保存路径是否存在: {os.path.exists(model_config['dataset_path'])}")
            print(f"pro_embed shape: {pro_embed.shape if isinstance(pro_embed, torch.Tensor) else 'Not Tensor'}")
            print(f"设备是否一致: model_device={next(model.parameters()).device}, perb_device={perb.device}")


    else:
        print("存在嵌入,直接读取")
        pass #存在，直接读取
    
    return torch.load(pro_embed_path)

import os
import argparse
import json
import copy
import torch
import pandas as pd
import numpy as np
from pykt.config import ERR_PATH, stu_pk, SET_TARGET_STU
from pykt.models import evaluate, evaluate_question, load_model
from pykt.datasets import init_test_datasets
import pykt.config as config_module

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

def extract_student_stats(stats_file_path):
    """从学生统计文件中提取统计数据"""
    if not os.path.exists(stats_file_path):
        print(f"警告：统计文件不存在: {stats_file_path}")
        return {
            'record_count': -1,
            'total_questions': -1,
            'avg_questions_per_concept': -1.0,
            'max_questions_per_concept': -1,
            'min_questions_per_concept': -1,
            'questions_range': -1,
            'overall_accuracy': -1.0,
            'accuracy_range': -1.0,
            'accuracy_variance': -1.0,
            'max_accuracy': -1.0,
            'min_accuracy': -1.0
        }
    
    try:
        # 读取统计文件
        stats_df = pd.read_csv(stats_file_path)
        
        # 如果空文件
        if stats_df.empty:
            return {
                'record_count': 0,
                'total_questions': 0,
                'avg_questions_per_concept': 0.0,
                'max_questions_per_concept': 0,
                'min_questions_per_concept': 0,
                'questions_range': 0,
                'overall_accuracy': 0.0,
                'accuracy_range': 0.0,
                'accuracy_variance': 0.0,
                'max_accuracy': 0.0,
                'min_accuracy': 0.0
            }
        
        # 提取第一条记录的统计信息（所有行数据相同）
        row = stats_df.iloc[0]
        
        return {
            'record_count': row.get('record_count', stats_df.shape[0]),
            'total_questions': row.get('total_questions', -1),
            'avg_questions_per_concept': row.get('avg_questions_per_concept', -1.0),
            'max_questions_per_concept': row.get('max_questions_per_concept', -1),
            'min_questions_per_concept': row.get('min_questions_per_concept', -1),
            'questions_range': row.get('questions_range', -1),
            'overall_accuracy': row.get('overall_accuracy', -1.0),
            'accuracy_range': row.get('accuracy_range', -1.0),
            'accuracy_variance': row.get('accuracy_variance', -1.0),
            'max_accuracy': row.get('max_accuracy', -1.0),
            'min_accuracy': row.get('min_accuracy', -1.0)
        }
    except Exception as e:
        print(f"读取统计文件时出错: {e}")
        return {}

def evaluate_single_student(params, student_id):
    """评估单个学生的函数"""
    print(f"\n开始评估学生 {student_id}")
    
    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]
    
    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, stu_id=student_id+3)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"学生 {student_id}: 开始预测模型 {model_name}, embtype: {emb_type}, dataset_name: {dataset_name}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    # 在评估前提取学生统计信息
    stats_file_path = os.path.join(data_config["dpath"], f"top_{student_id}_student.csv")
    student_stats = extract_student_stats(stats_file_path)
    print(f"学生 {student_id}: 已加载统计信息")
    
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
    
    dres = {
        "student_id": student_id,
    }

    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, f"{model.emb_type}_test_question_window_predictions_student_{student_id}.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc" + key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc" + key] = qw_testaccs[key]

    raw_config = json.load(open(os.path.join(save_dir, "config.json")))
    dres.update(raw_config['params'])
    
    # 添加学生统计信息到评估结果
    dres.update(student_stats)
    print(f"学生 {student_id}: 已添加统计信息到评估结果")

    # 输出关键指标
    if 'windowauclate_mean' in dres:
        print(f"学生 {student_id}: windowauclate_mean: {dres['windowauclate_mean']}")
    if 'windowacclate_mean' in dres:
        print(f"学生 {student_id}: windowacclate_mean: {dres['windowacclate_mean']}")
    
    # 将评估结果保存到单独的JSON文件
    results_path = os.path.join(save_dir, f"evaluation_results_{student_id}.json")
    try:
        with open(results_path, "w") as fout:
            json.dump(dres, fout, indent=4, ensure_ascii=False)
        print(f"学生 {student_id}: 评估结果已保存到 {results_path}")
    except Exception as e:
        print(f"学生 {student_id}: 保存评估结果时出错: {e}")

    return dres

def evaluate_overall(params):
    """评估整体（所有学生一起）的函数"""
    print(f"\n开始整体评估")
    
    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]
    
    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, stu_id=0)  # 0表示所有学生
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"整体评估: 开始预测模型 {model_name}, embtype: {emb_type}, dataset_name: {dataset_name}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))

    dres = {}

    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, f"{model.emb_type}_test_question_window_predictions_overall.txt")
        print(f"整体评估: stu_id=0 (所有学生)")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc" + key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc" + key] = qw_testaccs[key]

    raw_config = json.load(open(os.path.join(save_dir, "config.json")))
    dres.update(raw_config['params'])

    print(f"整体评估结果: {dres}")
    if 'windowauclate_mean' in dres:
        print(f"整体评估 windowauclate_mean: {dres['windowauclate_mean']}")
    if 'windowacclate_mean' in dres:
        print(f"整体评估 windowacclate_mean: {dres['windowacclate_mean']}")
    
    # 将整体评估结果保存到JSON文件
    results_path = os.path.join(save_dir, "evaluation_results_overall.json")
    try:
        with open(results_path, "w") as fout:
            json.dump(dres, fout, indent=4, ensure_ascii=False)
        print(f"整体评估结果已保存到 {results_path}")
    except Exception as e:
        print(f"保存整体评估结果时出错: {e}")

    return dres

def calculate_diversity_metric(auc_values, method='normalized_cv'):
    """
    计算数据差异指标（0-1之间）
    
    Args:
        auc_values: AUC值列表
        method: 计算方法
            - 'normalized_cv': 归一化变异系数
            - 'normalized_range': 归一化范围
            - 'normalized_std': 标准化标准差 2*std/(max-min)
            - 'normalized_variance': 标准化方差 variance/(max-min)^2
            - 'entropy_based': 基于熵的方法
    
    Returns:
        float: 0-1之间的差异指标，1表示差异很大
    """
    # 过滤掉NaN值和无效值
    clean_values = []
    for val in auc_values:
        if not (pd.isna(val) or np.isnan(val) or val == -1):
            clean_values.append(val)
    
    if len(clean_values) <= 1:
        print(f"警告：有效AUC值数量不足 ({len(clean_values)})，无法计算差异指标")
        return 0.0
    
    auc_array = np.array(clean_values)
    
    if method == 'normalized_cv':
        # 变异系数方法：CV = std/mean，然后归一化到0-1
        mean_val = np.mean(auc_array)
        std_val = np.std(auc_array)
        if mean_val == 0:
            return 1.0
        cv = std_val / mean_val
        # 使用sigmoid函数将CV映射到0-1
        return 1 - np.exp(-cv)
    
    elif method == 'normalized_range':
        # 归一化范围方法：(max-min)/max
        max_val = np.max(auc_array)
        min_val = np.min(auc_array)
        if max_val == 0:
            return 0.0
        return (max_val - min_val) / max_val
    
    elif method == 'normalized_std':
        # 标准化标准差方法：2*std/(max-min)
        # 这个方法更直观，2倍标准差通常覆盖95%的数据（假设正态分布）
        max_val = np.max(auc_array)
        min_val = np.min(auc_array)
        std_val = np.std(auc_array)
        
        if max_val == min_val:  # 所有值相同
            return 0.0
        
        normalized_std = (2 * std_val) / (max_val - min_val)
        # 将结果映射到0-1范围，值越接近1表示差异越大
        # 理论上normalized_std可能超过1，所以使用tanh函数进行soft限制
        return np.tanh(normalized_std)
    elif method == 'std':
        
        std_val = np.std(auc_array)
    
        return std_val
    elif method == 'normalized_variance':
        # 标准化方差方法：variance/(max-min)^2
        max_val = np.max(auc_array)
        min_val = np.min(auc_array)
        var_val = np.var(auc_array)
        
        if max_val == min_val:  # 所有值相同
            return 0.0
        normalized_data = (auc_array - min_val) / (max_val - min_val)
        std_dev = 2*np.std(normalized_data)

        return std_dev
    
    elif method == 'entropy_based':
        # 基于熵的方法：将数据分桶计算信息熵
        # 分成10个桶
        hist, _ = np.histogram(auc_array, bins=10)
        hist = hist + 1e-10  # 避免log(0)
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        max_entropy = np.log2(10)  # 最大熵
        return entropy / max_entropy
    
    else:
        raise ValueError(f"未知的方法: {method}")

def calculate_performance_metric(overall_auc, target_value, method='relative_absolute_diff'):
    """
    计算性能指标与目标值的差异（0-1之间）
    
    Args:
        overall_auc: 整体AUC值
        target_value: 目标值
        method: 计算方法
            - 'relative_absolute_diff': 相对绝对差异
            - 'exponential_decay': 指数衰减差异
            - 'sigmoid_diff': sigmoid差异
    
    Returns:
        float: 0-1之间的差异指标，1表示差异很大
    """
    # 检查overall_auc是否为NaN或无效值
    if pd.isna(overall_auc) or np.isnan(overall_auc) or overall_auc == -1:
        print(f"警告：整体AUC值无效 ({overall_auc})，性能指标设为1.0")
        return 1.0
    
    if method == 'relative_absolute_diff':
        # 相对绝对差异：|overall_auc - target_value| / max(overall_auc, target_value)
        if max(overall_auc, target_value) == 0:
            return 1.0 if overall_auc != target_value else 0.0
        return abs(overall_auc - target_value) / (overall_auc+target_value)
    
    elif method == 'exponential_decay':
        # 指数衰减：1 - exp(-|overall_auc - target_value| / target_value)
        if target_value == 0:
            return 1.0 if overall_auc != 0 else 0.0
        return 1 - np.exp(-abs(overall_auc - target_value) / target_value)
    
    elif method == 'sigmoid_diff':
        # sigmoid差异：使用sigmoid函数
        diff = abs(overall_auc - target_value)
        return 2 / (1 + np.exp(-10 * diff)) - 1
    
    else:
        raise ValueError(f"未知的方法: {method}")

def main(params):
    """主函数，执行完整的评估流程"""
    total_students = params.get('total_students', 770)
    start_student = params.get('start_student', 1)
    save_dir = params["save_dir"]
    
    # 评估参数
    target_value = params.get('target_value', 0.8)  # 目标AUC值
    diversity_weight = params.get('diversity_weight', 0.5)  # 差异指标权重
    performance_weight = params.get('performance_weight', 0.5)  # 性能指标权重
    diversity_method = params.get('diversity_method', 'normalized_cv')  # 差异计算方法
    performance_method = params.get('performance_method', 'relative_absolute_diff')  # 性能差异计算方法
    
    if params['use_wandb'] == 1:
        import wandb
        with open("../configs/wandb.json") as fin:
            wandb_config = json.load(fin)
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")
    
    # 存储所有学生的评估结果
    all_student_results = []
    
    print(f"=" * 60)
    print(f"开始完整评估流程")
    print(f"总学生数: {total_students}, 起始学生: {start_student}")
    print(f"目标AUC值: {target_value}")
    print(f"差异指标权重: {diversity_weight}, 性能指标权重: {performance_weight}")
    print(f"差异计算方法: {diversity_method}")
    print(f"  - normalized_cv: 归一化变异系数 (CV映射)")
    print(f"  - normalized_range: 归一化范围 (max-min)/max")
    print(f"  - normalized_std: 标准化标准差 2*std/(max-min) [推荐]")
    print(f"  - normalized_variance: 标准化方差 var/(max-min)^2")
    print(f"  - entropy_based: 基于信息熵")
    print(f"性能差异计算方法: {performance_method}")
    print(f"=" * 60)
    
    # 第一阶段：批量评估所有学生
    print(f"\n{'='*20} 第一阶段：单个学生批量评估 {'='*20}")
    
    for student_id in range(start_student, total_students + 1):
        try:
            # 评估单个学生
            result = evaluate_single_student(params, student_id)
            all_student_results.append(result)
            
            print(f"完成学生 {student_id} 的评估 ({student_id - start_student + 1}/{total_students - start_student + 1})")
            
        except Exception as e:
            print(f"评估学生 {student_id} 时出错: {e}")
            # 创建一个错误记录
            error_result = {
                "student_id": student_id,
                "error": str(e),
                "windowauclate_mean": -1,
                "windowacclate_mean": -1
            }
            all_student_results.append(error_result)
            continue
    
    # 第二阶段：整体评估
    print(f"\n{'='*20} 第二阶段：整体评估 {'='*20}")
    
    try:
        overall_result = evaluate_overall(params)
        overall_auc = overall_result.get('windowauclate_mean', -1)
        print(f"整体AUC: {overall_auc}")
    except Exception as e:
        print(f"整体评估时出错: {e}")
        overall_auc = -1
        overall_result = {"error": str(e)}
    
    # 第三阶段：计算差异指标和综合指标
    print(f"\n{'='*20} 第三阶段：指标计算 {'='*20}")
    
    # 提取有效的AUC值，排除NaN和-1
    all_aucs = []
    nan_count = 0
    error_count = 0
    
    for result in all_student_results:
        auc_value = result.get('windowauclate_mean', -1)
        if auc_value == -1:
            error_count += 1
        elif pd.isna(auc_value) or np.isnan(auc_value):
            nan_count += 1
        else:
            all_aucs.append(auc_value)
    
    valid_aucs = all_aucs
    
    print(f"AUC值统计:")
    print(f"  - 总学生数: {len(all_student_results)}")
    print(f"  - 错误/未计算: {error_count}")
    print(f"  - NaN值: {nan_count}")
    print(f"  - 有效值: {len(valid_aucs)}")
    
    if len(valid_aucs) > 0:
        # 计算差异指标a（学生间差异）
        diversity_metric = calculate_diversity_metric(valid_aucs, method=diversity_method)
        
        # 输出详细的差异指标信息
        print(f"学生差异指标a ({diversity_method}): {diversity_metric:.4f}")
        
        # 提供额外的统计信息帮助理解
        auc_array = np.array(valid_aucs)
        auc_mean = np.mean(auc_array)
        auc_std = np.std(auc_array)
        auc_max = np.max(auc_array)
        auc_min = np.min(auc_array)
        auc_range = auc_max - auc_min
        
        print(f"  详细统计:")
        print(f"    - 均值: {auc_mean:.4f}")
        print(f"    - 标准差: {auc_std:.4f}")
        print(f"    - 范围: [{auc_min:.4f}, {auc_max:.4f}]")
        print(f"    - 极差: {auc_range:.4f}")
        
        if diversity_method == 'normalized_std':
            raw_metric = (2 * auc_std) / auc_range if auc_range > 0 else 0
            print(f"    - 原始标准化标准差: {raw_metric:.4f}")
            print(f"    - tanh映射后: {diversity_metric:.4f}")
        elif diversity_method == 'normalized_cv':
            cv = auc_std / auc_mean if auc_mean > 0 else float('inf')
            print(f"    - 变异系数: {cv:.4f}")
        elif diversity_method == 'normalized_variance':
            var = np.var(auc_array)
            raw_metric = var / (auc_range ** 2) if auc_range > 0 else 0
            print(f"    - 方差: {var:.4f}")
            print(f"    - 原始标准化方差: {raw_metric:.4f}")
            print(f"    - tanh映射后: {diversity_metric:.4f}")
        
        # 如果用户选择显示所有方法的对比
        if params.get('show_all_methods', False):
            print(f"\n  所有差异指标方法对比:")
            all_methods = ["normalized_cv", "normalized_range", "normalized_std", "normalized_variance", "entropy_based"]
            for method in all_methods:
                try:
                    metric_value = calculate_diversity_metric(valid_aucs, method=method)
                    marker = " ★" if method == diversity_method else ""
                    print(f"    - {method}: {metric_value:.4f}{marker}")
                except Exception as e:
                    print(f"    - {method}: 计算失败 ({e})")
        
        # 计算性能指标b（整体性能与目标的差异）
        if overall_auc != -1:
            performance_metric = calculate_performance_metric(overall_auc, target_value, method=performance_method)
            print(f"性能差异指标b ({performance_method}): {performance_metric:.4f}")
            
            # 计算综合指标
            combined_metric = (diversity_weight * diversity_metric + performance_weight * performance_metric) / (diversity_weight + performance_weight)
            print(f"综合指标 (加权平均): {combined_metric:.4f}")
        else:
            performance_metric = 1.0  # 如果整体评估失败，设为最大差异
            combined_metric = diversity_metric  # 只考虑差异指标
            print(f"性能差异指标b: 1.0000 (整体评估失败)")
            print(f"综合指标 (仅差异): {combined_metric:.4f}")
    else:
        diversity_metric = 1.0
        performance_metric = 1.0
        combined_metric = 1.0
        print(f"警告：没有有效的学生AUC数据")
        print(f"所有指标设为最大值: 1.0000")
    
    # 第四阶段：保存结果
    print(f"\n{'='*20} 第四阶段：保存结果 {'='*20}")
    
    # 保存学生批量评估结果
    if all_student_results:
        df = pd.DataFrame(all_student_results)
        csv_path = os.path.join(save_dir, "batch_evaluation_results.csv")
        
        try:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"学生评估结果已保存到: {csv_path}")
        except Exception as e:
            print(f"保存学生评估CSV文件时出错: {e}")
    
    # 保存综合分析结果
    analysis_result = {
        "evaluation_params": {
            "total_students": total_students,
            "start_student": start_student,
            "target_value": target_value,
            "diversity_weight": diversity_weight,
            "performance_weight": performance_weight,
            "diversity_method": diversity_method,
            "performance_method": performance_method,
            "show_all_methods": params.get('show_all_methods', False)
        },
        "student_evaluation": {
            "total_evaluated": len(all_student_results),
            "valid_aucs_count": len(valid_aucs),
            "nan_count": nan_count,
            "error_count": error_count,
            "mean_auc": np.mean(valid_aucs) if valid_aucs else -1,
            "std_auc": np.std(valid_aucs) if valid_aucs else -1,
            "min_auc": np.min(valid_aucs) if valid_aucs else -1,
            "max_auc": np.max(valid_aucs) if valid_aucs else -1,
            "median_auc": np.median(valid_aucs) if valid_aucs else -1
        },
        "overall_evaluation": {
            "overall_auc": overall_auc,
            "evaluation_success": overall_auc != -1
        },
        "metrics": {
            "diversity_metric_a": diversity_metric,
            "performance_metric_b": performance_metric,
            "combined_metric": combined_metric,
            "diversity_details": {
                "method_used": diversity_method,
                "auc_mean": np.mean(valid_aucs) if valid_aucs else -1,
                "auc_std": np.std(valid_aucs) if valid_aucs else -1,
                "auc_range": (np.max(valid_aucs) - np.min(valid_aucs)) if valid_aucs else -1,
                "coefficient_of_variation": (np.std(valid_aucs) / np.mean(valid_aucs)) if valid_aucs and np.mean(valid_aucs) > 0 else -1
            },
            "performance_details": {
                "method_used": performance_method,
                "target_value": target_value,
                "actual_overall_auc": overall_auc,
                "absolute_difference": abs(overall_auc - target_value) if overall_auc != -1 else -1,
                "relative_difference": (abs(overall_auc - target_value) / target_value) if overall_auc != -1 and target_value > 0 else -1
            }
        },
        "detailed_results": {
            "student_results": all_student_results,
            "overall_result": overall_result
        }
    }
    
    # 如果显示所有方法，则添加方法对比结果
    if params.get('show_all_methods', False) and len(valid_aucs) > 0:
        method_comparison = {}
        all_methods = ["normalized_cv", "normalized_range", "normalized_std", "normalized_variance", "entropy_based"]
        for method in all_methods:
            try:
                metric_value = calculate_diversity_metric(valid_aucs, method=method)
                method_comparison[method] = metric_value
            except Exception as e:
                method_comparison[method] = f"计算失败: {str(e)}"
        analysis_result["metrics"]["all_methods_comparison"] = method_comparison
    
    analysis_path = os.path.join(save_dir, "comprehensive_analysis.json")
    try:
        with open(analysis_path, "w") as fout:
            json.dump(analysis_result, fout, indent=4, ensure_ascii=False)
        print(f"综合分析结果已保存到: {analysis_path}")
    except Exception as e:
        print(f"保存综合分析结果时出错: {e}")
    
    # 第五阶段：输出最终总结
    print(f"\n{'='*20} 评估完成总结 {'='*20}")
    print(f"✓ 学生评估统计:")
    print(f"  - 总计学生: {len(all_student_results)}")
    print(f"  - 有效AUC: {len(valid_aucs)}")
    print(f"  - NaN值: {nan_count}")
    print(f"  - 错误/未计算: {error_count}")
    if valid_aucs:
        print(f"  - 有效AUC统计: 均值={np.mean(valid_aucs):.4f}, 标准差={np.std(valid_aucs):.4f}")
        print(f"  - 有效AUC范围: [{np.min(valid_aucs):.4f}, {np.max(valid_aucs):.4f}]")
        print(f"  - 有效AUC中位数: {np.median(valid_aucs):.4f}")
    print(f"✓ 整体评估: AUC = {overall_auc:.4f}" if overall_auc != -1 else "✗ 整体评估失败")
    print(f"✓ 指标计算:")
    print(f"  - 学生差异指标a ({diversity_method}): {diversity_metric:.4f}")
    if diversity_method == 'normalized_std':
        print(f"    └─ 标准化标准差：2×标准差/(最大值-最小值)")
    elif diversity_method == 'normalized_variance':
        print(f"    └─ 标准化方差：方差/(最大值-最小值)²")
    print(f"  - 性能差异指标b ({performance_method}): {performance_metric:.4f}")
    print(f"  - 综合指标: {combined_metric:.4f}")
    print(f"{'='*60}")
    
    if params['use_wandb'] == 1:
        wandb.log(analysis_result["metrics"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256, help="批处理大小")
    parser.add_argument("--save_dir", type=str, default="saved_model", help="模型保存目录")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion", help="融合类型")
    parser.add_argument("--use_wandb", type=int, default=0, help="是否使用wandb")
    parser.add_argument("--total_students", type=int, default=770, help="总学生数量")
    parser.add_argument("--start_student", type=int, default=1, help="开始评估的学生ID")
    
    # 新增参数：综合指标计算
    parser.add_argument("--target_value", type=float, default=0.8, help="目标AUC值")
    parser.add_argument("--diversity_weight", type=float, default=0.5, help="差异指标权重")
    parser.add_argument("--performance_weight", type=float, default=0.5, help="性能指标权重")
    parser.add_argument("--diversity_method", type=str, default="std",
                      choices=["normalized_cv", "normalized_range", "normalized_std", "normalized_variance", "entropy_based"],
                      help="差异指标计算方法")
    parser.add_argument("--performance_method", type=str, default="relative_absolute_diff",
                      choices=["relative_absolute_diff", "exponential_decay", "sigmoid_diff"],
                      help="性能差异计算方法")
    parser.add_argument("--show_all_methods", action="store_true", default=False,
                      help="显示所有差异指标计算方法的对比结果")

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
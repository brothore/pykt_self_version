#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
import pandas as pd
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect evaluation results from best model directories.')
    parser.add_argument('--project_name', type=str, required=True,
                        help='Wandb项目名称，例如 "AKT_CONV_DYNROUTINE_V5_A9"')
    parser.add_argument('--start_sweep', type=int, required=True,
                        help='起始折叠编号（包含），例如 0')
    parser.add_argument('--end_sweep', type=int, required=True,
                        help='结束折叠编号（不包含），例如 5')
    parser.add_argument('--timeout', type=int, default=60,
                        help='等待所有 evaluation_results.json 生成的最大时间（分钟），默认为 60')
    parser.add_argument('--interval', type=int, default=60,
                        help='检查文件存在性的间隔时间（秒），默认为 60')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='输出的 CSV 文件名，默认为 "evaluation_summary_<START_SWEEP>_<END_SWEEP>.csv"')
    args = parser.parse_args()
    return args

def load_best_models(save_dir, start_sweep, end_sweep):
    json_filename = f'best_models_per_fold.json'
    json_path = os.path.join(save_dir, json_filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件 '{json_path}' 不存在。请确保之前的脚本已正确运行。")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        best_models = json.load(f)
    
    # 过滤指定范围内的折叠
    filtered_best_models = {int(fold): path for fold, path in best_models.items()
                            if start_sweep <= int(fold) < end_sweep}
    
    if not filtered_best_models:
        raise ValueError(f"在指定的折叠范围 START_SWEEP={start_sweep} 到 END_SWEEP={end_sweep} 内，没有找到对应的 model_save_path。")
    
    return filtered_best_models

def wait_for_files(model_paths, timeout_minutes, check_interval_seconds):
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    remaining = timeout_seconds
    missing_files = {fold: path for fold, path in model_paths.items()}
    
    print(f"开始等待 {len(missing_files)} 个 evaluation_results.json 文件生成，最长等待时间 {timeout_minutes} 分钟。")
    
    while remaining > 0 and missing_files:
        for fold, path in list(missing_files.items()):
            eval_json_path = os.path.join(path, 'evaluation_results.json')
            if os.path.exists(eval_json_path):
                print(f"找到 Fold {fold} 的 evaluation_results.json。")
                del missing_files[fold]
        
        if not missing_files:
            print("所有 evaluation_results.json 文件均已生成。")
            break
        
        print(f"等待中... 还有 {len(missing_files)} 个文件未生成。剩余时间: {int(remaining/60)} 分 {int(remaining%60)} 秒")
        time.sleep(check_interval_seconds)
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
    
    if missing_files:
        print(f"超时等待后，仍有 {len(missing_files)} 个文件未生成：")
        for fold, path in missing_files.items():
            print(f"  Fold {fold}: {os.path.join(path, 'evaluation_results.json')}")
        raise TimeoutError("未能在指定时间内生成所有 evaluation_results.json 文件。")
    else:
        print("所有需要的 evaluation_results.json 文件均已存在。")

def extract_metrics(model_paths):
    metrics = []
    for fold, path in model_paths.items():
        eval_json_path = os.path.join(path, 'evaluation_results.json')
        if not os.path.exists(eval_json_path):
            print(f"警告：Fold {fold} 的 evaluation_results.json 不存在，跳过。")
            continue
        
        with open(eval_json_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        
        windowauclate_mean = eval_results.get('windowauclate_mean', None)
        windowacclate_mean = eval_results.get('windowacclate_mean', None)
        
        if windowauclate_mean is None or windowacclate_mean is None:
            print(f"警告：Fold {fold} 的 evaluation_results.json 中缺少 'windowauclate_mean' 或 'windowacclate_mean'。")
            continue
        
        metrics.append({
            'fold': fold,
            'windowauclate_mean': windowauclate_mean,
            'windowacclate_mean': windowacclate_mean
        })
    
    if not metrics:
        raise ValueError("未能提取到任何折叠的指标数据。")
    
    return metrics

def compute_statistics(metrics):
    df = pd.DataFrame(metrics)
    summary = {
        'metric': ['windowauclate_mean', 'windowacclate_mean'],
        'mean': [df['windowauclate_mean'].mean(), df['windowacclate_mean'].mean()],
        'std': [df['windowauclate_mean'].std(), df['windowacclate_mean'].std()]
    }
    summary_df = pd.DataFrame(summary)
    return summary_df, df

def save_to_csv(summary_df, individual_df, save_dir, start_sweep, end_sweep, output_csv):
    if output_csv is None:
        output_csv = f'evaluation_summary_{start_sweep}_{end_sweep}.csv'
    csv_path = os.path.join(save_dir, output_csv)
    
    if os.path.exists(csv_path):
        print(f"文件 '{csv_path}' 已存在，进行覆盖。")
    
    # 合并 summary 和 individual 数据到一个 CSV 文件中
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Summary Metrics\n")
        summary_df.to_csv(f, index=False)
        f.write("\nIndividual Folds Metrics\n")
        individual_df.to_csv(f, index=False)
    
    print(f"统计结果已保存为 '{csv_path}'。")

def main():
    args = parse_arguments()
    
    # 获取必要的环境变量
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    if not WANDB_API_KEY:
        raise ValueError("请设置环境变量 WANDB_API_KEY，例如: export WANDB_API_KEY='您的API密钥'")
    
    PROJECT_NAME = args.project_name
    START_SWEEP = args.start_sweep
    END_SWEEP = args.end_sweep
    TIMEOUT = args.timeout
    INTERVAL = args.interval
    OUTPUT_CSV = args.output_csv
    
    # 设置保存目录
    save_dir = os.path.join('results', PROJECT_NAME)
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"保存目录 '{save_dir}' 不存在。请确保之前的脚本已正确运行。")
    
    # 加载最佳模型路径
    try:
        best_models = load_best_models(save_dir, START_SWEEP, END_SWEEP)
    except Exception as e:
        print(f"错误：{e}")
        return
    
    # 等待所有 evaluation_results.json 文件生成
    try:
        wait_for_files(best_models, TIMEOUT, INTERVAL)
    except Exception as e:
        print(f"错误：{e}")
        return
    
    # 提取指标
    try:
        metrics = extract_metrics(best_models)
    except Exception as e:
        print(f"错误：{e}")
        return
    
    # 计算统计量
    summary_df, individual_df = compute_statistics(metrics)
    print("\n统计结果：")
    print(summary_df)
    
    # 保存结果到 CSV
    try:
        save_to_csv(summary_df, individual_df, save_dir, START_SWEEP, END_SWEEP, OUTPUT_CSV)
    except Exception as e:
        print(f"错误：{e}")
        return

if __name__ == "__main__":
    main()

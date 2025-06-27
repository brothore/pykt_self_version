import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def process_sequence_string(seq_str):
    """
    处理序列字符串，将其转换为数值列表
    
    Args:
        seq_str: 字符串格式的序列，如 "0.8,0.9,0.7"
    
    Returns:
        list: 数值列表
    """
    if pd.isna(seq_str) or seq_str == '':
        return []
    
    # 去掉可能的方括号和空格
    seq_str = str(seq_str).strip('[]').replace(' ', '')
    
    try:
        # 分割字符串并转换为浮点数
        values = [float(x.strip()) for x in seq_str.split(',') if x.strip()]
        return values
    except ValueError as e:
        print(f"Warning: Could not parse sequence '{seq_str}': {e}")
        return []

def process_response_sequence(seq_str):
    """
    处理响应序列，将其转换为整数列表
    
    Args:
        seq_str: 字符串格式的响应序列
    
    Returns:
        list: 整数列表
    """
    if pd.isna(seq_str) or seq_str == '':
        return []
    
    # 去掉可能的方括号和空格
    seq_str = str(seq_str).strip('[]').replace(' ', '')
    
    try:
        # 分割字符串并转换为整数，过滤掉-1（填充值）
        values = [int(x.strip()) for x in seq_str.split(',') if x.strip() and int(x.strip()) != -1]
        return values
    except ValueError as e:
        print(f"Warning: Could not parse response sequence '{seq_str}': {e}")
        return []

def calculate_auc_for_group(predictions, responses):
    """
    为一组数据计算AUC
    
    Args:
        predictions: 预测概率列表
        responses: 真实标签列表
    
    Returns:
        float: AUC值，如果无法计算则返回NaN
    """
    if len(predictions) == 0 or len(responses) == 0:
        return np.nan
    
    # 确保预测值和真实值长度一致
    min_len = min(len(predictions), len(responses))
    pred_values = predictions[:min_len]
    true_values = responses[:min_len]
    
    # 过滤掉无效值
    valid_indices = []
    for i in range(len(pred_values)):
        if not np.isnan(pred_values[i]) and true_values[i] in [0, 1]:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return np.nan
    
    valid_pred = [pred_values[i] for i in valid_indices]
    valid_true = [true_values[i] for i in valid_indices]
    
    # 检查是否只有一个类别
    if len(set(valid_true)) <= 1:
        return np.nan
    
    try:
        return roc_auc_score(valid_true, valid_pred)
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        return np.nan

def add_auc_to_csv(csv_path, uid_col='uid', predictions_col='predictions', 
                   responses_col='responses', output_path=None):
    """
    为CSV文件添加AUC列，按uid分组计算
    
    Args:
        csv_path: 输入CSV文件路径
        uid_col: 用户ID列名
        predictions_col: 预测结果列名
        responses_col: 真实响应列名
        output_path: 输出文件路径，如果为None则覆盖原文件
    
    Returns:
        DataFrame: 处理后的DataFrame
    """
    print(f"Reading CSV file: {csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # 检查必要的列是否存在
    required_cols = [uid_col, predictions_col, responses_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"Processing {len(df)} rows...")
    
    # 处理预测结果和响应序列
    print("Parsing prediction sequences...")
    df['parsed_predictions'] = df[predictions_col].apply(process_sequence_string)
    
    print("Parsing response sequences...")
    df['parsed_responses'] = df[responses_col].apply(process_response_sequence)
    
    # 按uid分组计算AUC
    print("Calculating AUC for each user...")
    auc_results = []
    
    grouped = df.groupby(uid_col)
    total_groups = len(grouped)
    
    for i, (uid, group) in enumerate(grouped):
        if i % 100 == 0:  # 每处理100个用户打印进度
            print(f"Processing user {i+1}/{total_groups}: {uid}")
        
        # 合并该用户的所有预测和响应
        all_predictions = []
        all_responses = []
        
        for _, row in group.iterrows():
            pred_list = row['parsed_predictions']
            resp_list = row['parsed_responses']
            
            all_predictions.extend(pred_list)
            all_responses.extend(resp_list)
        
        # 计算该用户的AUC
        user_auc = calculate_auc_for_group(all_predictions, all_responses)
        
        # 为该用户的所有行添加AUC值
        for idx in group.index:
            auc_results.append(user_auc)
    
    # 添加AUC列
    df['auc'] = auc_results
    
    # 删除临时列
    df = df.drop(['parsed_predictions', 'parsed_responses'], axis=1)
    
    # 统计结果
    valid_aucs = df['auc'].dropna()
    print(f"\nAUC Calculation Results:")
    print(f"Total users: {df[uid_col].nunique()}")
    print(f"Users with valid AUC: {len(valid_aucs)}")
    print(f"Users with invalid AUC: {df[uid_col].nunique() - len(valid_aucs)}")
    
    if len(valid_aucs) > 0:
        print(f"Mean AUC: {valid_aucs.mean():.4f}")
        print(f"Median AUC: {valid_aucs.median():.4f}")
        print(f"Min AUC: {valid_aucs.min():.4f}")
        print(f"Max AUC: {valid_aucs.max():.4f}")
    
    # 保存文件
    output_file = output_path if output_path else csv_path
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df

def analyze_auc_distribution(df, uid_col='uid', auc_col='auc'):
    """
    分析AUC分布情况
    
    Args:
        df: 包含AUC的DataFrame
        uid_col: 用户ID列名
        auc_col: AUC列名
    """
    print("\n" + "="*50)
    print("AUC Distribution Analysis")
    print("="*50)
    
    # 获取每个用户的唯一AUC值
    user_aucs = df.groupby(uid_col)[auc_col].first().dropna()
    
    if len(user_aucs) == 0:
        print("No valid AUC values found!")
        return
    
    # 基本统计
    print(f"Total users with valid AUC: {len(user_aucs)}")
    print(f"Mean AUC: {user_aucs.mean():.4f}")
    print(f"Median AUC: {user_aucs.median():.4f}")
    print(f"Std AUC: {user_aucs.std():.4f}")
    print(f"Min AUC: {user_aucs.min():.4f}")
    print(f"Max AUC: {user_aucs.max():.4f}")
    
    # AUC分布区间
    print(f"\nAUC Distribution:")
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins)-1):
        count = ((user_aucs >= bins[i]) & (user_aucs < bins[i+1])).sum()
        percentage = count / len(user_aucs) * 100
        print(f"  {bins[i]:.1f} - {bins[i+1]:.1f}: {count:4d} users ({percentage:5.1f}%)")
    
    # 显示AUC最高和最低的用户
    print(f"\nTop 5 users by AUC:")
    top_users = user_aucs.nlargest(5)
    for uid, auc in top_users.items():
        print(f"  User {uid}: {auc:.4f}")
    
    print(f"\nBottom 5 users by AUC:")
    bottom_users = user_aucs.nsmallest(5)
    for uid, auc in bottom_users.items():
        print(f"  User {uid}: {auc:.4f}")

def main():
    """
    主函数 - 示例用法
    """
    # 示例用法
    csv_path = "/root/autodl-tmp/pykt_self_version/data/assist2009/train_valid_sequences.csv"   # 替换为你的CSV文件路径
    
    # 处理CSV文件并添加AUC列
    df = add_auc_to_csv(
        csv_path=csv_path,
        uid_col='uid',                    # 用户ID列名
        predictions_col='predictions',     # 预测结果列名
        responses_col='responses',         # 真实响应列名
        output_path=None                  # 输出路径，None表示覆盖原文件
    )
    
    if df is not None:
        # 分析AUC分布
        analyze_auc_distribution(df)
        
        # 显示前几行结果
        print("\nSample results:")
        print(df[['uid', 'predictions', 'responses', 'auc']].head(10))

if __name__ == "__main__":
    # 如果直接运行此脚本，请修改下面的路径
    csv_file_path = "/root/autodl-tmp/pykt_self_version/data/assist2009/train_valid_sequences.csv"  # 替换为实际的CSV文件路径
    
    print("Starting AUC calculation...")
    result_df = add_auc_to_csv(csv_path=csv_file_path,
        uid_col='uid',                    # 用户ID列名
        predictions_col='predictions',     # 预测结果列名
        responses_col='responses',         # 真实响应列名
        output_path=None                  # 输出路径，None表示覆盖原文件
        )
    
    if result_df is not None:
        analyze_auc_distribution(result_df)
        print("\nProcessing completed successfully!")
    else:
        print("Processing failed!")
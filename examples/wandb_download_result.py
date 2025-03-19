import os
import wandb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import argparse
import subprocess
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Fetch Wandb runs and generate prediction scripts.')
    parser.add_argument('--new', type=int, choices=[0,1], default=0,
                        help='是否覆盖已存在的文件，0: 不覆盖, 1: 覆盖')
    args = parser.parse_args()
    NEW = args.new

    # 从环境变量获取 Wandb API 密钥和项目名称
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    if not WANDB_API_KEY:
        raise ValueError("请设置环境变量 WANDB_API_KEY，例如: export WANDB_API_KEY='您的API密钥'")

    PROJECT_NAME = os.getenv('PROJECT_NAME')
    if not PROJECT_NAME:
        raise ValueError("请设置环境变量 PROJECT_NAME，例如: export PROJECT_NAME='AKT_CONV_DYNROUTINE_V5_A9'")

    NUMS = os.getenv('NUMS')
    if not NUMS:
        raise ValueError("请设置环境变量 NUMS，例如: export NUMS='0,1,2,3,5'")

    # 从环境变量获取预测的折叠范围
    START_SWEEP = os.getenv('START_SWEEP')
    END_SWEEP = os.getenv('END_SWEEP')
    if START_SWEEP is None or END_SWEEP is None:
        raise ValueError("请设置环境变量 START_SWEEP 和 END_SWEEP，例如: export START_SWEEP='0' export END_SWEEP='5'")

    try:
        START_SWEEP = int(START_SWEEP)
        END_SWEEP = int(END_SWEEP)
    except ValueError:
        raise ValueError("环境变量 START_SWEEP 和 END_SWEEP 必须为整数，例如: export START_SWEEP=0 和 export END_SWEEP=5")

    if START_SWEEP < 0 or END_SWEEP <= START_SWEEP:
        raise ValueError("请确保 0 <= START_SWEEP < END_SWEEP")

    # 解析 NUMS 环境变量为设备列表
    device_list = [device.strip() for device in NUMS.split(',')]
    num_folds = END_SWEEP - START_SWEEP  # 根据 START_SWEEP 和 END_SWEEP 计算fold数量

    if len(device_list) < num_folds:
        raise ValueError(f"设备数量 ({len(device_list)}) 少于fold数量 ({num_folds})。请确保有足够的设备。")

    # 设置保存文件的目录
    save_dir = os.path.join('results', PROJECT_NAME)
    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录设置为: {save_dir}")

    # 登录 Wandb
    wandb.login(key=WANDB_API_KEY)

    # 初始化 Wandb API
    api = wandb.Api()

    # 指定项目路径，例如 "username/project_name"
    username = "byoung9527-Jinan University"
    project_path = f"{username}/{PROJECT_NAME}"
    print("API 配置完成。")

    # 获取项目中的所有运行
    runs = api.runs(project_path)
    print(f"已获取 {len(runs)} 个运行。")

    # 定义一个函数，用于提取单个运行的数据
    def extract_run_data(run):
        config = run.config
        fold = config.get('fold', 'N/A')  # 获取 fold 信息，若不存在则返回 'N/A'
        run_id = run.id
        validauc = run.summary.get('validauc', 'N/A')
        validacc = run.summary.get('validacc', 'N/A')
        timestamp = run.summary.get('_timestamp', 'N/A')
        best_epoch = run.summary.get('best_epoch', 'N/A')
        model_save_path = run.summary.get('model_save_path', 'N/A')

        # 去除 model_save_path 末尾的 'qid_model.ckpt'
        if isinstance(model_save_path, str) and model_save_path.endswith('qid_model.ckpt'):
            # 对于 Python 3.9+，使用 removesuffix
            try:
                model_save_path = model_save_path.removesuffix('qid_model.ckpt')
            except AttributeError:
                # 对于较早版本，使用字符串切片
                model_save_path = model_save_path[:-len('qid_model.ckpt')]

        return {
            'run_id': run_id,
            'fold': fold,
            'validauc': validauc,
            'validacc': validacc,
            'timestamp': timestamp,
            'best_epoch': best_epoch,
            'model_save_path': model_save_path
        }

    # 使用 ThreadPoolExecutor 进行并行处理，并确保进度条显示正确
    data = []
    csv_path = os.path.join(save_dir, "wandb_runs_data.csv")
    if not os.path.exists(csv_path) or NEW == 1:
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 使用 executor.map 结合 tqdm 以正确显示进度条
            results = list(tqdm(executor.map(extract_run_data, runs), total=len(runs), desc="并行获取运行数据"))
            data.extend(results)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(csv_path)
    # 创建 DataFrame

    # 验证 run_id 的唯一性
    if df['run_id'].duplicated().any():
        print("存在重复的 run_id。正在移除重复项...")
        df = df.drop_duplicates(subset=['run_id'])
        print("重复的 run_id 已移除。")
    else:
        print("所有 run_id 均唯一，无重复数据。")

    # 保存 DataFrame 为 CSV 文件（根据 NEW 参数决定是否覆盖）
    if os.path.exists(csv_path) and NEW == 0:
        print(f"文件 '{csv_path}' 已存在，且 NEW=0，不进行覆盖。")
    else:
        df.to_csv(csv_path, index=False)
        print(f"数据已成功保存为 '{csv_path}'。")

    # 按照 fold 分类，获取每个 fold 下 validauc 最高的 model_save_path
    # 首先，确保 validauc 是数值类型
    df['validauc'] = pd.to_numeric(df['validauc'], errors='coerce')

    # 删除 validauc 为 NaN 的行
    df = df.dropna(subset=['validauc'])

    # 获取每个 fold 下 validauc 最高的运行
    idx = df.groupby('fold')['validauc'].idxmax()

    # 提取对应的 model_save_path
    best_models_df = df.loc[idx, ['fold', 'model_save_path']]

    # 将结果转换为字典
    best_models = best_models_df.set_index('fold')['model_save_path'].to_dict()

    print("每个 fold 下 validauc 最高的 model_save_path：")
    for fold, path in best_models.items():
        print(f"Fold {fold}: {path}")

    # 将字典保存为 JSON 文件（根据 NEW 参数决定是否覆盖）
    json_filename = f'best_models_per_fold.json'
    json_path = os.path.join(save_dir, json_filename)
    if os.path.exists(json_path) and NEW == 0:
        print(f"文件 '{json_path}' 已存在，且 NEW=0，不进行覆盖。")
    else:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_models, f, indent=4, ensure_ascii=False)
        print(f"最佳模型路径已保存为 '{json_path}'。")

    # 过滤折叠范围
    filtered_best_models = {fold: path for fold, path in best_models.items() if START_SWEEP <= fold < END_SWEEP}

    # 检查是否有折叠在指定范围内
    if not filtered_best_models:
        print(f"在指定的折叠范围 START_SWEEP={START_SWEEP} 到 END_SWEEP={END_SWEEP} 内，没有找到对应的 model_save_path。")
    else:
        # 生成预测脚本
        script_filename = f'run_predictions_{START_SWEEP}_{END_SWEEP}.sh'
        prediction_script_path = os.path.join(save_dir, script_filename)

        # 检查脚本是否存在，根据 NEW 参数决定是否覆盖
        if os.path.exists(prediction_script_path) and NEW == 0:
            print(f"预测脚本 '{prediction_script_path}' 已存在，且 NEW=0，不进行覆盖。")
        else:
            with open(prediction_script_path, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\n\n")
                for fold in sorted(filtered_best_models.keys()):
                    device = device_list[fold]
                    model_dir = filtered_best_models[fold]
                    # 确保 model_dir 以 '/' 结尾
                    if not model_dir.endswith('/'):
                        model_dir += '/'
                    # 生成预测命令，添加 nohup 和 &，并确保 CUDA_VISIBLE_DEVICES 分配正确
                    command = f'CUDA_VISIBLE_DEVICES={device} nohup python wandb_predict.py --use_wandb 0 --save_dir "{model_dir}" > {save_dir}/fold{fold}_predict.log 2>&1 &\n'
                    f.write(command)
            print(f"预测脚本已保存为 '{prediction_script_path}'。")

            # 给予预测脚本执行权限
            try:
                os.chmod(prediction_script_path, 0o755)
                print(f"已赋予 '{prediction_script_path}' 执行权限。")
                
                # 新增：运行脚本
        
            except Exception as e:
                print(f"操作失败，错误: {e}")
                
        try:
            # 使用 subprocess.run 替代 os.system
            subprocess.run(
                ["bash", prediction_script_path],
                check=True,  # 检查命令执行状态
                shell=False,  # 避免潜在的安全风险
                env=dict(os.environ, MKL_THREADING_LAYER="GNU")  # 添加环境变量解决 MKL 兼容问题
            )
            print(f"✅ 脚本 '{prediction_script_path}' 执行成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 脚本执行失败，错误码：{e.returncode}")
            print(f"详细错误：{e.stderr}")  # 如果需捕获错误输出，可添加 stderr=subprocess.PIPE 参数
        except Exception as e:
            print(f"❌ 发生未知错误：{str(e)}")
        print(f"已启动脚本 '{prediction_script_path}'")

if __name__ == "__main__":
    main()

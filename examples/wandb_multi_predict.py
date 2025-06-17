import os
import argparse
import json
import copy
import torch
import pandas as pd
from pykt.config import ERR_PATH, stu_pk
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
            'record_count': row.get('record_count', stats_df.shape[0]),  # 使用文件行数作为record_count
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
    
    
    if params['use_wandb'] == 1:
        import wandb
        with open("../configs/wandb.json") as fin:
            wandb_config = json.load(fin)
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

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
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size,stu_id=student_id+3)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"学生 {student_id}: 开始预测模型 {model_name}, embtype: {emb_type}, dataset_name: {dataset_name}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = os.path.join(save_dir, f"{model.emb_type}_test_predictions_student_{student_id}.txt")
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
    
    # if ERR_PATH:
    #     save_result_path = os.path.join(data_config["dpath"], f"predict_result_{stu_pk}.csv")
    #     print(f"学生 {student_id}: 保存到目录 {save_result_path}")
    # else:
    #     save_result_path = ""
    
    # if model.model_name == "rkt":
    #     testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path, save_result_path)
    # else:
    #     testauc, testacc = evaluate(model, test_loader, model_name, save_test_path, save_result_path)
    
    # print(f"学生 {student_id}: testauc: {testauc}, testacc: {testacc}")

    # window_testauc, window_testacc = -1, -1
    # save_test_window_path = os.path.join(save_dir, f"{model.emb_type}_test_window_predictions_student_{student_id}.txt")
    # if model.model_name == "rkt":
    #     window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel)
    # else:
    #     window_testauc, window_testacc = evaluate(model, test_window_loader, model_name)
    
    # print(f"学生 {student_id}: testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    # dres = {
    #     "student_id": student_id,
    #     "testauc": testauc, 
    #     "testacc": testacc, 
    #     "window_testauc": window_testauc, 
    #     "window_testacc": window_testacc,
    # }
    dres = {
        "student_id": student_id,
    }
    # q_testaucs, q_testaccs = -1, -1
    # qw_testaucs, qw_testaccs = -1, -1
    # if "test_question_file" in data_config and not test_question_loader is None:
    #     save_test_question_path = os.path.join(save_dir, f"{model.emb_type}_test_question_predictions_student_{student_id}.txt")
    #     q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
    #     for key in q_testaucs:
    #         dres["oriauc" + key] = q_testaucs[key]
    #     for key in q_testaccs:
    #         dres["oriacc" + key] = q_testaccs[key]

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

    if params['use_wandb'] == 1:
        wandb.log(dres)

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


def main(params):
    """主函数，循环评估所有学生"""
    total_students = params.get('total_students', 770)
    start_student = params.get('start_student', 1)
    save_dir = params["save_dir"]
    
    # 存储所有学生的评估结果
    all_results = []
    
    print(f"开始批量评估，共 {total_students} 个学生，从学生 {start_student} 开始")
    
    for student_id in range(start_student, total_students + 1):
        try:
            # 评估单个学生
            result = evaluate_single_student(params, student_id)
            all_results.append(result)
            
            print(f"完成学生 {student_id} 的评估 ({student_id - start_student + 1}/{total_students - start_student + 1})")
            
        except Exception as e:
            print(f"评估学生 {student_id} 时出错: {e}")
            # 创建一个错误记录
            error_result = {
                "student_id": student_id,
                "error": str(e),
                "testauc": -1,
                "testacc": -1,
                "window_testauc": -1,
                "window_testacc": -1
            }
            all_results.append(error_result)
            continue
    
    # 将所有结果转换为DataFrame并保存为CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 生成CSV文件路径
        csv_path = os.path.join(save_dir, "batch_evaluation_results.csv")
        
        try:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n所有评估结果已保存到 CSV 文件: {csv_path}")
            
            # 输出统计信息
            print(f"\n评估完成统计:")
            print(f"总计评估学生数: {len(all_results)}")
            
            # 计算统计列的基本统计信息
            stat_columns = [
                'total_questions', 'avg_questions_per_concept', 'max_questions_per_concept',
                'min_questions_per_concept', 'questions_range', 'overall_accuracy',
                'accuracy_range', 'accuracy_variance', 'max_accuracy', 'min_accuracy'
            ]
            
            for col in stat_columns:
                if col in df.columns:
                    valid_data = df[col][df[col] >= 0]  # 只取有效值
                    if not valid_data.empty:
                        print(f"\n{col}统计:")
                        print(f"  最小值: {valid_data.min():.2f}")
                        print(f"  最大值: {valid_data.max():.2f}")
                        print(f"  平均值: {valid_data.mean():.2f}")
                        print(f"  标准差: {valid_data.std():.2f}")
            
            # 如果有windowauclate_mean和windowacclate_mean字段，计算统计信息
            if 'windowauclate_mean' in df.columns:
                valid_windowauc = df['windowauclate_mean'][df['windowauclate_mean'] != -1]
                if len(valid_windowauc) > 0:
                    print(f"\nwindowauclate_mean统计:")
                    print(f"  最小值: {valid_windowauc.min():.4f}")
                    print(f"  最大值: {valid_windowauc.max():.4f}")
                    print(f"  平均值: {valid_windowauc.mean():.4f}")
                    print(f"  标准差: {valid_windowauc.std():.4f}")
            
            if 'windowacclate_mean' in df.columns:
                valid_windowacc = df['windowacclate_mean'][df['windowacclate_mean'] != -1]
                if len(valid_windowacc) > 0:
                    print(f"\nwindowacclate_mean统计:")
                    print(f"  最小值: {valid_windowacc.min():.4f}")
                    print(f"  最大值: {valid_windowacc.max():.4f}")
                    print(f"  平均值: {valid_windowacc.mean():.4f}")
                    print(f"  标准差: {valid_windowacc.std():.4f}")
                    
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")
    else:
        print("警告：没有收集到任何评估结果")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--total_students", type=int, default=770, help="总学生数量")
    parser.add_argument("--start_student", type=int, default=1, help="开始评估的学生ID")
    # 添加新参数：统计信息文件目录
    # parser.add_argument("--stats_dir", type=str, default="", required=True, help="学生统计信息文件目录")

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
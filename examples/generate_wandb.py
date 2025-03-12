import os
import sys
import json
import argparse

def str2bool(str):
    return True if str.lower() == "true" else False

# 生成启动sweep的脚本
def main(params):
    src_dir = params["src_dir"]
    project_name = params["project_name"]
    dataset_names = params["dataset_names"]
    model_names = params["model_names"]
    folds = params["folds"]
    save_dir_suffix = params["save_dir_suffix"]
    all_dir = params["all_dir"]
    launch_file = params["launch_file"]
    generate_all = params["generate_all"]
    emb_types = params["emb_types"]
    start_sweep = params["start_sweep"]
    end_sweep = params["end_sweep"]
    new = params["new"]  # 获取 new 参数

    all_folds = ", ".join(map(str, range(start_sweep, end_sweep)))
    emb_types = [x.strip() for x in emb_types.split(",")]
    launch_file = os.path.join("logsScripts", f"{project_name}_all_start_{start_sweep}_{end_sweep}.sh")
    print(f"launch_file: {launch_file}")

    if not os.path.exists(all_dir):
        os.makedirs(all_dir)

    with open("../configs/wandb.json") as fin, open(launch_file, "w") as fallsh:
        wandb_config = json.load(fin)
        WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        if WANDB_API_KEY is None:
            WANDB_API_KEY = wandb_config.get("api_key", "")
        print(f"WANDB_API_KEY: {WANDB_API_KEY}")
        pre = f"WANDB_API_KEY={WANDB_API_KEY} wandb sweep "

        for dataset_name in dataset_names.split(","):
            dataset_name = dataset_name.strip()
            files = os.listdir(src_dir)
            for m in model_names.split(","):
                m = m.strip()
                for _type in emb_types:
                    _type = _type.strip()
                    for fold in range(start_sweep, end_sweep):
                        fname = f"{project_name}_{dataset_name}_{m}_{_type.replace('linear', '')}_{fold}.yaml"
                        ftarget = os.path.join(all_dir, fname)
                        fpath = os.path.join(src_dir, f"{m}.yaml")

                        print(f"Source YAML: {fpath}, Target YAML: {ftarget}")

                        # 检查是否需要生成 YAML 文件
                        if new or not os.path.exists(ftarget):
                            with open(fpath, "r") as fin_yaml, open(ftarget, "w") as fout:
                                data = fin_yaml.read()
                                data = data.replace("xes", dataset_name)
                                data = data.replace("tiaocan", f"{project_name}_tiaocan_{dataset_name}{save_dir_suffix}")
                                if "[\"qid" in data and "[\"qid\"]" not in data:
                                    pass
                                else:
                                    data = data.replace("[\"qid\"]", f"['{_type}']")
                                data = data.replace("[0, 1, 2, 3, 4]", str([fold]))
                                data = data.replace('BATCH_SIZE', str(params["batch_size"]))
                                fout.write(f"name: {fname.split('.')[0]}\n")
                                fout.write(data)
                            print(f"Generated YAML: {ftarget}")
                        else:
                            print(f"YAML already exists and `new` is 0. Skipping: {ftarget}")

                        if not generate_all:
                            fallsh.write(f"{pre}{ftarget} -p {project_name}\n")
                            print(f"wrote{pre}{ftarget} -p {project_name}\n")

        if generate_all:
            files = sorted(os.listdir(all_dir))
            for f in files:
                fpath = os.path.join(all_dir, f)
                fallsh.write(f"{pre}{fpath} -p {project_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and manage YAML configurations for sweeps.")

    parser.add_argument("--src_dir", type=str, default="./seedwandb/", help="Source directory containing model YAML files.")
    parser.add_argument("--project_name", type=str, default="kt_toolkits", help="Name of the project.")
    parser.add_argument("--dataset_names", type=str, default="assist2015", help="Comma-separated list of dataset names.")
    parser.add_argument("--model_names", type=str, default="dkt,dkt+,dkt_forget,kqn,atktfix,dkvmn,sakt,saint,akt,gkt", help="Comma-separated list of model names.")
    parser.add_argument("--emb_types", type=str, default="qid", help="Comma-separated list of embedding types.")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated list of fold indices.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--save_dir_suffix", type=str, default="", help="Suffix for the save directory.")
    parser.add_argument("--all_dir", type=str, default="all_wandbs", help="Directory to save all generated YAML files.")
    parser.add_argument("--launch_file", type=str, default="all_start.sh", help="Filename for the launch script.")
    parser.add_argument("--generate_all", type=str2bool, default=False, help="Whether to generate all sweeps.")
    parser.add_argument("--start_sweep", type=int, default=0, help="Start index for sweep.")
    parser.add_argument("--end_sweep", type=int, default=5, help="End index for sweep.")
    parser.add_argument("--new", type=int, choices=[0, 1], default=0, help="Flag to indicate whether to overwrite existing YAML files (1 to overwrite, 0 to skip).")

    args = parser.parse_args()
    params = vars(args)
    print(f"Parameters: {params}")
    main(params)

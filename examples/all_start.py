import os, sys
import json

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)
    if WANDB_API_KEY == None:
        WANDB_API_KEY = wandb_config["api_key"]
# print(WANDB_API_KEY)

logf = sys.argv[1]
outf = open(sys.argv[2], "w")
start = int(sys.argv[3])
end = int(sys.argv[4])
sweep_ids = []
dataset_name = sys.argv[5]
model_name = sys.argv[6]
nums = sys.argv[7].split(",")
# print(f"{dataset_name}_{model_name}")
if len(sys.argv) == 8:
    project_name = "kt_toolkits"
else:
    project_name = sys.argv[8]
# 将超参数存储在字典中
hyperparams = {
    "logf": logf,
    "outf": outf.name,  # 打印文件名而不是文件对象
    "start": start,
    "end": end,
    "dataset_name": dataset_name,
    "model_name": model_name,
    "nums": nums,
    "project_name": project_name
}

# 打印超参数
# print("超参数:")
# for key, value in hyperparams.items():
#     print(f"{key}: {value}")
    
cmdpre = f"WANDB_API_KEY={WANDB_API_KEY} nohup "
endcmdpre =f"WANDB_API_KEY={WANDB_API_KEY} "

# 生成 stop 和 resume 的脚本
def generate_stop_script(sweep_ids, project_name):
    stop_script_name = f"./logsScripts/stopSweep_{project_name}_{start}_{end}.sh"
    with open(stop_script_name, "w") as stop_script:
        stop_script.write("#!/bin/bash\n")
        for sweep_id in sweep_ids:
            stop_script.write(f"{endcmdpre} wandb sweep --stop {sweep_id}\n")
    os.chmod(stop_script_name, 0o755)
    print(f"Stop script {stop_script_name} generated.")

def generate_resume_script(sweep_ids, project_name):
    resume_script_name = f"./logsScripts/resumeSweep_{project_name}_{start}_{end}.sh"
    with open(resume_script_name, "w") as resume_script:
        resume_script.write("#!/bin/bash\n")
        for sweep_id in sweep_ids:
            resume_script.write(f"{endcmdpre} wandb sweep --resume {sweep_id}\n")
    os.chmod(resume_script_name, 0o755)
    print(f"Resume script {resume_script_name} generated.")

def generate_killing_script(sweep_ids, project_name):
    resume_script_name = f"./logsScripts/killSweep_{project_name}_{start}_{end}.sh"
    with open(resume_script_name, "w") as resume_script:
        resume_script.write("#!/bin/bash\n")
        for sweep_id in sweep_ids:
            resume_script.write(f"{endcmdpre} wandb sweep --cancel {sweep_id}\n")
    os.chmod(resume_script_name, 0o755)
    print(f"killing script {resume_script_name} generated.")

idx = start
with open(logf, "r") as fin:
    i = 0
    lines = fin.readlines()
    l = []
    num = 0

    while i < len(lines):
        # print("heer")
        if lines[i].strip().startswith("wandb: Creating sweep from: "):
            fname = lines[i].strip().split(": ")[-1].split("/")[-1]
        else:
            print("error!")
        if lines[i+3].strip().startswith("wandb: Run sweep agent with: "):
            sweepid = lines[i+3].strip().split(": ")[-1]
        else:
            print("error!")
        fname = fname.split(".")[0]
        print(f"fname is {fname}")
        if not fname.startswith(project_name) or fname.find("_" + model_name + "_") == -1:
            i += 4
            continue
        print(f"dataset_name: {dataset_name}, model_name: {model_name}, fname: {fname}")
        if idx >= start and idx < end:
            cmd = "CUDA_VISIBLE_DEVICES=" + str(nums[num]) +" " + cmdpre + sweepid + " &"
            print(cmd)
            outf.write(cmd + "\n")
            num += 1
            sweepid = sweepid.replace("wandb agent ", "")
            sweep_ids.append(sweepid)
        idx += 1
        i += 4
# Generate stop and resume scripts
generate_stop_script(sweep_ids, project_name)
generate_resume_script(sweep_ids, project_name)
generate_killing_script(sweep_ids, project_name)
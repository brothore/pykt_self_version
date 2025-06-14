# Knowledge Tracing 

## Overview
pykt self version
## dataset
This dataset is made up of math exercises, collected from the free online tutoring ASSISTments platform in the school year 2009-2010. The dataset consists of 346,860 interactions, 4,217 students, and 26,688 questions and is widely used and has been the standard benchmark for KT methods over the last decade.

https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

## Installation

### Prerequisites
- Python 3.7+
- Conda (recommended)

### Setup
```bash
conda create --name=tcn_abqr python=3.7.5
conda activate tcn_abqr
pip install -U pykt-toolkit
```

## Quick Start

### 1. Data Preparation
#### Download Dataset
Place your dataset in the `data/{dataset_name}` folder.

#### Preprocessing
```bash
cd examples
python data_preprocess.py --dataset_name=assist2009 --min_seq_len=3 --maxlen=200 --kfold=5
```

### 2. Training TCN_ABQR
```bash
CUDA_VISIBLE_DEVICES=0 nohup python wandb_tcn_abqr_train.py \
    --dataset_name=assist2009 \
    --use_wandb=1 \
    --add_uuid=0 \
    --num_blocks=2 \
    > tcn_abqr_train.log &
```

### 3. Evaluation
```bash
python wandb_predict.py \
    --bz=256 \
    --save_dir="saved_model" \
    --fusion_type="late_fusion" \
    --use_wandb=1
```

## Hyperparameter Tuning

### 1. Wandb Setup
1. Create a Wandb account at [wandb.ai](https://wandb.ai)
2. Add your API key to `configs/wandb.json`

### 2. Generate Sweep Configuration
```bash
export PROJECT_NAME="TCN_ABQR_TUNING"
export DATASET_NAME="assist2009"
export MODEL_NAME="tcn_abqr"
export WANDB_API_KEY=your_api_key_here
export NUMS="0,1,2,3"
export START_SWEEP=0
export END_SWEEP=5

python generate_wandb.py \
    --dataset_names "$DATASET_NAME" \
    --project_name "$PROJECT_NAME" \
    --model_names "$MODEL_NAME" \
    --start_sweep "$START_SWEEP" \
    --end_sweep "$END_SWEEP" \
    --new 1
```

### 3. Launch Sweep
```bash
sh ./logsScripts/${PROJECT_NAME}_all_start_${START_SWEEP}_${END_SWEEP}.sh > ./logsScripts/${PROJECT_NAME}.all 2>&1

sh run_all.sh ./logsScripts/${PROJECT_NAME}.all "$START_SWEEP" "$END_SWEEP" "$DATASET_NAME" "$MODEL_NAME" "$NUMS" "$PROJECT_NAME"

sh ./logsScripts/${PROJECT_NAME}_${START_SWEEP}_${END_SWEEP}.sh >> ./logsScripts/${PROJECT_NAME}_${START_SWEEP}_${END_SWEEP}.log &
```

## Evaluation Pipeline

### 1. Extract Best Models
```python
df = wandb_api.get_best_run(dataset_name="assist2009", model_name="tcn_abqr")
wandb_api.extract_best_models(df, "assist2009", "tcn_abqr",
                            fpath="../examples/seedwandb/predict.yaml",
                            wandb_key=your_api_key_here)
```

### 2. Run Evaluation
```bash
sh start_predict.sh > pred.log 2>&1
WANDB_API_KEY=xxx sh run_all.sh pred.log 0 5 assist2009 tcn_abqr 0 TCN_ABQR_EVAL
sh start_sweep_0_5.sh
```

## Utility Scripts

### Resume Sweeps
```bash
./logsScripts/resumeSweep_${PROJECT_NAME}_${START_SWEEP}_${END_SWEEP}.sh
```

### Stop Sweeps
```bash
./logsScripts/stopSweep_${PROJECT_NAME}_${START_SWEEP}_${END_SWEEP}.sh
```

### Kill Sweeps
```bash
./logsScripts/killSweep_${PROJECT_NAME}_${START_SWEEP}_${END_SWEEP}.sh
```

## Results Processing
```bash
python wandb_download_result.py --new 0
python wandb_fusion2csv.py \
    --project_name "$PROJECT_NAME" \
    --start_sweep "$START_SWEEP" \
    --end_sweep "$END_SWEEP" \
    --interval 10 \
    --timeout 3000
```

## Model Architecture
TCN_ABQR combines:
- Temporal Convolutional Networks for sequence processing
- Attention mechanisms for question representation
- Bidirectional processing of learning sequences
- Question-response interaction modeling

## Citation
If you use this code in your research, please cite:
```
@inproceedings{liupykt2022,
  title={pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models},
  author={Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Tang, Jiliang and Luo, Weiqi},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

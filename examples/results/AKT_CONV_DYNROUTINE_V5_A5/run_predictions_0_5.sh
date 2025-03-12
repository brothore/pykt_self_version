#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python wandb_predict.py --use_wandb 0 --save_dir "models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005/algebra2005_akt_qid_models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005_42_0_0.3_128_256_8_1_0.0001_1_1_[3, 5]_09e20b43-83a3-454b-842a-ccc06490f790/" > results/AKT_CONV_DYNROUTINE_V5_A5/fold0_predict.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --use_wandb 0 --save_dir "models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005/algebra2005_akt_qid_models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005_42_1_0.3_128_256_4_1_0.0001_1_1_[3, 5]_f0c0243a-cda0-4bae-a98a-2fb7c81894b5/" > results/AKT_CONV_DYNROUTINE_V5_A5/fold1_predict.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python wandb_predict.py --use_wandb 0 --save_dir "models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005/algebra2005_akt_qid_models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005_42_2_0.3_128_256_4_1_0.0001_1_1_[3, 5]_a2c29037-cd42-41d9-a7d5-c42b2c6bea70/" > results/AKT_CONV_DYNROUTINE_V5_A5/fold2_predict.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --use_wandb 0 --save_dir "models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005/algebra2005_akt_qid_models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005_42_3_0.3_128_256_8_1_0.0001_1_1_[3, 5]_7caf0f78-0094-4697-a4b5-a83017fd63c4/" > results/AKT_CONV_DYNROUTINE_V5_A5/fold3_predict.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python wandb_predict.py --use_wandb 0 --save_dir "models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005/algebra2005_akt_qid_models/akt_AKT_CONV_DYNROUTINE_V5_A5_tiaocan_algebra2005_42_4_0.3_128_256_8_1_0.0001_1_1_[3, 5]_1c823ed8-b4e5-44ce-9fed-f45f7b56abde/" > results/AKT_CONV_DYNROUTINE_V5_A5/fold4_predict.log 2>&1 &

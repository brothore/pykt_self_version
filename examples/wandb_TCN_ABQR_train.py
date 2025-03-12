import argparse
from wandb_train import main

if __name__ == "__main__":
    
    # num_channels, kernel_size, dropout
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="TCN_ABQR")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    # parser.add_argument("--input_size", type=int, default=1)
    # parser.add_argument("--output_size", type=int, default=1)

    
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--drop_feat1", type=float, default=0.2)
    parser.add_argument("--drop_feat2", type=float, default=0.3)
    parser.add_argument("--drop_edge1", type=float, default=0.3)
    parser.add_argument("--drop_edge2", type=float, default=0.2)
    parser.add_argument("--lamda", type=int, default=5)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--p", type=float, default=0.4)
    
    # ---------------k7-------------------
    parser.add_argument("--kernel_size", type=int, default=3)
    # parser.add_argument("--num_channels", type=str, default="[32, 64]")#0.8291765463549284  forward  21.776633 秒
    # parser.add_argument("--num_channels", type=str, default="[32, 64，128]")#0.8281308185758689  forward  26.716344 秒
    # parser.add_argument("--num_channels", type=str, default="[64，128]")#0.8303997639281949 每个 forward 的总体平均时间: 39.553931 秒
    # parser.add_argument("--num_channels", type=str, default="[64]")#0.82608198768864 每个 forward 的总体平均时间: 0.045070 秒
    # parser.add_argument("--num_channels", type=str, default="[64,64]")# 0.8312437921016369   每个 forward 的总体平均时间: 7.408256 秒
    # parser.add_argument("--num_channels", type=str, default="[64,64,64]")#0.8332966398457486 每个 forward 的总体平均时间: 1.635043 秒
    # parser.add_argument("--num_channels", type=str, default="[64,64,64,64]")#0.8338733726130751 每个 forward 的总体平均时间: 2.494371 秒
    # parser.add_argument("--num_channels", type=str, default="[64,64,64,64,64]")#0.8350638867818436 每个 forward 的总体平均时间: 3.463990 秒
    # parser.add_argument("--num_channels", type=str, default="[64,64,64,64,64,64]")#0.8340539469060881 每个 forward 的总体平均时间: 4.223120 秒
    # parser.add_argument("--num_channels", type=str, default="[16, 32, 64]")#0.829327430870436 每个 forward 的总体平均时间: 27.847911 秒
    # parser.add_argument("--num_channels", type=str, default="[32,32,32,32,32]")#0.8342376915315478 每个 forward 的总体平均时间: 1.196500 秒
    # parser.add_argument("--num_channels", type=str, default="[32,32,32,32]")#0.8335423949454198 每个 forward 的总体平均时间: 1.196500 秒

    # parser.add_argument("--num_channels", type=str, default="[32,32,32,32,32,32,32,32]")# 0.8295731166732233  每个 forward 的总体平均时间: 27.403332 秒
    # parser.add_argument("--num_channels", type=str, default="[32,32,32,32,32,32]")#0.8340587094919394 每个 forward 的总体平均时间: 1.998193 秒
    
    # parser.add_argument("--num_channels", type=str, default="[16, 32, 64, 128]") #0.8212 k7
    # parser.add_argument("--num_channels", type=str, default="[1, 2, 4, 8, 16,32]")#0.8136

    # parser.add_argument("--num_channels", type=str, default="[1, 2, 4, 8, 16,32,64]")#0.8117
    # parser.add_argument("--num_channels", type=str, default="[ 2, 4, 8, 16,32,64,128]")#0.6329
    # parser.add_argument("--num_channels", type=str, default="[ 2, 4, 8, 16,32]")#0.8202 forward 10.393655 秒
    # parser.add_argument("--num_channels", type=str, default="[1, 2, 4, 8, 16, 32,64,128]")
    # parser.add_argument("--num_channels", type=str, default="[1, 2, 4 ]")
    
    # ---------------k3-------------------
    # parser.add_argument("--kernel_size", type=int, default=7)
    
    # parser.add_argument("--num_channels", type=str, default="[16, 32, 64, 128]")#0.8252181004456571 forward 21.781360 秒
    #============================ABQR================================
    
    # parser.add_argument("--num_channels", type=str, default="[1, 2, 4, 8, 16,32]")
    # parser.add_argument("--num_channels", type=str, default="[1, 2, 4, 8, 16,32,32,32]")
    #===========================ConvLSTM=============================
    # parser.add_argument("--num_channels", type=str, default="[64]")#0.7863
    # parser.add_argument("--num_channels", type=str, default="[32,32]")
    # parser.add_argument("--num_channels", type=str, default="[32,32,32]")
    #===========================TCN_ABQR_ATTN=============================
    parser.add_argument("--num_channels", type=str, default="[32,32,32,32,32,32]")




    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--name", type=str, default="TCN_MULTIATTNDEC_ABQR")
    
    args = parser.parse_args()

    params = vars(args)
    main(params)

import argparse
from wandb_train import main

if __name__ == "__main__":
    
    # num_channels, kernel_size, dropout
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="CTNKT")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--emb_size", type=int, default=256)
    # parser.add_argument("--input_size", type=int, default=1)
    # parser.add_argument("--output_size", type=int, default=1)

    #ConvTimeNet 参数
    parser.add_argument('--model', type=str, required=False, default='DePatchConv',
                    help='model name, options: [Autoformer, Informer, Transformer]')
    # 变量，单个还是多个
    parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # ConvTimeNet
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

    parser.add_argument('--dw_ks', type=str, default='37,37,43,43,53,53', help="kernel size of the deep-wise. default:9")
    parser.add_argument('--re_param', type=int, default=1, help='Reparam the DeepWise Conv when train')
    parser.add_argument('--enable_res_param', type=int, default=1, help='Learnable residual')
    parser.add_argument('--re_param_kernel', type=int, default=3)

    # Patch
    parser.add_argument('--patch_ks', type=int, default=32, help="kernel size of the patch window. default:32")
    parser.add_argument('--patch_sd', type=float, default=0.5, \
                        help="stride of the patch window. default: 0.5. if < 1, then sd = patch_sd * patch_ks")


    # Other Parameter
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') 
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')

    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--CTX', type=str, default='0', required=False, help='visuable device ids')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')



    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--name", type=str, default="use_caps")
    
    args = parser.parse_args()

    params = vars(args)
    main(params)

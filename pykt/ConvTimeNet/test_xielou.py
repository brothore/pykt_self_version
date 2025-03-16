from pykt.ConvTimeNet.Patch_layers import *
from pykt.ConvTimeNet.RevIN import RevIN
from pykt.ConvTimeNet.ConvTimeNet_backbone import ConvTimeNet_backbone
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 动态获取设备

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
parser.add_argument('--seq_len', type=int, default=215, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=199, help='prediction sequence length')
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
parser.add_argument('--patch_ks', type=int, default=32, help="kernel size of the patch window. default:32") #patch_len = configs["patch_ks"]  #patch_count
parser.add_argument('--patch_sd', type=float, default=0.5, \
                    help="stride of the patch window. default: 0.5. if < 1, then sd = patch_sd * patch_ks")


# Other Parameter
parser.add_argument('--enc_in', type=int, default=123, help='encoder input size') 
parser.add_argument('--c_out', type=int, default=123, help='output size')
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
parser.add_argument('--batch_size', type=int, default=
                    
                    8, help='batch size of train input data')
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
parser.add_argument("--predict_after_train", type=int, default="0")


args = parser.parse_args()
configs = vars(args) 








# 测试用例
# def test_causal_constraint():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 动态获取设备

#     seq_len = 100
#     patch_size = 10
#     stride = 5
#     model = DepatchSampling(in_feats=1, seq_len=seq_len, patch_size=patch_size, stride=stride).to(device)
    
#     # 构造输入数据（假设当前时间步为t=50）
#     X = torch.randn(1, 1, seq_len).to(device)   # [B=1, C=1, L=100]
    
#     # 前向传播
#     output = model(X)
    
#     # 检查补丁的时间范围是否不超过当前时间
#     _, _, patch_count, patch_size = output.shape
#     for i in range(patch_count):
#         start = i * stride
#         end = start + patch_size
#         assert end <= seq_len, f"补丁{i}越界：{end} > {seq_len}"
        
#     print("测试通过！所有补丁未泄露未来信息。")

# test_causal_constraint()
# 测试左侧填充是否导致补丁覆盖未来数据
    
num_c = 123
model_name = "CTNKT"

# print(f"configs{configs}")
# load parameters 参数初始化
c_in = configs["enc_in"]
# c_in = configs["emb_size"]
context_window = configs["seq_len"]
target_window = configs["pred_len"]
emb_size = configs["emb_size"]
# 模型结构参数
n_layers = configs["e_layers"]
d_model = configs["d_model"]
d_ff = configs["d_ff"]
dropout = configs["dropout"]
head_dropout = configs["head_dropout"]
# 分块相关参数
patch_len = configs["patch_ks"]
# print(f"patch_len{patch_len}")
patch_sd = max(1, int(configs["patch_ks"] * configs["patch_sd"])) if configs["patch_sd"] <= 1 else int(configs["patch_sd"])
stride = patch_sd
print(f"stridesd: {stride}")
padding_patch = configs["padding_patch"]
# 归一化相关参数
revin = configs["revin"]
affine = configs["affine"]
subtract_last = configs["subtract_last"]
# 深度卷积参数
seq_len = configs["seq_len"]
# print(f"seq_len{seq_len}")
dw_ks = configs["dw_ks"]
emb_type = 256
re_param = configs["re_param"]
re_param_kernel = configs["re_param_kernel"]
enable_res_param = configs["enable_res_param"]
interaction_emb = nn.Embedding(num_c * 2,num_c)
padding_stride = stride  # 填充量
norm='batch'
act="gelu" 
head_type = 'flatten'


model = ConvTimeNet_backbone(c_in=c_in, seq_len=seq_len, context_window = context_window,
                            target_window=target_window, patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model, d_ff=d_ff, dw_ks=dw_ks, norm=norm, dropout=dropout, act=act,head_dropout=head_dropout, padding_patch = padding_patch, head_type=head_type, 
                            revin=revin, affine=affine, deformable=True, subtract_last=subtract_last, enable_res_param=enable_res_param, re_param=re_param, re_param_kernel=re_param_kernel).to(device)
    
model.eval()  # 确保在测试模式
# --------------------------------------------
# 测试1：验证RevIN归一化是否使用未来信息
# --------------------------------------------
def test_revin():
    # 构造测试数据：前半部分全0，后半部分全1
    test_input = torch.cat([
        torch.zeros(1, c_in, context_window//2),
        torch.ones(1, c_in, context_window//2)
    ], dim=-1).to(device)
    
    # 前向过程
    with torch.no_grad():
        output = model(test_input)
    # print(f"output: {output}")
    # 检查反归一化后的输出范围
    if output.max() <= 1.0 and output.min() >= 0.0:
        print("RevIN测试通过：输出范围正常")
    else:
        print("RevIN警告：检测到非常规输出范围")

# --------------------------------------------
# 测试2：验证分块采样不访问未来信息
# --------------------------------------------
def test_sampling():
    patch_num = int((context_window - patch_len)/stride + 1)
    # 构造具有明显时间特征的测试数据
    time_series = torch.arange(context_window).float().view(1, 1, context_window).to(device)
    time_series = time_series.repeat(1, c_in, 1).to(device) 
    if padding_patch == 'end': # can be modified to general case
        # self.padding_patch_layer = nn.ReplicationPad1d((0, stride*2)) 
        padding_patch_layer = nn.ReplicationPad1d(( stride,0)) 
        time_series = padding_patch_layer(time_series)
        patch_num += 1
    
    # 获取采样位置
    with torch.no_grad():
        sampling_locations = model.deformable_sampling.get_sampling_location(time_series)[0]
    
    # 验证采样位置不超过当前窗口
    max_position = (sampling_locations[..., 0] * (context_window-1)).max().item()
    if max_position < context_window-1:
        print("采样测试通过：未检测到未来位置采样")
    else:
        print(f"采样警告：检测到未来位置采样（最大位置 {max_position:.1f}）")

# --------------------------------------------
# 测试3：验证因果卷积的有效性
# --------------------------------------------
def test_causal_conv():
    # 构造脉冲输入测试数据
    test_input = torch.zeros(1, c_in, context_window).to(device)
    test_input[0, 0, context_window//2] = 1.0  # 中间位置放置脉冲
    
    # 检查所有卷积层
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv1d):
            # 验证卷积的padding方式
            
            if layer.padding[0] != (layer.kernel_size[0]-1):
                print(f"layer.padding[0]: {layer.padding[0]}")
                print(f"卷积层 '{name}' 警告：检测到非因果填充方式")
    
    # 前向传播验证
    with torch.no_grad():
        output = model(test_input)
    
    # 检查响应位置
    response_pos = output.argmax()
    if response_pos >= context_window//2:
        print("因果卷积测试通过：响应位置正确")
    else:
        print(f"因果卷积警告：检测到提前响应（位置 {response_pos.item()}）")

# --------------------------------------------
# 测试4：完整前向传播泄露测试
# --------------------------------------------
def test_full_forward():
    # 构造测试数据：前N-1个时间步为随机噪声，最后1步为特殊值
    normal_data = torch.randn(1, c_in, context_window-1)
    test_input = torch.cat([
        normal_data,
        torch.full((1, c_in, 1), 10.0)
    ], dim=-1).to(device)
    
    # 前向过程
    with torch.no_grad():
        output = model(test_input)
    
    # 分析输出特征
    output_window = output[0, 0, :].cpu().numpy()
    anomaly_ratio = (output_window > 8.0).mean()
    
    if anomaly_ratio < 0.2:
        print("完整前向测试通过：未检测到异常传播")
    else:
        print(f"泄露警告：检测到异常传播比例 {anomaly_ratio*100:.1f}%")

# 执行所有测试
if __name__ == "__main__":
    print("开始泄露测试...")
    test_revin()
    test_sampling()
    test_causal_conv()
    test_full_forward()
    print("测试完成")
# def test_padding_operation():
    
    

#     # 构造输入数据（假设当前时间点为t=context_window）
#     X = torch.randn(1, c_in, context_window) .to(device)
    
#     # 前向传播
#     output = model(X)
    
#     # 检查填充后的长度是否符合预期
#     padded_length = context_window + padding_stride
#     print(f"model.padding_patch_layer.padding: {model.padding_patch_layer.padding}")
#     assert model.padding_patch_layer.padding == (padding_stride, 0), "填充方向错误！应为左侧填充"
    
    

#     # 验证补丁生成是否仅使用历史数据
#     unfold = X.unfold(dimension=-1, size=patch_len, step=stride)
#     assert unfold.shape[-1] == patch_len, "补丁长度错误"
#     print("测试通过：填充操作未泄露未来信息")

# test_padding_operation()

# def test_deformable_patch_boundaries():


#     # 创建可变形补丁模块
#     depatch = DepatchSampling(c_in, seq_len, patch_len, stride).to(device)

#     # 构造输入数据（当前时间点为t=seq_len）
#     X = torch.randn(2, c_in, seq_len).to(device)  # [B=2, C=1, L=100]

#     # 获取采样位置
#     sampling_locations, bound = depatch.get_sampling_location(X)

#     # 将归一化坐标转换为实际索引
#     bound_indices = bound * (seq_len - 1)
#     right_bound = bound_indices[..., 1]  # 右边界索引

#     # 检查所有右边界是否不超过当前时间点
#     violation = (right_bound > seq_len - 1).any()
#     assert not violation, "可变形补丁右边界泄露未来信息！"
#     print("测试通过：可变形补丁边界未越界")

# test_deformable_patch_boundaries()

# def test_revin_normalization():
#     # 验证RevIN是否独立处理每个序列

#     model = RevIN(c_in, affine=affine, subtract_last=subtract_last)

#     # 构造两个不同均值的序列
#     X1 = torch.randn(1, seq_len, c_in) * 10 + 5  # 均值~5
#     X2 = torch.randn(1, seq_len, c_in) * 10 - 5  # 均值~-5

#     # 归一化与反归一化
#     norm_X1 = model(X1, 'norm')
#     denorm_X1 = model(norm_X1, 'denorm')

#     norm_X2 = model(X2, 'norm')
#     denorm_X2 = model(norm_X2, 'denorm')

#     # 检查是否恢复原始数据
#     assert torch.allclose(X1, denorm_X1, atol=1e-4), "RevIN反归一化失败"
#     assert torch.allclose(X2, denorm_X2, atol=1e-4), "RevIN反归一化失败"

#     # 检查是否使用全局统计量（应失败）
#     global_mean = X1.mean(dim=1, keepdim=True)
#     local_mean = model.mean
#     assert not torch.allclose(global_mean, local_mean), "RevIN使用了全局统计量！"
#     print("测试通过：RevIN无信息泄露")

# test_revin_normalization()

# def test_unfold_operation():


#     # 构造输入数据（当前时间点为t=100）
#     X = torch.randn(1, c_in, context_window)  # [B=1, C=1, L=100]

#     # 生成补丁
#     # patches = X.unfold(dimension=-1, size=patch_len, step=stride)
#     X = X.unsqueeze(1).permute(0, 1, 3, 2)
#     patches = F.unfold(X, kernel_size=(patch_len, c_in), stride=stride)
#     # 检查最后一个补丁的右边界
#     last_patch_end = (patches.shape[-1] - 1) * stride + patch_len
#     print(f"last_patch_end: {last_patch_end}")
#     print(f"context_window: {context_window}")
#     assert last_patch_end <= context_window, "补丁展开泄露未来信息！"
#     print("测试通过：普通补丁展开未越界")

# test_unfold_operation()

# ########################################
# # 检测工具函数
# ########################################
# def check_temporal_leakage(model, seq_len=5, verbose=True):
#     """测试时间窗口是否包含未来信息"""
#     # 生成测试数据
#     X = torch.arange(seq_len).float().view(1, 1, seq_len).repeat(1, model.channel, 1)
    
#     # 计算unfold参数
#     K = model.patch_size
#     S = model.stride
#     num_patches = (seq_len - K) // S + 1
    
#     # 分析每个patch的时间范围
#     leakages = []
#     for i in range(num_patches):
#         start = i * S
#         end = start + K - 1
#         if end >= seq_len - S:  # 最后一个patch的结束位置超过当前允许范围
#             leakages.append((i, start, end))
    
#     if verbose:
#         print(f"Time window analysis (seq_len={seq_len}, patch_size={K}, stride={S}):")
#         for i, s, e in leakages:
#             print(f"  Patch {i}: [{s}-{e}] contains future timesteps")
    
#     return len(leakages) > 0

# def causality_test(model, inject_position=-1):
#     """通过注入特殊值检测因果关系"""
#     # 生成测试数据
#     seq_len = 5
#     X = torch.randn(1, model.channel, seq_len)
    
#     # 注入特殊值（如NaN）
#     original_value = X[0, 0, inject_position].clone()
#     X[0, 0, inject_position] = float('nan')
    
#     try:
#         with torch.no_grad():
#             output = model(X)
#             affected = torch.isnan(output).any()
#             return not affected  # 如果输出不受影响，说明没有使用未来信息
#     except Exception as e:
#         print(f"Error during inference: {str(e)}")
#         return False

# def autoregressive_simulation(model, steps=3):
#     """模拟自回归推理过程"""
#     history = []
#     current_seq = torch.randn(1, model.channel, seq_len)  # 初始只有一个时间步
    
#     for t in range(steps):
#         try:
#             # 扩展序列长度

            
#             # 尝试生成offset
#             with torch.no_grad():
#                 _ = model(current_seq)
                
#         except RuntimeError as e:
#             if "unfold" in str(e) and "out of range" in str(e):
#                 print(f"At step {t}: Failed to process due to future data requirement")
#                 return True  # 检测到需要未来数据
#             raise
#     return False

# ########################################
# # 综合测试
# ########################################
# def run_full_diagnosis(in_feats=1, patch_size=3, stride=1):
#     print("="*60)
#     print(f"Running diagnostics for OffsetPredictor (patch_size={patch_size}, stride={stride})")
#     print("="*60)
    
#     model = OffsetPredictor(in_feats, patch_size, stride)
    
#     # 测试1: 时间窗口分析
#     has_leakage = check_temporal_leakage(model, seq_len=5)
#     print(f"\nTest 1 - Temporal Window Analysis: {'Leakage detected' if has_leakage else 'No leakage'}")
    
#     # 测试2: 因果性测试（注入特殊值）
#     causal_safe = causality_test(model)
#     print(f"Test 2 - Causality Test: {'Safe' if causal_safe else 'Leakage detected'}")
    
#     # 测试3: 自回归模拟
#     requires_future = autoregressive_simulation(model)
#     print(f"Test 3 - Autoregressive Simulation: {'Valid' if not requires_future else 'Invalid (needs future data)'}")
    
#     # 综合结论
#     if any([has_leakage, not causal_safe, requires_future]):
#         print("\n🚨 Conclusion: MODERN LEAKAGE DETECTED!")
#     else:
#         print("\n✅ Conclusion: No leakage detected")

# ########################################
# # 运行测试
# ########################################
# if __name__ == "__main__":
#     # 测试不同配置
#     run_full_diagnosis(patch_size=patch_len, stride=stride)  # 明显会泄露的配置
#     run_full_diagnosis(patch_size=patch_len, stride=stride)  # 安全配置（stride等于patch_size）

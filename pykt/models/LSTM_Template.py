import os

import numpy as np
import torch
from torch import nn
# from TCN.tcn import TemporalConvNet
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import ast
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import ast
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
import copy
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#---------------------------v1 0.835 k7 [64,64,64,64,64] ---------------------------------------

class LSTM_Template(nn.Module):
    def __init__(self, num_c, emb_size, num_channels, kernel_size, dropout,emb_type='qid', emb_path="", pretrain_dim=768,name = "random"):
        super(LSTM_Template, self).__init__()
        
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.model_name = "LSTM_Template"
        self.max_seq_length = 200
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.num_channels = ast.literal_eval(num_channels)
        # 初始化 TCN 网络和线性输出层
        self.tcn = TemporalConvNet(self.emb_size, self.num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(self.num_channels[-1],self.num_c)# 最后一层将 TCN 的输出映射到目标大小
        self.init_weights()# 初始化权重
        self.capsule = capsule(d_model=self.num_channels[-1],n_question=num_c,kernel_sizes=[3,5],distance_decay=0.2)    
        self.position_encoding = nn.Embedding(self.max_seq_length, emb_size)

    def init_weights(self):
        # 初始化线性层权重，正态分布
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, y):
        # print(f"input size:{x.shape}")

        
        if self.emb_type == "qid":
            x = x + self.num_c * y
            xemb = self.interaction_emb(x)
        #====================position encoding==============    
        # positions = torch.arange(xemb.size(1), device=xemb.device).unsqueeze(0).expand(xemb.size(0), -1)
        # pos_emb = self.position_encoding(positions)
        # xemb = xemb + pos_emb 
        
        #===================================================
        
        xemb = xemb.permute(0, 2, 1)
        # print(f"embinput size:{xemb.shape}")
        
        
        # 前向传播：输入数据经过 TCN 网络，最后只取最后一个时间步的输出
        y1 = self.tcn(xemb)
        y1 = y1.permute(0, 2, 1)
        # print(f"tcn_out.shape is {y1.shape}")
        
        
        #===============应用胶囊=====================
        y1 = self.capsule(y1)
        # print(f"cap_out.shape is {y1.shape}")
        
        #===========================================
        
        y2 = self.linear(y1[:, :, :])
        # print(f"y2.shape is {y2.shape}")
        y2 = torch.sigmoid(y2)
        return y2
    
    
class Chomp1d(nn.Module):#用于修正填充后的结果。时间卷积需要确保输出中每个时间步的信息仅依赖于当前或之前的时间步。为了避免未来信息泄露，卷积后的填充部分需要修剪。
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 修剪填充后的多余部分，保证因果卷积的时序一致性
        return x[:, :, :-self.chomp_size].contiguous()

    
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         # 第一层卷积，带有权重归一化
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp1 = Chomp1d(padding)# 修剪填充部分
#         self.relu1 = nn.ReLU()# 激活函数
#         self.dropout1 = nn.Dropout(dropout)# 随机丢弃部分神经元
#         # 第二层卷积，类似于第一层
#         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#         # 构建序列化的网络结构（包括两层卷积）
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         # 下采样模块，用于调整输入和输出维度一致（跳跃连接）
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         # 初始化卷积层权重
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):

#         # print(f"input size3:{x.shape}")
#         out = self.net(x)
        
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 使用常量填充层，填充值为 -1，仅在左侧填充
        self.pad1 = nn.ConstantPad1d((padding, 0), value=-1)
        # 第一层卷积，移除原有的 padding 参数
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()  # 激活函数
        self.dropout1 = nn.Dropout(dropout)  # 随机丢弃部分神经元

        # 第二层卷积，类似于第一层
        self.pad2 = nn.ConstantPad1d((padding, 0), value=-1)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 构建序列化的网络结构（包括两层卷积和填充）
        self.net = nn.Sequential(
            self.pad1, self.conv1, self.relu1, self.dropout1,
            self.pad2, self.conv2, self.relu2, self.dropout2
        )

        # 下采样模块，用于调整输入和输出维度一致（跳跃连接）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # 初始化卷积层权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # 按层构建 TemporalBlock，每层的膨胀系数为 2^i
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)# 将所有层串联成网络

    def forward(self, x):
        # print(f"input size2:{x.shape}")
        
        return self.network(x)# 前向传播
    
    
class CausalConvSmoothing(nn.Module):
    def __init__(self, kernel_size, d_model):
        super(CausalConvSmoothing, self).__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, groups=d_model, bias=False)
        with torch.no_grad():
            self.conv.weight.data.fill_(1.0 / kernel_size)

    def forward(self, x):
        # x: [N, L, C]
        x = x.permute(0, 2, 1)
        padding_left = self.kernel_size - 1
        x = F.pad(x, (padding_left, 0))
        x_smoothed = self.conv(x)
        x_smoothed = x_smoothed.permute(0, 2, 1)
        return x_smoothed
    
    
    
class MultiScaleConvDecomp(nn.Module):
    def __init__(self, kernel_sizes, d_model):
        super(MultiScaleConvDecomp, self).__init__()
        self.convs = nn.ModuleList([CausalConvSmoothing(k, d_model) for k in kernel_sizes])

    def forward(self, x):
        seasonal_list = []
        trend_list = []
        for conv_layer in self.convs:
            moving_mean = conv_layer(x)    # [N, L, C]
            trend_list.append(moving_mean) 
            seasonal_list.append(x - moving_mean)
        return seasonal_list, trend_list

    
class capsule(nn.Module):
    def __init__(self,d_model,n_question=0,kernel_sizes=[3,5], distance_decay=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_question = n_question
        self.kernel_sizes = kernel_sizes
        
        self.distance_decay = distance_decay
        self.decomposition = MultiScaleConvDecomp(kernel_sizes=self.kernel_sizes, d_model=d_model)
        self.Linear_Seasonal = nn.Linear(d_model, n_question)
        self.Linear_Trend = nn.Linear(d_model, n_question)
        self.Linear_Seasonal.weight = nn.Parameter((1/d_model)*torch.ones([n_question,d_model]))
        self.Linear_Trend.weight = nn.Parameter((1/d_model)*torch.ones([n_question,d_model]))
        self.alpha = nn.Parameter(torch.tensor(0.7))
        # 融合层，将 seasonal_fused, trend_fused 融合后映射回d_model维
        self.fusion = nn.Linear(2*d_model, d_model)
        
    def squash(self,v, dim=-1):
        norm = torch.norm(v, dim=dim, keepdim=True)
        scale = (norm**2) / (1 + norm**2)
        direction = v / (norm + 1e-9)
        return scale * direction

    def dynamic_routing(self,input_vecs, num_iterations=2):
        N, L, M, C = input_vecs.shape
        logits = torch.zeros(N, L, M, device=input_vecs.device)
        for i in range(num_iterations):
            c = F.softmax(logits, dim=-1)  # [N, L, M]
            c_exp = c.unsqueeze(-1)
            s = torch.sum(c_exp * input_vecs, dim=2)  # [N, L, C]
            v = self.squash(s, dim=-1)
            if i < num_iterations - 1:
                sim = torch.sum(input_vecs * v.unsqueeze(2), dim=-1)  # [N, L, M]
                logits = logits + sim
        return v        
    def time_attention(self, seq_mean):
        N, L, C = seq_mean.shape
        sim = torch.matmul(seq_mean, seq_mean.transpose(1, 2))  # [N,L,L]

        mask = torch.tril(torch.ones(L, L, device=seq_mean.device), diagonal=0).bool()
        sim = sim.masked_fill(~mask, float('-inf'))

        dist = torch.arange(L, device=seq_mean.device).unsqueeze(1) - torch.arange(L, device=seq_mean.device).unsqueeze(0)
        dist = dist.float()
        decay = torch.exp(-self.distance_decay * dist) * mask
        decay = decay.unsqueeze(0)

        weighted_sim = sim + torch.log(decay)
        attn_weights = F.softmax(weighted_sim, dim=-1) # [N, L, L]
        out = torch.matmul(attn_weights, seq_mean) # [N, L, C]
        return out
    
    # def forward(self,qa_embed_data):
    #     seasonal_list, trend_list = self.decomposition(qa_embed_data)
    #     seasonal_stack = torch.stack(seasonal_list, dim=2)  # [N,L,M,C]
    #     trend_stack = torch.stack(trend_list, dim=2)        # [N,L,M,C]

    #     seasonal_fused = dynamic_routing(seasonal_stack, num_iterations=3) # [N,L,C]
    #     trend_fused = dynamic_routing(trend_stack, num_iterations=3)       # [N,L,C]

    #     # 可选的时间加权
    #     seasonal_fused = self.time_attention(seasonal_fused)
    #     trend_fused = self.time_attention(trend_fused)
    #     fused = self.fusion(torch.cat([seasonal_fused, trend_fused], dim=-1)) # [N,L,d_model]
    #     return(fused)


    def forward(self, *args):
        
        if len(args) == 1:
            # print("args 1")
            qa_embed_data = args[0]
            seasonal_list, trend_list = self.decomposition(qa_embed_data)
            seasonal_stack = torch.stack(seasonal_list, dim=2)  # [N,L,M,C]
            trend_stack = torch.stack(trend_list, dim=2)        # [N,L,M,C]

            seasonal_fused = self.dynamic_routing(seasonal_stack, num_iterations=3) # [N,L,C]
            trend_fused = self.dynamic_routing(trend_stack, num_iterations=3)       # [N,L,C]

            # 可选的时间加权
            seasonal_fused = self.time_attention(seasonal_fused)
            trend_fused = self.time_attention(trend_fused)
            fused = self.fusion(torch.cat([seasonal_fused, trend_fused], dim=-1)) # [N,L,d_model]
            return fused       
            
        elif len(args) == 2:
            # print("args 2")
            
            data1, data2 = args
            u = torch.stack([data1, data2], dim=2)
            fused = self.dynamic_routing(u)
            return fused

            
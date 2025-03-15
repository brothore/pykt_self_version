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
# from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
# from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
#         MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
# from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
import copy
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from pykt.ConvTimeNet.ConvTimeNet_backbone import ConvTimeNet_backbone
#---------------------------v1 0.835 k7 [64,64,64,64,64] ---------------------------------------


class CTNKT(nn.Module):
    def __init__(self,num_c, configs,emb_type, norm:str='batch', act:str="gelu", head_type = 'flatten'):
        
        super().__init__()
        self.num_c = num_c
        self.model_name = "CTNKT"
        
        # print(f"configs{configs}")
        # load parameters 参数初始化
        c_in = configs["enc_in"]
        # c_in = configs["emb_size"]
        context_window = configs["seq_len"]
        target_window = configs["pred_len"]
        self.emb_size = configs["emb_size"]
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
        self.emb_type = emb_type
        re_param = configs["re_param"]
        re_param_kernel = configs["re_param_kernel"]
        enable_res_param = configs["enable_res_param"]
        self.interaction_emb = nn.Embedding(self.num_c * 2,self.num_c)
        # self.interaction_emb = nn.Embedding(self.num_c * 2,self.emb_size)

        
        # model 模型主干
        self.model = ConvTimeNet_backbone(c_in=c_in, seq_len=seq_len, context_window = context_window,
                                target_window=target_window, patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model, d_ff=d_ff, dw_ks=dw_ks, norm=norm, dropout=dropout, act=act,head_dropout=head_dropout, padding_patch = padding_patch, head_type=head_type, 
                                revin=revin, affine=affine, deformable=True, subtract_last=subtract_last, enable_res_param=enable_res_param, re_param=re_param, re_param_kernel=re_param_kernel)
        
    def forward(self, q,r):
        if self.emb_type == "qid":
            x = q + self.num_c * r
            # print(f"x.shape: {x.shape}")
            x = self.interaction_emb(x)
        # print(f"xemb shape: {x.shape}")
        # x = x.unsqueeze(-1)
        # print(f"after xemb shape: {x.shape}")
        
        x = x.permute(0,2,1)    # 维度转换 [B, L, C] -> [B, C, L]
        # print(f"x.shape: {x.shape}")
        x = self.model(x)
        x = x.permute(0,2,1)  # 维度还原  [B, C, L] -> [B, L, C]
        # print(f"x: {x.shape}")
        x = x.squeeze(dim=-1)  
        x = torch.sigmoid(x)
        # print(f"x.shape{x.shape}")
        # print(f"x{x}")
        return x
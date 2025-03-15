import os

import numpy as np
import torch
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class LiquidCell(nn.Module):
  
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_x = nn.Linear(input_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.activation = torch.tanh

    def forward(self, h, x):
        dh = (-h + self.activation(self.W_h(h) + self.W_x(x))) / self.tau
        h_new = h + dh  
        return h_new

class DKT(nn.Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size  
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = nn.Embedding(self.num_c * 2, self.emb_size)

        # 用液体神经元单元替换 LSTM 层
        self.liquid_cell = LiquidCell(self.hidden_size, self.emb_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, self.num_c)
        
    def forward(self, q, r):
        if self.emb_type == "qid":
            # 交互编码：例如 q + num_c * r
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)  # [batch, seq_len, emb_size]
        batch_size, seq_len, _ = xemb.shape

        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_size, device=xemb.device)
        outputs = []
        for t in range(seq_len):
            h = self.liquid_cell(h, xemb[:, t, :])
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # [batch, seq_len, hidden_size]
        
        h_drop = self.dropout_layer(outputs)
        y = self.out_layer(h_drop)  # [batch, seq_len, num_c]
        y = torch.sigmoid(y)
        return y
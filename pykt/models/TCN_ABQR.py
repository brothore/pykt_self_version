import os
import copy
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
import pykt.models.glo


class GCNConv(nn.Module):
    """
    作用：
    实现经典的 GCN 卷积操作。
    聚合节点邻居的信息，并结合权重和偏置更新节点特征。
    关键点：
    torch.sparse.mm: 高效的稀疏矩阵乘法，用于处理稀疏图。
    """
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):#AHW + b A：邻接矩阵  H 第 l 层的节点特征矩阵


        x = self.dropout(x)
        x = torch.matmul(x, self.w)             # 对应 H * W（特征线性变换）
        # #print(f"adj shape{adj.T.shape}")     # 对应 A * (H * W)（邻接矩阵聚合）
        x = torch.sparse.mm(adj.float(), x)     # 对应 + b（添加偏置）
        # #print(f"x shape{x.shape}")           # 输出 H^{(l+1)}
        # #print(f"b shape{self.b.shape}")
        x = x + self.b

        return x

class MLP_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def normalize_graph(A):
    eps = 1e-8
    A = A.to_dense()
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A.to_sparse()

def drop_adj(edge_index, drop_prob):
    begin_size = edge_index.size()
    use_edge = edge_index._indices()
    drop_mask = torch.empty(
        (use_edge.size(1),),
        dtype=torch.float32,
        device=use_edge.device).uniform_(0, 1) >= drop_prob
    y = use_edge.clone()
    res = y[:, drop_mask]
    values = torch.ones(res.shape[1]).to(device)
    size = begin_size
    graph = torch.sparse.FloatTensor(res, values, size)
    graph = normalize_graph(graph)
    return graph

def augment_graph(x, feat_drop, edge, edge_drop):
    drop_x = drop_feature(x, feat_drop)
    drop_edge = drop_adj(edge, edge_drop)
    return drop_x, drop_edge

def drop_adj_gat(edge_index, drop_prob):
    begin_size = edge_index.size()
    use_edge = edge_index._indices()
    drop_mask = torch.empty(
        (use_edge.size(1),),
        dtype=torch.float32,
        device=use_edge.device).uniform_(0, 1) >= drop_prob
    y = use_edge.clone()
    res = y[:, drop_mask]
    values = torch.ones(res.shape[1]).to(device)
    size = begin_size
    graph = torch.sparse.FloatTensor(res, values, size)
    return graph

def augment_graph_gat(x, feat_drop, edge, edge_drop):
    drop_x = drop_feature(x, feat_drop)
    drop_edge = drop_adj_gat(edge, edge_drop)
    return drop_x, drop_edge

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

class BGRL(nn.Module):
    """
    作用：
    构建在线编码器、目标编码器，并通过对比学习更新在线表征。
    关键点：
    在线编码器和目标编码器初始相同，但目标编码器通过 EMA 更新。
    预测器用于让在线表征预测目标表征。
    """
    def __init__(self, d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2):
        super(BGRL, self).__init__()
        #print(f"d{d}p{p}")
        self.drop_feat1, self.drop_feat2, self.drop_edge1, self.drop_edge2 = drop_feat1, drop_feat2, drop_edge1, drop_edge2

        # self.online_encoder = GraphAttentionLayer(d, d, 0.2, p)

        self.online_encoder = GCNConv(d, d, p)

        self.decoder = GCNConv(d, d, p)

        self.predictor = MLP_Predictor(d, d, d)

        self.target_encoder = copy.deepcopy(self.online_encoder)

        # self.GMAE = GMAE(d, p, mask_rate, alpha)

        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

        # self.target_encoder.reset_parameters()

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.enc_mask_token = nn.Parameter(torch.zeros(1, d))
        self.encoder_to_decoder = nn.Linear(d, d, bias=False)

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        num_noise_nodes = int(0.1 * num_mask_nodes)
        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        token_nodes = mask_nodes[perm_mask[: int(0.9 * num_mask_nodes)]]
        noise_nodes = mask_nodes[perm_mask[-int(0.1 * num_mask_nodes):]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[noise_nodes] = x[noise_to_be_chosen]

        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes, keep_nodes

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def compute_batch_loss(self,x, y):

        z1 = self.project(z1)
        z2 = self.project(z2)

        c1 = F.normalize(z1, dim=-1, p=2)
        c2 = F.normalize(z2, dim=-1, p=2)

        batch_size = 15000
        device = x.device
        num_nodes = x.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            batch_pos_matrix = glo.get_value('regist_pos_matrix')[mask]  # batch n
            item_loss = torch.matmul(c1[mask], c2.T)  # batch n
            item_loss = 2 - 2 * item_loss  # batch n
            need_loss = item_loss * batch_pos_matrix  # batch n

            need_sum = need_loss.sum(dim=-1, keepdims=True)  # batch 1
            # need_mean = need_sum / (item_pos_sum[mask] + 1e-8)  # batch n
            need_mean = need_sum

            losses.append(need_mean)

        return -torch.cat(losses).mean()

    def getGraphMAE_loss(self, x, adj):
        mask_rate = 0.3
        use_x, mask_nodes, keep_nodes = self.encoding_mask_noise(x, mask_rate)

        enc_rep = self.online_encoder(use_x, adj)

        rep = self.encoder_to_decoder(enc_rep)

        rep[mask_nodes] = 0

        recon = self.decoder(rep, adj)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = sce_loss(x_rec, x_init, 3)
        return enc_rep, loss

    def forward(self, x, adj, perb=None):

        #         if perb is not None:
        #             x1, adj1 = x, adj
        #             x2, adj2 = x1 + perb, copy.deepcopy(adj1)
        #             embed = x2 + self.online_encoder(x2, regist_pos_matrix)
        #         else:
        #             x1, adj1 = augment_graph_gat(x, self.drop_feat1, adj, self.drop_edge1)
        #             x2, adj2 = augment_graph_gat(x, self.drop_feat2, adj, self.drop_edge2)
        #             embed = x + self.online_encoder(x, regist_pos_matrix)
        if perb is None:
            #print(f"x{x.shape},adj{adj.shape}")
            return x + self.online_encoder(x, adj), 0

        x1, adj1 = x, copy.deepcopy(adj)
        #print(f"x{x.shape},perb{perb.shape}")
        x2, adj2 = x + perb, copy.deepcopy(adj)

        embed = x2 + self.online_encoder(x2, adj2)

        online_x = self.online_encoder(x1, adj1)
        online_y = self.online_encoder(x2, adj2)

        with torch.no_grad():
            #             detach_gat = self.target_encoder(x, adj).detach()
            #             target_y = self.project(detach_gat)
            #             target_x = self.project(detach_gat + perb)
            target_y = self.target_encoder(x1, adj1).detach()
            target_x = self.target_encoder(x2, adj2).detach()

        online_x = self.predictor(online_x)
        online_y = self.predictor(online_y)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean()

        return embed, loss




#---------------------------v1 0.835 k7 [64,64,64,64,64] ---------------------------------------

class TCN_ABQR(nn.Module):
    def __init__(self, num_c, emb_size, num_channels, kernel_size, dropout, drop_feat1, drop_feat2, drop_edge1, drop_edge2, num_q, lamda,
                 
                  d, p,num_heads=4,attn_dropout_rate=0.3, head=1,emb_type='qid',name = "random"):
        """
        skill_max: 知识点（技能）总数，用于分类层的输出维度。
        drop_feat1, drop_feat2: 特征丢弃的概率，增强图特征。
        drop_edge1, drop_edge2: 边丢弃的概率，用于图的结构增强。
        positive_matrix: 正样本关系矩阵，定义对比学习中的正样本对。
        pro_max: 题目总数，用于初始化题目表征。
        lamda: 对抗扰动的步长，用于优化对抗学习。
        contrast_batch: 对比学习的批量大小。
        tau: 对比学习的温度参数。
        d: 特征维度，表示每个题目或答案嵌入的维度。
        p: Dropout 概率，用于正则化。
        head: 多头注意力的头数。
        graph_aug: 图增强策略，默认为 knn。
        gnn_mode: 图神经网络模式，默认为 gcn。
        """
        
        
        super(TCN_ABQR, self).__init__()
        self.num_heads = num_heads
        self.attn_dropout_rate = attn_dropout_rate
        #=========================ABQR=========================
        self.lamda = lamda
        pro_max = num_q
        self.head = head

        # self.gcl = Multi_level_GCL(positive_matrix, contrast_batch, tau, lamda1, top_k, d, p, head, graph_aug, gnn_mode)
        d = emb_size
        self.gcl = BGRL(d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2)

        self.gcn = GCNConv(d, d, p)

        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)))
        nn.init.xavier_uniform_(self.pro_embed)

        self.ans_embed = nn.Embedding(2, d)

        self.attn = nn.MultiheadAttention(d, 8, dropout=p)
        self.attn_dropout = nn.Dropout(p)
        self.attn_layer_norm = nn.LayerNorm(d)

        self.FFN = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d),
            nn.Dropout(p),
        )
        self.FFN_layer_norm = nn.LayerNorm(d)

        self.pred = nn.Linear(d, 1)

        self.lstm = nn.LSTM(d, d, batch_first=True)

        self.origin_lstm = nn.LSTM(2 * d, 2 * d, batch_first=True)
        self.oppo_lstm = nn.LSTM(d, d, batch_first=True)

        self.origin_lstm2 = nn.LSTM(d, d, batch_first=True)
        self.oppo_lstm2 = nn.LSTM(d, d, batch_first=True)

        self.dropout2 = nn.Dropout(p=p)
        #=============================TCN|LSTM==========================
        # self.origin_out = nn.Sequential(
        #     nn.Linear(2 * d, d),
        #     nn.ReLU(),
        #     nn.Dropout(p=p),
        #     nn.Linear(d, 1)
        # )
        
        #=============================LSTM_TCN==========================
        self.origin_out = nn.Sequential(
            nn.Linear( d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )
        #=========================================================
        self.oppo_out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )
        self.origin_out2 = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )
        self.encoder_lstm = nn.LSTM(d, d, batch_first=True)
        self.decoder_lstm = nn.LSTM(d, d, batch_first=True)

        self.enc_token = nn.Parameter(torch.rand(1, d))
        self.enc_dec = nn.Linear(d, d)

        self.classify = nn.Sequential(
            nn.Linear(d, num_c)
        )
        # nn.Linear(d, skill_max)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        
        final_fc_dim = 512
        
        #=========================tcn==========================
        
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.model_name = "TCN_ABQR"
        self.max_seq_length = 200
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.num_channels = ast.literal_eval(num_channels)
        # 初始化 TCN 网络和线性输出层
        self.tcn = TemporalConvNet(self.emb_size, self.num_channels, kernel_size=kernel_size, dropout=dropout,num_heads=num_heads)
        self.convlstm = ConvLSTM(self.emb_size, self.num_channels, kernel_size=kernel_size)
        
        # self.linear = nn.Linear(self.num_channels[-1],emb_size)# 最后一层将 TCN 的输出映射到目标大小
        # self.init_weights()# 初始化权重
        self.capsule = capsule(d_model=self.num_channels[-1],n_question=num_c,kernel_sizes=[3,5],distance_decay=0.2)    
        self.position_encoding = nn.Embedding(self.max_seq_length, emb_size)

        self.combined_output = nn.Linear(2 * d, d)
        # print(f"attn_dropot :{attn_dropout}")
        # self.attention = MultiHeadAttention(d,num_heads=self.num_heads,dropout=self.attn_dropout_rate)
        # self.attention_small = MultiHeadAttention(self.num_channels[-1],num_heads=self.num_heads,dropout=self.attn_dropout_rate)
        # self.attention_decay = MultiHeadAttentionWithDecay(d,num_heads=self.num_heads,dropout=self.attn_dropout_rate)
        # print(f"type{type(num_channels[-1])},type2{type(self.num_heads)}")
        # self.attention = AttentionLayer(self.num_channels[-1], self.num_heads)
        self.attention = AttentionLayerDecay(self.emb_size, self.num_heads,dropout=self.attn_dropout_rate)
        # self.attention = AttentionLayer(self.emb_size, self.num_heads)
        self.out = nn.Sequential(
            nn.Linear(self.num_channels[-1],
                      final_fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(dropout),
            nn.Linear(256, emb_size)
        )
    # def init_weights(self):
    #     # 初始化线性层权重，正态分布
    #     self.linear.weight.data.normal_(0, 0.01)

    def compute_loss(self, pro_clas, true_clas):
        pro_clas = pro_clas.view(-1, pro_clas.shape[-1])
        true_clas = true_clas.view(-1)
        loss = F.cross_entropy(pro_clas, true_clas)
        return loss

    def encoding_mask_seq(self, x, mask_rate=0.3):
        # batch seq d
        num_nodes = x.shape[1]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[:, token_nodes] = self.enc_token
        return out_x, mask_nodes, keep_nodes
    
    
    def forward(self,last_pro, last_ans, last_skill, next_pro,
                next_skill, perb=None,pro_embed=None):
        # #print(f"input size:{x.shape}")
        
        
        #=================ABQR=================
        """
        last_pro: 当前交互序列的题目 ID。
        last_ans: 当前交互序列的答案（0/1）。
        last_skill: 当前交互序列的技能。
        next_pro: 下一题的题目 ID。
        next_skill: 下一题的技能。
        matrix: 题目交互图的邻接矩阵。
        perb: 对抗扰动（可选）。
        """
        device = last_pro.device

        batch, seq = last_pro.shape[0], last_pro.shape[1]

        
        #print(f"self.pro_embed{self.pro_embed.shape}")
        #print(f"matrix{matrix.shape}")
        # #print(f"perb{perb.shape}")
        #原版ABQR
        # pro_embed, contrast_loss = self.gcl(self.pro_embed, matrix, perb)
        
        
        
        # print(f"used ABQR")
        

        last_pro_embed = F.embedding(last_pro, pro_embed)
        next_pro_embed = F.embedding(next_pro, pro_embed)

        ans_embed = self.ans_embed(last_ans)

        X = last_pro_embed + ans_embed
        # print(f"X{X.shape}")
        X = self.dropout2(X)
        #print(f"input vec{X.shape}")#torch.Size([256, 199, 128])

        # X, _ = self.lstm(X)

        #===================输入层修改，在前面新加一个lstm=======================
        
        
        # X, _ = self.lstm(X)
        
        # print(f"input shape{X.shape}")
        
        # #===================TCN=================

        # xemb = X
        # xemb = xemb.permute(0, 2, 1)
        # # #print(f"embinput size:{xemb.shape}")
        
        # # 前向传播：输入数据经过 TCN 网络，最后只取最后一个时间步的输出
        # y1 = self.tcn(xemb)
        # y1 = y1.permute(0, 2, 1)
        
        # #print(f"tcn_out.shape is {y1.shape}")
        # # # #========================输出层版本1=========================
        # # # y1 = self.attention_small(y1)#加上attention
        
        
        
        # y2 = self.linear(y1[:, :, :])  
        # # y2 = self.attention(y2) 
        # # y2 = self.attention(y2)#加上attention2
        # # y2 = self.attention_decay(y2)#加上attention2
        # # print(f"y2.shape is {y2.shape}")

        # # print(f"next_pro_embed.shape is {next_pro_embed.shape}")
        
        # t1 = torch.cat([y2, next_pro_embed], dim=-1)
        # # print(f"t1.shape is {t1.shape}")
        
        # P = torch.sigmoid(self.origin_out(t1)).squeeze(-1)
        
        
        # false_P = P
        # #===========================================================
        
        # #===================输出层修改2 LSTM===================
        print(f"X.shape: {X.shape}")
        # y2 = self.linear(X)  # [batch, seq, emb_size]
        

        lstm_out, _ = self.lstm(X)  # [batch, seq, emb_size]
        # lstm_out, _ = self.lstm(y2)  # [batch, seq, emb_size]

        P = torch.sigmoid(self.origin_out(lstm_out)).squeeze(-1)  # [batch, seq]
        # #===========================================
        
        # # #========================输出层版本3=========================
        # # y1 = self.attention_small(y1)#加上attention
        
        
        # y2 = self.out(y1)
        # # y2 = self.linear(y1[:, :, :])  
        # # y2 = self.attention(y2) 
        # # y2 = self.attention(y2)#加上attention2
        # # y2 = self.attention_decay(y2)#加上attention2
        # # print(f"y2.shape is {y2.shape}")

        # # print(f"next_pro_embed.shape is {next_pro_embed.shape}")
        
        # t1 = torch.cat([y2, next_pro_embed], dim=-1)
        # # print(f"t1.shape is {t1.shape}")
        
        # P = torch.sigmoid(self.origin_out(t1)).squeeze(-1)
        
        
        # false_P = P
        
        
        
        
        # #==================DKT===================================
        # X, _ = self.lstm(X)
        # # 连接LSTM输出和下一题的嵌入，进行预测
        # concatenated = torch.cat([X, next_pro_embed], dim=-1)
        # #print(f"Concatenated X and next_pro_embed shape: {concatenated.shape}")  # [batch, seq, 2d] ([42, 199, 256])

        # # 通过全连接层和Sigmoid激活得到预测概率
        # P = torch.sigmoid(self.origin_out(concatenated)).squeeze(-1)
        # #==========================================================
        # #==================ConvLSTM===================================
        
        # # print("tthere!!!!!!!!!!!!!!!!")
        
        # X = self.convlstm(X)
        # # print("here!!!!!!!!!!!!!!!!")
        # # 连接LSTM输出和下一题的嵌入，进行预测
        # X = self.linear(X[:, :, :])  
        # # print(f"Concatenated X {X.shape}")  # 
        # # print(f"next_pro_embed X {next_pro_embed.shape}")  # 
        # concatenated = torch.cat([X, next_pro_embed], dim=-1)

        

        # # 通过全连接层和Sigmoid激活得到预测概率
        # P = torch.sigmoid(self.origin_out(concatenated)).squeeze(-1)
        # #==========================================================
        
        # #============================融合输出===================================
        # combined = torch.cat([X, y2, next_pro_embed], dim=-1)
        
        # P = torch.sigmoid(self.origin_out2(combined)).squeeze(-1)
        
        
        return P
    
    
class Chomp1d(nn.Module):#用于修正填充后的结果。时间卷积需要确保输出中每个时间步的信息仅依赖于当前或之前的时间步。为了避免未来信息泄露，卷积后的填充部分需要修剪。
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 修剪填充后的多余部分，保证因果卷积的时序一致性
        return x[:, :, :-self.chomp_size].contiguous()

    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, num_heads,dropout=0.2):
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
        self.bn1 = nn.BatchNorm1d(n_outputs)  # 添加BatchNorm
        self.bn2 = nn.BatchNorm1d(n_outputs)  # 添加BatchNorm
        # 构建序列化的网络结构（包括两层卷积和填充）
        # self.net = nn.Sequential(
        #     self.pad1, self.conv1, self.relu1, self.dropout1,
        #     self.pad2, self.conv2, self.relu2, self.dropout2
        # )


        self.net = nn.Sequential(
            self.pad1, self.conv1, self.bn1, self.relu1, self.dropout1,
            self.pad2, self.conv2, self.bn2, self.relu2, self.dropout2
        )
        # 下采样模块，用于调整输入和输出维度一致（跳跃连接）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        self.attn = AttentionLayerDecay(n_outputs, num_heads,dropout=dropout)
        self.attn_norm = nn.LayerNorm(n_outputs)
        
    def init_weights(self):
        # 初始化卷积层权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # print(f"out{out.shape}")
        out_transposed = out.permute(0, 2, 1) 
        attn_output= self.attn(out_transposed)
        attn_output = self.attn_norm(out_transposed + attn_output)
        attn_output = attn_output.permute(0, 2, 1)
        return self.relu(attn_output + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels,num_heads, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # 按层构建 TemporalBlock，每层的膨胀系数为 2^i
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,num_heads=num_heads,dropout=dropout)]

        self.network = nn.Sequential(*layers)# 将所有层串联成网络

    def forward(self, x):
        # #print(f"input size2:{x.shape}")
        
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
            # #print("args 1")
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
            # #print("args 2")
            
            data1, data2 = args
            u = torch.stack([data1, data2], dim=2)
            fused = self.dynamic_routing(u)
            return fused

class CausalDilatedConv1d(nn.Module):
    """
    一维因果膨胀卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalDilatedConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        padding = (self.kernel_size - 1) * self.dilation
        # 左侧填充，填充值为 -1
        x = F.pad(x, (padding, 0), "constant", -1)
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元，结合了因果卷积和膨胀卷积
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, dilation=1):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = CausalDilatedConv1d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, dilation=dilation)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # 当前隐藏状态和细胞状态

        # 拼接输入和隐藏状态
        combined = torch.cat([input_tensor, h_cur], dim=1)  # dim=1为通道维

        # 进行因果膨胀卷积
        conv_output = self.conv(combined)
        # conv_output 的通道数为 4 * hidden_dim，用于门控机制

        # 分割 gate
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        # 初始化隐藏状态和细胞状态为 0
        return (torch.zeros(batch_size, self.hidden_dim, 1, device=device),
                torch.zeros(batch_size, self.hidden_dim, 1, device=device))
        
        
class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM，结合因果卷积和膨胀卷积
    修改为使用 num_channels 列表定义每层的通道数
    """
    def __init__(self, emb_size, num_channels, kernel_size):
        """
        初始化多层 ConvLSTM

        参数：
        - emb_size (int): 输入序列的嵌入维度（第一层的输入通道数）
        - num_channels (List[int]): 每层的输出通道数列表，长度即为层数
        - kernel_size (int): 卷积核大小
        - dilation_rates (List[int]): 每层的膨胀率列表，长度应 >= 层数
        - device (torch.device): 设备（CPU 或 GPU）
        """
        super(ConvLSTM, self).__init__()

        self.num_layers = len(num_channels)
        self.num_channels = num_channels
        

        layers = []
        for i in range(self.num_layers):
            # dilation = dilation_rates[i] if i < len(dilation_rates) else 1
            dilation = 2 ** i
            input_channel = emb_size if i == 0 else num_channels[i-1]
            hidden_dim = num_channels[i]
            layers.append(ConvLSTMCell(input_channel, hidden_dim, kernel_size, dilation))

        self.layers = nn.ModuleList(layers)

    def forward(self, input_seq):
        """
        前向传播

        参数：
        - input_seq (torch.Tensor): 输入序列，形状为 (batch, seq_len, emb_size)

        返回：
        - output (torch.Tensor): 输出序列，形状为 (batch, seq_len, num_channels[-1])
        """
        device = input_seq.device
        # 转换输入形状为 (batch, emb_size, seq_len)
        input_seq = input_seq.transpose(1, 2)  # (batch, emb_size, seq_len)
        batch_size, _, seq_len = input_seq.size()

        # 初始化所有层的隐藏状态
        hidden_states = [cell.init_hidden(batch_size, device=device) for cell in self.layers]

        # 初始化每层的输出列表
        layer_outputs = [ [] for _ in range(self.num_layers) ]

        # 逐时间步处理序列
        for t in range(seq_len):
            x = input_seq[:, :, t:t+1]  # 当前时刻输入，形状: (batch, emb_size 或 hidden_dim, 1)
            for i, cell in enumerate(self.layers):
                h, c = hidden_states[i]
                h, c = cell(x, (h, c))
                hidden_states[i] = (h, c)
                x = h  # 下一层的输入是当前层的隐藏状态
                layer_outputs[i].append(h)

        # 将每层的时间步输出拼接
        final_outputs = []
        for i in range(self.num_layers):
            # 拼接时间步，形状: (batch, hidden_dim, seq_len)
            layer_output = torch.cat(layer_outputs[i], dim=2)
            final_outputs.append(layer_output)

        # 选择最后一层的输出作为最终输出，并转换形状为 (batch, seq_len, hidden_dim)
        output = final_outputs[-1].transpose(1, 2)  # (batch, seq_len, num_channels[-1})
        return output
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x 的形状: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        return x
    
class AttentionLayerDecay(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, zero_pad=True):
        super(AttentionLayerDecay, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.zero_pad = zero_pad
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 定义线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # 定义gamma参数，用于衰减机制
        # gamma的初始值可以根据需要调整
        self.gamma = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, x, pdiff=None):
        """
        x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        mask: 掩码张量，形状为 (batch_size, seq_len)
        pdiff: 位置差异张量，形状为 (batch_size, seq_len, seq_len)，可选
        """
        batch_size, seq_len, embed_dim = x.size()
        device = x.device
         # 4. 创建掩码
        # 假设没有外部掩码需求，只需创建一个全1的掩码
        
        # 1. 线性变换
        Q = self.q_linear(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_linear(x)  # (batch_size, seq_len, embed_dim)
        V = self.v_linear(x)  # (batch_size, seq_len, embed_dim)
        
        # 2. 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        head_dim = embed_dim // self.num_heads
        d_k = head_dim
        
        # 3. 计算缩放点积注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, num_heads, seq_len, seq_len)
        # print(f"socres{scores.shape}")
        # 4. 应用掩码

        mask = torch.ones(batch_size, seq_len, device=device).bool()  # (batch_size, seq_len)
    
        # 将 mask 扩展到 [batch_size, num_heads, seq_len, seq_len]
        mask = mask.unsqueeze(1).unsqueeze(2).expand(batch_size, self.num_heads, seq_len, seq_len)  # (batch_size, num_heads, seq_len, seq_len)
        
        scores = scores.masked_fill(mask == 0, -1e32)
        
        # 5. 计算标准softmax注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            attn_weights = attn_weights * mask.float()
        
        # 6. 计算累积分数和总分数
        distcum_scores = torch.cumsum(attn_weights, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        disttotal_scores = torch.sum(attn_weights, dim=-1, keepdim=True)  # (batch_size, num_heads, seq_len, 1)
        
        # 7. 计算位置差值
        x1 = torch.arange(seq_len).expand(seq_len, -1).to(device)  # (seq_len, seq_len)
        x2 = x1.transpose(0, 1).contiguous()  # (seq_len, seq_len)
        position_effect = torch.abs(x1 - x2).float()  # (seq_len, seq_len)
        position_effect = position_effect.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # 8. 计算距离分数，并进行截断
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)  # (batch_size, num_heads, seq_len, seq_len)
        dist_scores = dist_scores.sqrt().detach()  # (batch_size, num_heads, seq_len, seq_len)
        
        # 9. 处理gamma参数
        m = nn.Softplus()
        gamma = -1. * m(self.gamma).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, num_heads, 1, 1)
        
        # 10. 计算总效应，考虑是否使用pdiff
        if pdiff is None:
            total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)  # (batch_size, num_heads, seq_len, seq_len)
        else:
            # 假设pdiff的形状为 (batch_size, num_heads, seq_len, seq_len)
            diff = pdiff.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_len, seq_len)
            diff = diff.sigmoid().exp()
            total_effect = torch.clamp((dist_scores * gamma * diff).exp(), min=1e-5, max=1e5)  # (batch_size, num_heads, seq_len, seq_len)
        
        # 11. 将总效应应用到原始分数上
        scores = scores * total_effect  # (batch_size, num_heads, seq_len, seq_len)
        
        # 12. 再次应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e32)
        
        # 13. 重新计算softmax注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None and self.zero_pad:
            pad_zero = torch.zeros(batch_size, self.num_heads, 1, seq_len).to(device)
            attn_weights = torch.cat([pad_zero, attn_weights[:, :, 1:, :]], dim=2)  # 第一行分数置0
        
        # 14. 应用dropout
        attn_weights = self.dropout(attn_weights)
        
        # 15. 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 16. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # 17. 通过线性变换
        output = self.out_linear(attn_output)  # (batch_size, seq_len, embed_dim)
        
        # 18. 残差连接和层归一化
        output = self.layer_norm(x + self.dropout(output))
        
        return output
    

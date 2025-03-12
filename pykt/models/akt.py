import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import ast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.parameter import Parameter
# class LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         variance = ((x - mean) ** 2).mean(-1, keepdim=True)
#         x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
#         return self.weight * x + self.bias

# class CausalConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
#         super(CausalConv1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)

#     def forward(self, x):
#         return self.conv(x)[:, :, :-(self.conv.padding[0])]  # 因果卷积，切掉右侧填充部分

# class FrequencyLayer(nn.Module):
#     def __init__(self, dropout, hidden_size, kernel_size=5):
#         super(FrequencyLayer, self).__init__()
#         self.out_dropout = nn.Dropout(dropout)
#         self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
#         self.causal_conv = CausalConv1d(hidden_size, hidden_size, kernel_size)
#         self.c = kernel_size // 2 + 1
#         self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

#     def forward(self, input_tensor):
#         # [batch, seq_len, hidden]
#         batch, seq_len, hidden = input_tensor.shape

#         # 因果卷积
#         input_tensor = input_tensor.permute(0, 2, 1)  # 转换为 [batch, hidden, seq_len]
#         low_pass = self.causal_conv(input_tensor)
#         low_pass = low_pass.permute(0, 2, 1)  # 转换回 [batch, seq_len, hidden]

#         # 高通滤波器
#         high_pass = input_tensor.permute(0, 2, 1) - low_pass  # 转换为 [batch, seq_len, hidden]
#         sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

#         hidden_states = self.out_dropout(sequence_emb_fft)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor.permute(0, 2, 1))

#         return hidden_states

# class Dim(IntEnum):
#     batch = 0
#     seq = 1
#     feature = 2

# class AKT(nn.Module):
#     def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
#             kq_same=1, final_fc_dim=512, num_attn_heads=4, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
#         super().__init__()
#         """
#         Input:
#             d_model: dimension of attention block
#             final_fc_dim: dimension of final fully connected net before prediction
#             num_attn_heads: number of heads in multi-headed attention
#             d_ff : dimension for fully conntected net inside the basic block
#             kq_same: if key query same, kq_same=1, else = 0
#         """
#         self.model_name = "akt"
#         self.n_question = n_question
#         self.dropout = dropout
#         self.kq_same = kq_same
#         self.n_pid = n_pid
#         self.l2 = l2
#         self.model_type = self.model_name
#         self.separate_qa = separate_qa
#         self.emb_type = emb_type
#         embed_l = d_model
#         if self.n_pid > 0:
#             self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
#             self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
#             self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
#         if emb_type.startswith("qid"):
#             # n_question+1 ,d_model
#             self.q_embed = nn.Embedding(self.n_question, embed_l)
#             if self.separate_qa: 
#                 self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
#             else: # false default
#                 self.qa_embed = nn.Embedding(2, embed_l)

#         # Architecture Object. It contains stack of attention block
#         self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
#                                     d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

#         self.out = nn.Sequential(
#             nn.Linear(d_model + embed_l,
#                       final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
#             nn.Linear(final_fc_dim, 256), nn.ReLU(
#             ), nn.Dropout(self.dropout),
#             nn.Linear(256, 1)
#         )
#         self.reset()

#     def reset(self):
#         for p in self.parameters():
#             if p.size(0) == self.n_pid+1 and self.n_pid > 0:
#                 torch.nn.init.constant_(p, 0.)

#     def base_emb(self, q_data, target):
#         q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
#         if self.separate_qa:
#             qa_data = q_data + self.n_question * target
#             qa_embed_data = self.qa_embed(qa_data)
#         else:
#             # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
#             qa_embed_data = self.qa_embed(target)+q_embed_data
#         return q_embed_data, qa_embed_data

#     def forward(self, q_data, target, pid_data=None, qtest=False):
#         emb_type = self.emb_type
#         # Batch First
#         if emb_type.startswith("qid"):
#             q_embed_data, qa_embed_data = self.base_emb(q_data, target)

#         pid_embed_data = None
#         if self.n_pid > 0: # have problem id
#             q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
#             pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
#             q_embed_data = q_embed_data + pid_embed_data * \
#                 q_embed_diff_data  # uq *d_ct + c_ct # question encoder

#             qa_embed_diff_data = self.qa_embed_diff(
#                 target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
#             if self.separate_qa:
#                 qa_embed_data = qa_embed_data + pid_embed_data * \
#                     qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
#             else:
#                 qa_embed_data = qa_embed_data + pid_embed_data * \
#                     (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
#             c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2 # rasch部分loss
#         else:
#             c_reg_loss = 0.

#         # BS.seqlen,d_model
#         # Pass to the decoder
#         # output shape BS,seqlen,d_model or d_model//2
#         d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

#         concat_q = torch.cat([d_output, q_embed_data], dim=-1)
#         # print("concat_q.shape:", concat_q.shape)
#         output = self.out(concat_q).squeeze(-1)
#         m = nn.Sigmoid()
#         preds = m(output)
#         # print("akt_preds.shape:", preds.shape)
#         if not qtest:
#             return preds, c_reg_loss
#         else:
#             return preds, c_reg_loss, concat_q


# class Architecture(nn.Module):
#     def __init__(self, n_question,  n_blocks, d_model, d_feature,
#                  d_ff, n_heads, dropout, kq_same, model_type, emb_type, output_attention=False):
#         super().__init__()
#         """
#             n_block : number of stacked blocks in the attention
#             d_model : dimension of attention input/output
#             d_feature : dimension of input in each of the multi-head attention part.
#             n_head : number of heads. n_heads*d_feature = d_model
#         """
#         self.d_model = d_model
#         self.model_type = model_type

#         if model_type in {'akt'}:
#             self.blocks_1 = nn.ModuleList([
#                 TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
#                                  d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type, output_attention=False)
#                 for _ in range(n_blocks)
#             ])
#             self.blocks_2 = nn.ModuleList([
#                 TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
#                                  d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type, output_attention=False)
#                 for _ in range(n_blocks*2)
#             ])
#         # self.filter_layer = FrequencyLayer(dropout,d_model)

#     def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
#         # target shape  bs, seqlen
#         seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

#         qa_pos_embed = qa_embed_data
#         q_pos_embed = q_embed_data

#         y = qa_pos_embed
#         seqlen, batch_size = y.size(1), y.size(0)
#         x = q_pos_embed
#         x = self.filter_layer(x)
#         y = self.filter_layer(y)


#         # encoder
#         for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
#             y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data) # yt^
#         flag_first = True
#         for block in self.blocks_2:
#             if flag_first:  # peek current question
#                 x = block(mask=1, query=x, key=x,
#                           values=x, apply_pos=False, pdiff=pid_embed_data) # False: 没有FFN, 第一层只有self attention, 对应于xt^
#                 flag_first = False
#             else:  # dont peek current response
#                 x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
#                 # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
#                 # print(x[0,0,:])
#                 flag_first = True
#         return x

# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, d_feature,
#                  d_ff, n_heads, dropout,  kq_same, emb_type, output_attention=False):
#         super().__init__()
#         """
#             This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
#         """
#         kq_same = kq_same == 1
#         # Multi-Head Attention Block
#         self.masked_attn_head = MultiHeadAttention(
#             d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

#         # Two layer norm layer and two droput layer
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ff, d_model)

#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)
#         self.output_attention = output_attention
#         self.attn_weights = [] 

#     def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
#         """
#         Input:
#             block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
#             mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
#             query : Query. In transformer paper it is the input for both encoder and decoder
#             key : Keys. In transformer paper it is the input for both encoder and decoder
#             Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

#         Output:
#             query: Input gets changed over the layer and returned.

#         """

#         seqlen, batch_size = query.size(1), query.size(0)
#         nopeek_mask = np.triu(
#             np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
#         src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
#         if mask == 0:  # If 0, zero-padding is needed.
#             # Calls block.masked_attn_head.forward() method
#             query2, attn_weights = self.masked_attn_head(
#                 query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
#         else:
#             # Calls block.masked_attn_head.forward() method
#             query2, attn_weights = self.masked_attn_head(
#                 query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)
#         if self.output_attention:
#           self.attn_weights.append(attn_weights)
#         query = query + self.dropout1((query2)) # 残差1
#         query = self.layer_norm1(query) # layer norm
#         if apply_pos:
#             query2 = self.linear2(self.dropout( # FFN
#                 self.activation(self.linear1(query))))
#             query = query + self.dropout2((query2)) # 残差
#             query = self.layer_norm2(query) # lay norm
#         return query


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
#         super().__init__()
#         """
#         It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
#         """
#         self.d_model = d_model
#         self.emb_type = emb_type
#         if emb_type.endswith("avgpool"):
#             # pooling
#             #self.pool =  nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
#             pool_size = 3
#             self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
#             self.out_proj = nn.Linear(d_model, d_model, bias=bias)
#         elif emb_type.endswith("linear"):
#             # linear
#             self.linear = nn.Linear(d_model, d_model, bias=bias)
#             self.out_proj = nn.Linear(d_model, d_model, bias=bias)
#         elif emb_type.startswith("qid"):
#             self.d_k = d_feature
#             self.h = n_heads
#             self.kq_same = kq_same

#             self.v_linear = nn.Linear(d_model, d_model, bias=bias)
#             self.k_linear = nn.Linear(d_model, d_model, bias=bias)
#             if kq_same is False:
#                 self.q_linear = nn.Linear(d_model, d_model, bias=bias)
#             self.dropout = nn.Dropout(dropout)
#             self.proj_bias = bias
#             self.out_proj = nn.Linear(d_model, d_model, bias=bias)
#             self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
#             torch.nn.init.xavier_uniform_(self.gammas)
#             self._reset_parameters()


#     def _reset_parameters(self):
#         xavier_uniform_(self.k_linear.weight)
#         xavier_uniform_(self.v_linear.weight)
#         if self.kq_same is False:
#             xavier_uniform_(self.q_linear.weight)

#         if self.proj_bias:
#             constant_(self.k_linear.bias, 0.)
#             constant_(self.v_linear.bias, 0.)
#             if self.kq_same is False:
#                 constant_(self.q_linear.bias, 0.)
#             # constant_(self.attnlinear.bias, 0.)
#             constant_(self.out_proj.bias, 0.)

#     def forward(self, q, k, v, mask, zero_pad, pdiff=None):

#         bs = q.size(0)

#         if self.emb_type.endswith("avgpool"):
#             # v = v.transpose(1,2)
#             scores = self.pooling(v)
#             concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
#             # concat = concat.transpose(1,2)#.contiguous().view(bs, -1, self.d_model)
#         elif self.emb_type.endswith("linear"):
#             # v = v.transpose(1,2)
#             scores = self.linear(v)
#             concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
#             # concat = concat.transpose(1,2)
#         elif self.emb_type.startswith("qid"):
#             # perform linear operation and split into h heads

#             k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
#             if self.kq_same is False:
#                 q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
#             else:
#                 q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
#             v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

#             # transpose to get dimensions bs * h * sl * d_model

#             k = k.transpose(1, 2)
#             q = q.transpose(1, 2)
#             v = v.transpose(1, 2)
#             # calculate attention using function we will define next
#             gammas = self.gammas
#             if self.emb_type.find("pdiff") == -1:
#                 pdiff = None
#             scores, attention_map = attention(q, k, v, self.d_k,
#                             mask, self.dropout, zero_pad, gammas, pdiff)

#             # concatenate heads and put through final linear layer
#             concat = scores.transpose(1, 2).contiguous()\
#                 .view(bs, -1, self.d_model)

#         output = self.out_proj(concat)

#         return output, attention_map

#     def pad_zero(self, scores, bs, dim, zero_pad):
#         if zero_pad:
#             # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
#             pad_zero = torch.zeros(bs, 1, dim).to(device)
#             scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
#         return scores


# def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
#     """
#     This is called by Multi-head atention object to find the values.
#     """
#     # d_k: 每一个头的dim
#     scores = torch.matmul(q, k.transpose(-2, -1)) / \
#         math.sqrt(d_k)  # BS, 8, seqlen, seqlen
#     bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

#     x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
#     x2 = x1.transpose(0, 1).contiguous()

#     with torch.no_grad():
#         scores_ = scores.masked_fill(mask == 0, -1e32)
#         scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
#         scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
#         distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
#         disttotal_scores = torch.sum(
#             scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
#         # print(f"distotal_scores: {disttotal_scores}")
#         position_effect = torch.abs(
#             x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
#         # bs, 8, sl, sl positive distance
#         dist_scores = torch.clamp(
#             (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
#         dist_scores = dist_scores.sqrt().detach()
#     m = nn.Softplus()
#     gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
#     # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
#     if pdiff == None:
#         total_effect = torch.clamp(torch.clamp(
#             (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
#     else:
#         diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
#         diff = diff.sigmoid().exp()
#         total_effect = torch.clamp(torch.clamp(
#             (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
#     scores = scores * total_effect

#     # scores.masked_fill_(combined_mask == 0, -1e32)
#     scores.masked_fill_(mask == 0, -1e32)
#     scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
#     # print(f"before zero pad scores: {scores.shape}")
#     # print(zero_pad)
#     if zero_pad:
#         pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
#         scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
#     # print(f"after zero pad scores: {scores}")
#     scores = dropout(scores)
#     output = torch.matmul(scores, v)
#     # import sys
#     # sys.exit()
#     return output, scores


# class LearnablePositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = 0.1 * torch.randn(max_len, d_model)
#         pe = pe.unsqueeze(0)
#         self.weight = nn.Parameter(pe, requires_grad=True)

#     def forward(self, x):
#         return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


# class CosinePositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = 0.1 * torch.randn(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.weight = nn.Parameter(pe, requires_grad=False)

#     def forward(self, x):
#         return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
#-------------------------------------------AKT_CasualConv_DynRouting-----------------------------------------------

### 来自AKT代码的组件 ###

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x + self.bias

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-(self.conv.padding[0])]  # 因果卷积，切掉右侧填充部分

class FrequencyLayer(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_size=5):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalConv1d(hidden_size, hidden_size, kernel_size)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        input_tensor_t = input_tensor.permute(0, 2, 1) 
        low_pass = self.causal_conv(input_tensor_t)
        low_pass = low_pass.permute(0, 2, 1)
        high_pass = input_tensor - low_pass 
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()
        self.kernel_bias = ParallelKerpleLog(n_heads)
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            nn.init.xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            nn.init.constant_(self.k_linear.bias, 0.)
            nn.init.constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                nn.init.constant_(self.q_linear.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        scores, attention_map = attention(q, k, v, self.d_k,
                                          mask, self.dropout, zero_pad, gammas,kernel_bias=self.kernel_bias)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output, attention_map

# def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
#     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
#     bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
#     x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
#     x2 = x1.transpose(0, 1).contiguous()
#     with torch.no_grad():
#         scores_ = scores.masked_fill(mask == 0, -1e32)
#         scores_ = F.softmax(scores_, dim=-1)
#         scores_ = scores_ * mask.float().to(device)
#         distcum_scores = torch.cumsum(scores_, dim=-1)
#         disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
#         position_effect = torch.abs(x1 - x2)[None, None, :, :].float().to(device)
#         dist_scores = torch.clamp((disttotal_scores - distcum_scores)*position_effect, min=0.)
#         dist_scores = dist_scores.sqrt().detach()

#     m = nn.Softplus()
#     gamma = -1. * m(gamma).unsqueeze(0)  
#     if pdiff is None:
#         total_effect = torch.clamp(torch.clamp((dist_scores*gamma).exp(), min=1e-5), max=1e5)
#     else:
#         diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
#         diff = diff.sigmoid().exp()
#         total_effect = torch.clamp(torch.clamp((dist_scores*gamma*diff).exp(), min=1e-5), max=1e5)

#     scores = scores * total_effect
#     scores.masked_fill_(mask == 0, -1e32)
#     scores = F.softmax(scores, dim=-1)
#     if zero_pad:
#         pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
#         scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
#     scores = dropout(scores)
#     output = torch.matmul(scores, v)
#     return output, scores

def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None,kernel_bias=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    scores = kernel_bias(scores)
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output,scores





# def attention(q, k, v, d_k, mask, dropout, zero_pad):
#     """
#     This is called by Multi-head atention object to find the values.
#     """
#     # d_k: 每一个头的dim
#     scores = torch.matmul(q, k.transpose(-2, -1)) / \
#         math.sqrt(d_k)  # BS, 8, seqlen, seqlen
#     bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

#     scores.masked_fill_(mask == 0, -1e32)
#     scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
#     # print(f"before zero pad scores: {scores.shape}")
#     # print(zero_pad)
#     if zero_pad:
#         pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
#         scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
#     # print(f"after zero pad scores: {scores}")
#     scores = dropout(scores)
#     output = torch.matmul(scores, v)
#     # import sys
#     # sys.exit()
    # return output, scores

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type, output_attention=False):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.output_attention = output_attention
        self.attn_weights = [] 

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            query2, attn_weights = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff)
        else:
            query2, attn_weights = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)
        if self.output_attention:
            self.attn_weights.append(attn_weights)
        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type,seq_len, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.filter_layer = FrequencyLayer(dropout, d_model)

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type, output_attention=False)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type, output_attention=False)
                for _ in range(n_blocks*2)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        
    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        
        
        # q_posemb = self.position_emb(q_embed_data)
        # q_embed_data = q_embed_data + q_posemb
        # qa_posemb = self.position_emb(qa_embed_data)
        # qa_embed_data = qa_embed_data + qa_posemb
        # qa_pos_embed = qa_embed_data
        # q_pos_embed = q_embed_data

        # y = qa_pos_embed
        # seqlen, batch_size = y.size(1), y.size(0)
        # x = q_pos_embed
        
        
        
        
        x = q_embed_data
        y = qa_embed_data
        x = self.filter_layer(x)
        y = self.filter_layer(y)

        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False, pdiff=pid_embed_data)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data)
                flag_first = True
        return x


### 多尺度分解与动态路由组件 ###

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

def squash(v, dim=-1):
    norm = torch.norm(v, dim=dim, keepdim=True)
    scale = (norm**2) / (1 + norm**2)
    direction = v / (norm + 1e-9)
    return scale * direction

def dynamic_routing(input_vecs, num_iterations=2):
    N, L, M, C = input_vecs.shape
    logits = torch.zeros(N, L, M, device=input_vecs.device)
    for i in range(num_iterations):
        c = F.softmax(logits, dim=-1)  # [N, L, M]
        c_exp = c.unsqueeze(-1)
        s = torch.sum(c_exp * input_vecs, dim=2)  # [N, L, C]
        v = squash(s, dim=-1)
        if i < num_iterations - 1:
            sim = torch.sum(input_vecs * v.unsqueeze(2), dim=-1)  # [N, L, M]
            logits = logits + sim
    return v


### 最终整合后的模型 ###

class AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
                 kq_same=1, final_fc_dim=512, num_attn_heads=4, separate_qa=False, l2=1e-5, 
                 emb_type="qid", emb_path="", pretrain_dim=768,
                 kernel_sizes="[3,5,7,9]", distance_decay=0.2,seq_len=200):
        super().__init__()
        self.model_name = "akt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.seq_len = seq_len
        self.l2 = l2
        self.model_type = "akt"
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.distance_decay = distance_decay
        self.d_model = d_model

        self.kernel_sizes = ast.literal_eval(kernel_sizes)
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) 
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type,seq_len=self.seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        # 多尺度分解与动态路由
        self.decomposition = MultiScaleConvDecomp(kernel_sizes=self.kernel_sizes, d_model=d_model)
        self.Linear_Seasonal = nn.Linear(d_model, n_question)
        self.Linear_Trend = nn.Linear(d_model, n_question)
        self.Linear_Seasonal.weight = nn.Parameter((1/d_model)*torch.ones([n_question,d_model]))
        self.Linear_Trend.weight = nn.Parameter((1/d_model)*torch.ones([n_question,d_model]))
        self.alpha = nn.Parameter(torch.tensor(0.7))

        # 融合层，将 d_output, q_embed_data, seasonal_fused, trend_fused 融合后映射回d_model维
        self.fusion = nn.Linear(4*d_model, d_model)

        self.reset()

    def reset(self):
        if self.n_pid > 0:
          for p in self.parameters():
            if p.dim() > 0 and p.size(0) == self.n_pid + 1:
              torch.nn.init.constant_(p, 0.)


    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

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

    def forward(self, q_data, target, pid_data=None, qtest=False):
      # 不切片 q_data 和 target，而是在内部对 target 进行shift
      # target_shifted 在时间步t使用原始target[t-1]，避免标签泄露
      target_shifted = torch.cat([torch.zeros_like(target[:,0:1]), target[:, :-1]], dim=1)
      # q_data与target保持相同长度 L
      
      
      #这里提取出了base_emb的东西
      q_embed_data = self.q_embed(q_data)
      if self.separate_qa:
          qa_data = q_data + self.n_question * target_shifted
          qa_embed_data = self.qa_embed(qa_data)
      else:
          qa_embed_data = self.qa_embed(target_shifted) + q_embed_data

      pid_embed_data = None
      if self.n_pid > 0:
          q_embed_diff_data = self.q_embed_diff(q_data)
          pid_embed_data = self.difficult_param(pid_data)
          q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

          qa_embed_diff_data = self.qa_embed_diff(target_shifted)
          if self.separate_qa:
              qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
          else:
              qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)
          c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
      else:
          c_reg_loss = 0.

      seasonal_list, trend_list = self.decomposition(qa_embed_data)
      seasonal_stack = torch.stack(seasonal_list, dim=2)  # [N,L,M,C]
      trend_stack = torch.stack(trend_list, dim=2)        # [N,L,M,C]

      seasonal_fused = dynamic_routing(seasonal_stack, num_iterations=3) # [N,L,C]
      trend_fused = dynamic_routing(trend_stack, num_iterations=3)       # [N,L,C]

      # 可选的时间加权
      seasonal_fused = self.time_attention(seasonal_fused)
      trend_fused = self.time_attention(trend_fused)

      d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data) # [N,L,d_model]

      concat_all = torch.cat([d_output, q_embed_data, seasonal_fused, trend_fused], dim=-1) # [N,L,4*d_model]
      fused = self.fusion(concat_all) # [N,L,d_model]

      output = self.out(fused).squeeze(-1) # [N,L]
      preds = torch.sigmoid(output)

      if not qtest:
          return preds, c_reg_loss
      else:
          return preds, c_reg_loss, fused


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
    
class ParallelKerpleLog(nn.Module):
    """Kernel Bias"""
    def __init__(self, num_attention_heads):
        super().__init__()
        self.heads = num_attention_heads  # int
        self.num_heads_per_partition = self.heads  # int
        # self.pos_emb = pos_emb  # str
        self.eps = 1e-2
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                    self.num_heads_per_partition,
                    dtype=torch.float32,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                    self.num_heads_per_partition,
                    dtype=torch.float32,
                )[:, None, None] * scale)
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')
        self.cached_matrix = None
        self.cached_seq_len = None
    
    def stats(self):
        def get_stats(name, obj):
            return {
                name + '_mean': obj.mean().detach().cpu(),
                name + '_std': obj.std().detach().cpu(),
                name + '_max': obj.max().detach().cpu(),
                name + '_min': obj.min().detach().cpu()
            }
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd
    
    def forward(self, x):
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel
        
        if seq_len_q != seq_len_k:  
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if not isinstance(bias, float):
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
        return x + bias

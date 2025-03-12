import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import sys
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib
ii = 0
font_size  = 25
font_bold = 'bold'
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = font_size  # Adjust font size as needed
matplotlib.rcParams['font.weight'] = font_bold 
stored_scores_0 = []  # 用于存储 ifTime=0 的 scores
stored_scores_1 = []  # 用于存储 ifTime=1 的 scores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, dropout, hidden_size, kernel_size=3):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalConv1d(hidden_size, hidden_size, kernel_size)
        # self.c = kernel_size // 2 + 1
        # self.c = 5
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # 因果卷积
        input_tensor = input_tensor.permute(0, 2, 1)  # 转换为 [batch, hidden, seq_len]
        low_pass = self.causal_conv(input_tensor)
        low_pass = low_pass.permute(0, 2, 1)  # 转换回 [batch, seq_len, hidden]

        # 高通滤波器
        high_pass = input_tensor.permute(0, 2, 1) - low_pass  # 转换为 [batch, seq_len, hidden]
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor.permute(0, 2, 1))

        return hidden_states

class FrequencyLayer2(nn.Module):
    def __init__(self, dropout,hidden_size):
        super(FrequencyLayer2, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.c = 10
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FrequencyLayer3(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_sizes):
        super(FrequencyLayer3, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        kernel_sizess = [kernel_sizes, kernel_sizes+2]
        # 创建多个因果卷积层
        self.causal_convs = nn.ModuleList([
            CausalConv1d(hidden_size, hidden_size, ks) for ks in kernel_sizess
        ])
        
        # 为每个卷积层创建对应的 sqrt_beta 参数
        self.sqrt_betas = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, hidden_size)) for _ in kernel_sizess
        ])
        
        # 可学习的权重参数用于合并
        self.combine_weights = nn.Parameter(torch.randn(len(kernel_sizess)))
    
    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        
        # 准备输入以适应卷积操作
        input_tensor_conv = input_tensor.permute(0, 2, 1)  # [batch, hidden, seq_len]
        
        # 存储每个卷积层的输出
        sequence_emb_fft_list = []
        
        for i, causal_conv in enumerate(self.causal_convs):
            # 低通滤波
            low_pass = causal_conv(input_tensor_conv)  # [batch, hidden, seq_len]
            low_pass = low_pass.permute(0, 2, 1)  # [batch, seq_len, hidden]
            
            # 高通滤波
            high_pass = input_tensor - low_pass  # [batch, seq_len, hidden]
            
            # 组合低频和高频成分
            sqrt_beta = self.sqrt_betas[i]
            sequence_emb_fft = low_pass + (sqrt_beta ** 2) * high_pass  # [batch, seq_len, hidden]
            
            sequence_emb_fft_list.append(sequence_emb_fft)
        
        # 将多个卷积层的输出堆叠 [num_kernels, batch, seq_len, hidden]
        sequence_emb_fft_stack = torch.stack(sequence_emb_fft_list, dim=0)
        
        # 对合并权重进行softmax归一化
        combine_weights = torch.softmax(self.combine_weights, dim=0)  # [num_kernels]
        combine_weights = combine_weights.view(-1, 1, 1, 1)  # [num_kernels, 1, 1, 1]
        
        # 加权求和合并
        combined_sequence_emb_fft = (combine_weights * sequence_emb_fft_stack).sum(dim=0)  # [batch, seq_len, hidden]
        
        # Dropout和层归一化
        hidden_states = self.out_dropout(combined_sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class timeGap2(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size) -> None:
        super().__init__()
        self.num_rgap, self.num_sgap, self.num_pcount = num_rgap, num_sgap, num_pcount
        if num_rgap != 0:
            self.rgap_eye = torch.eye(num_rgap)
        if num_sgap != 0:
            self.sgap_eye = torch.eye(num_sgap)
        if num_pcount != 0:
            self.pcount_eye = torch.eye(num_pcount)

        input_size = num_rgap + num_sgap + num_pcount
        
        print(f"self.num_rgap: {self.num_rgap}, self.num_sgap: {self.num_sgap}, self.num_pcount: {self.num_pcount}, input_size: {input_size}")

        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
        infs = []
        if self.num_rgap != 0:
            rgap = self.rgap_eye[rgap].to(device)
            infs.append(rgap)
        if self.num_sgap != 0:
            sgap = self.sgap_eye[sgap].to(device)
            infs.append(sgap)
        if self.num_pcount != 0:
            pcount = self.pcount_eye[pcount].to(device)
            infs.append(pcount)

        tg = torch.cat(infs, -1)
        tg_emb = self.time_emb(tg)

        return tg_emb

class BAKTTime(nn.Module):
    def __init__(self, n_question, n_pid, num_rgap, num_sgap, num_pcount, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, kernel_size=7, freq=True,
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "bakt_time"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=self.emb_type, kernel_size=kernel_size, freq=freq).to(device)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        if emb_type.endswith("hasw") != -1:
            self.c_weight = nn.Linear(d_model, d_model)
        
        if emb_type in ["qidonlyrgap", "qidonlysgap", "qidonlypcount", "qidrsgap", "qidrpgap", "qidspgap"]:
            self.c_weight = nn.Linear(d_model, d_model)
            self.t_weight = nn.Linear(d_model, d_model)
        
        if emb_type.endswith("onlyrgap"):
            self.time_emb = timeGap2(num_rgap, 0, 0, d_model)
        if emb_type.endswith("onlysgap"):
            self.time_emb = timeGap2(0, num_sgap, 0, d_model)
        if emb_type.endswith("onlypcount"):
            self.time_emb = timeGap2(0, 0, num_pcount, d_model)
            
        if emb_type.endswith("rsgap"):
            self.time_emb = timeGap2(num_rgap, num_sgap, 0, d_model)
        if emb_type.endswith("rpgap"):
            self.time_emb = timeGap2(num_rgap, 0, num_pcount, d_model)
        if emb_type.endswith("spgap"):
            self.time_emb = timeGap2(0, num_sgap, num_pcount, d_model)
            
        if emb_type in ["qidonlyrgap", "qidonlysgap", "qidonlypcount", "qidrsgap", "qidrpgap", "qidspgap"]:
            self.model2 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
                                    dropout=dropout, d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,
                                    kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len,emb_type=emb_type, kernel_size=kernel_size, freq=freq)

        if self.emb_type == "qid" or self.emb_type in ["qidtrue","qidfalse"]:
            self.c_weight = nn.Linear(d_model, d_model)
            self.t_weight = nn.Linear(d_model, d_model)
            self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
            self.model2 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
                                       dropout=dropout, d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,
                                       kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len,emb_type=emb_type, kernel_size=kernel_size, freq=freq)

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, dcur, dgaps, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        emb_type = self.emb_type
        q_data = q_data.to(device)
        target = target.to(device)
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data.to(device))  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

        if emb_type == "qid" or emb_type in ["qidonlyrgap", "qidonlysgap", "qidonlypcount", "qidrsgap", "qidrpgap", "qidspgap", "qidtrue","qidfalse"]:
            rg, sg, p = dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long()
            rgshft, sgshft, pshft = dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long()

            r_gaps = torch.cat((rg[:, 0:1], rgshft), dim=1)
            s_gaps = torch.cat((sg[:, 0:1], sgshft), dim=1)
            pcounts = torch.cat((p[:, 0:1], pshft), dim=1)

            temb = self.time_emb(r_gaps, s_gaps, pcounts)
            # time attention
            # t_out = self.model2(temb, self.qa_embed(target)+temb)
            t_out = self.model2(temb, qa_embed_data,ifTime = 1) # 计算时间信息和基本信息的attention？
        #ifTime = 1说明是time attention
        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        
        # elif emb_type in ["qidonlyrgap", "qidonlysgap", "qidonlypcount", "qidrsgap", "qidrpgap", "qidspgap"]
        if emb_type == "qid" or emb_type in ["qidonlyrgap", "qidonlysgap", "qidonlypcount", "qidrsgap", "qidrpgap", "qidspgap","qidtrue","qidfalse"]:
            d_output = self.model(q_embed_data, qa_embed_data,ifTime = 0)
            
            w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_out)) # w = sigmoid(基本信息编码 + 时间信息编码)，每一维设置为0-1之间的数值
            print("t_out plus one times")
            
            d_output = w * d_output + (1 - w) * t_out # 每一维加权平均后的综合信息
            q_embed_data = q_embed_data + temb # 原始的题目信息和时间信息

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.endswith("hasw"):
            d_output = self.model(q_embed_data, qa_embed_data)
            
            w = torch.sigmoid(self.c_weight(d_output))
            d_output = w * d_output
            
            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len, emb_type, kernel_size, freq):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        self.emb_type =emb_type
        self.freq = freq
        if model_type in {'bakt_time'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        if self.emb_type.find("true") != -1:
            self.filter_layer = FrequencyLayer3(dropout,d_model, kernel_size)
        else:
            self.filter_layer = FrequencyLayer(dropout,d_model, kernel_size)
        #   self.filter_layer2 = FrequencyLayer2(dropout,d_model)

    def forward(self, q_embed_data, qa_embed_data,ifTime = 0):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        if self.emb_type.find("true") != -1:
          x = self.filter_layer(x)
          y = self.filter_layer(y)
        #   print("F3 True")
        #   import sys
        #   sys.exit()
        else:
          x = self.filter_layer(x)
          y = self.filter_layer(y)
        #   print("F1 True")
        #   import sys
        #   sys.exit()

        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True,ifTime = ifTime) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True,ifTime = 0):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True,ifTime = ifTime) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False,ifTime = ifTime)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
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

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)
    
            
    def forward(self, q, k, v, mask, zero_pad,ifTime = 0):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        #
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad,ifTime=ifTime)
        
        
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

def visualize_all_attention_scores(scores, ifTime):
    """
    可视化前 10 个批次的注意力得分，并对多个头求平均，仅显示左上角 25x25 的矩阵区域。
    每四次函数调用后，根据 ifTime 对数据进行配对，并画图保存。

    参数:
    - scores: 注意力得分，形状为 (batch_size, num_heads, query_len, key_len)
    - ifTime: 控制颜色调，0 为蓝色，1 为红色
    """
    global ii
    global stored_scores_0, stored_scores_1

    print(f"Type of scores: {type(scores)}")
    batch_size, num_heads, query_len, key_len = scores.shape

    # 根据 ifTime 存储数据
    if ifTime == 0:
        stored_scores_0.append(scores)
    else:
        stored_scores_1.append(scores)

    # 确保 'picture' 目录存在
    if not os.path.exists("picture"):
        os.makedirs("picture")

    # 当 ifTime=0 和 ifTime=1 都存储了两个矩阵时，开始配对绘制
    if len(stored_scores_0) >= 2 and len(stored_scores_1) >= 2 and len(stored_scores_0) == len(stored_scores_1):
        for pair_idx in range(2):  # 处理两对图
            # 分别获取 ifTime=0 和 ifTime=1 的 scores
            scores_0 = stored_scores_0.pop(0)
            scores_1 = stored_scores_1.pop(0)

            # 限制批次数量为前 10 个
            for batch_idx in range(min(batch_size, 10)):
                # 计算平均分数并提取 25x25 的子矩阵
                avg_attention_scores_0 = scores_0[batch_idx].mean(dim=0)[:25, :25]
                avg_attention_scores_1 = scores_1[batch_idx].mean(dim=0)[:25, :25]

                # 创建配对图像
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 两个子图，左右排列

                # 绘制 ifTime=0 的图（蓝色）
                sns.heatmap(avg_attention_scores_0.detach().cpu().numpy(), cmap='Blues', ax=axes[0])
                axes[0].set_title(f'Question Domain Attention',fontweight=font_bold,fontsize=font_size )
                axes[0].set_xlabel('Key Positions',fontweight=font_bold,fontsize=font_size)
                axes[0].set_ylabel('Query Positions',fontweight=font_bold,fontsize=font_size)

                # 绘制 ifTime=1 的图（红色）
                sns.heatmap(avg_attention_scores_1.detach().cpu().numpy(), cmap='Reds', ax=axes[1])
                axes[1].set_title(f'Time Domain Attention',fontweight=font_bold,fontsize=font_size)
                axes[1].set_xlabel('Key Positions',fontweight=font_bold,fontsize=font_size)
                axes[1].set_ylabel('Query Positions',fontweight=font_bold,fontsize=font_size)

                # 调整布局，避免重叠
                plt.tight_layout()

                # 保存图像
                file_name = f"picture/NIPS34_attention_scores{ii}_pair_batch_{batch_idx + 1}_pair_{pair_idx + 1}.png"
                plt.savefig(file_name)
                plt.close()  # 关闭图像，避免内存泄漏
                ii= ii+1
                print(f"Saved: {file_name}")

def attention(q, k, v, d_k, mask, dropout, zero_pad,ifTime):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, num_heads, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    
    
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    #根据ifTime画图
    visualize_all_attention_scores(scores,ifTime)
    
    print(f" the v {v.shape}")
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


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

class timeGap(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        input_size = num_rgap + num_sgap + num_pcount

        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
        rgap = self.rgap_eye[rgap].to(device)
        sgap = self.sgap_eye[sgap].to(device)
        pcount = self.pcount_eye[pcount].to(device)

        tg = torch.cat((rgap, sgap, pcount), -1)
        tg_emb = self.time_emb(tg)

        return tg_emb


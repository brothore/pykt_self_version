import os, sys
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from typing import List, Tuple, Dict, Any
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
from imblearn.over_sampling import SMOTE
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
from pykt.config import FOCAL_LOSS,freeze_epoch
from pykt.config import MULTI_LEVEL_TRAIN,SMOTE_METHOD
import pandas as pd
import time  # 导入时间模块
import pykt.models.glo
import traceback
from pykt.models.TCN_ABQR import BGRL
from itertools import zip_longest
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ii = 0
TCN_ABQR = 0
class PredictionWriter:
    def __init__(self, csv_path: str, concept_col: str = 'concepts', response_col: str = 'responses'):
        """
        初始化预测结果写入器
        
        Args:
            csv_path: CSV文件路径
            concept_col: 概念列名
            response_col: 响应列名
        """
        self.csv_path = csv_path
        self.concept_col = concept_col
        self.response_col = response_col
        self.df = pd.read_csv(csv_path)
        self.predictions = []
        
    def process_sequence(self, sequence_str: str) -> List[int]:
        """
        处理序列字符串，去掉填充值和无效值
        
        Args:
            sequence_str: 逗号分隔的序列字符串
            
        Returns:
            处理后的序列列表
        """
        if pd.isna(sequence_str):
            return []
        
        # 转换为整数列表
        sequence = [int(x) for x in str(sequence_str).split(',') if x.strip()]
        
        # 从后往前去掉所有的-1（填充值）
        while sequence and sequence[-1] == -1:
            sequence.pop()
            
        return sequence
    
    def process_concept_sequence(self, concept_str: str) -> List[int]:
        """
        处理概念序列，按照特定规则处理
        
        Args:
            concept_str: 概念序列字符串
            
        Returns:
            处理后的概念序列
        """
        sequence = self.process_sequence(concept_str)
        
        # 去掉最后一个有效值
        if sequence:
            sequence.pop()
            
        # 从后往前去掉所有0值
        while sequence and sequence[-1] == 0:
            sequence.pop()
            
        return sequence
    
    def find_matching_row(self, c_tensor: torch.Tensor) -> int:
        """
        根据概念张量找到CSV中对应的行
        
        Args:
            c_tensor: 概念张量 (batch_size=1的情况下是[1, seq_len])
            
        Returns:
            匹配的行索引，如果未找到返回-1
        """
        # 将张量转换为列表（去掉batch维度）
        c_list = c_tensor.squeeze(0).cpu().numpy().tolist()
        
        # 去掉序列中的填充值和0值（与模型输入预处理保持一致）
        c_processed = [x for x in c_list if x != 0 and x != -1]
        
        # 在DataFrame中查找匹配的行
        for idx, row in self.df.iterrows():
            concept_sequence = self.process_concept_sequence(row[self.concept_col])
            
            # 比较处理后的序列
            if concept_sequence == c_processed:
                print(f"找到匹配行！")
                return idx
                
        return -1
    
    def extract_predictions_from_model_forward(self, model, data_loader: DataLoader, device: torch.device):
        """
        从模型前向传播中提取预测结果并保存
        
        Args:
            model: 训练好的模型
            data_loader: 数据加载器（batch_size=1）
            device: 设备
        """
        model.eval()
        predictions_data = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                # 获取数据（根据你的代码结构调整）
                if isinstance(data, dict):
                    dcur = data
                    c = dcur["cseqs"].to(device)
                    r = dcur["rseqs"].to(device)
                    cshft = dcur["shft_cseqs"].to(device)
                else:
                    # 根据你的数据结构调整
                    c, r = data[0].to(device), data[1].to(device)
                    cshft = data[2].to(device) if len(data) > 2 else None
                
                # 模型前向传播
                if model.model_name == "dkt":
                    y, _ = model(c.long(), r.long())
                    if cshft is not None:
                        from torch.nn.functional import one_hot
                        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
                elif model.model_name == "TCN_ABQR":
                    # 根据你的TCN_ABQR模型调整
                    q = dcur["qseqs"].to(device)
                    qshft = dcur["shft_qseqs"].to(device)
                    y = model(q.long(), r.long(), c.long(), qshft.long(), cshft.long())
                else:
                    # 其他模型类型
                    y = model(c.long(), r.long())
                
                # 找到对应的CSV行
                row_idx = self.find_matching_row(c)
                
                if row_idx != -1:
                    # 保存预测结果
                    predictions_data.append({
                        'csv_row_idx': row_idx,
                        'concept_sequence': c.squeeze(0).cpu().numpy().tolist(),
                        'predictions': y.squeeze(0).cpu().numpy().tolist() if y.dim() > 1 else [y.item()],
                        'batch_idx': batch_idx
                    })
                else:
                    print(f"Warning: Could not find matching row for batch {batch_idx}")
                    
        self.predictions = predictions_data
        return predictions_data
    
    def write_predictions_to_csv(self, prediction_col: str = 'predictions', output_path: str = None):
        """
        将预测结果写回CSV文件
        
        Args:
            prediction_col: 预测结果列名
            output_path: 输出文件路径，如果为None则覆盖原文件
        """
        if not self.predictions:
            print("No predictions to write. Please run extract_predictions_from_model_forward first.")
            return
        
        # 创建副本以避免修改原始数据
        df_output = self.df.copy()
        
        # 初始化预测列
        if prediction_col not in df_output.columns:
            df_output[prediction_col] = None
        
        # 写入预测结果
        for pred_data in self.predictions:
            row_idx = pred_data['csv_row_idx']
            predictions = pred_data['predictions']
            
            # 将预测结果转换为字符串（如果是列表）
            if isinstance(predictions, list):
                pred_str = ','.join(map(str, predictions))
            else:
                pred_str = str(predictions)
                
            df_output.loc[row_idx, prediction_col] = pred_str
        
        # 保存文件
        output_file = output_path if output_path else self.csv_path
        df_output.to_csv(output_file, index=False)
        print(f"Predictions written to {output_file}")
        
        return df_output
    
    def create_detailed_report(self, report_path: str = None):
        """
        创建详细的预测报告
        
        Args:
            report_path: 报告文件路径
        """
        if not self.predictions:
            print("No predictions to report.")
            return
        
        report_data = []
        for pred_data in self.predictions:
            row_idx = pred_data['csv_row_idx']
            original_row = self.df.loc[row_idx]
            
            report_data.append({
                'csv_row_index': row_idx,
                'original_concept': original_row[self.concept_col],
                'original_response': original_row[self.response_col],
                'processed_concept': pred_data['concept_sequence'],
                'predictions': pred_data['predictions'],
                'batch_index': pred_data['batch_idx']
            })
        
        report_df = pd.DataFrame(report_data)
        
        if report_path:
            report_df.to_csv(report_path, index=False)
            print(f"Detailed report saved to {report_path}")
        
        return report_df

def run_prediction_pipeline(model, data_loader, csv_path: str, device: torch.device):
    """
    运行完整的预测管道
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        csv_path: CSV文件路径
        device: 设备
    """
    # 创建预测写入器
    writer = PredictionWriter(csv_path)
    
    # 提取预测结果
    print("Extracting predictions from model...")
    predictions = writer.extract_predictions_from_model_forward(model, data_loader, device)
    print(f"Extracted {len(predictions)} predictions")
    
    # 写入CSV
    print("Writing predictions to CSV...")
    output_df = writer.write_predictions_to_csv()
    
    # 创建详细报告
    report_path = csv_path.replace('.csv', '_prediction_report.csv')
    print("Creating detailed report...")
    report_df = writer.create_detailed_report(report_path)
    
    print("Pipeline completed successfully!")
    return output_df, report_df
def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name
    
    # 定义 Focal Loss 函数
    def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        y_pred: 模型输出的概率 [N]
        y_true: 真实标签 [N]
        alpha: 类别平衡权重
        gamma: 聚焦参数
        reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        
        # 计算 p_t = p if y=1 else 1-p
        p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
        
        # 计算调制因子 (1 - p_t)^gamma
        focal_term = (1 - p_t) ** gamma
        
        # 应用类别平衡权重 alpha_t
        alpha_t = torch.where(y_true == 1, alpha, 1 - alpha)
        loss = alpha_t * focal_term * bce_loss
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss
    
    # 根据全局变量选择损失函数
    if FOCAL_LOSS:
        loss_fn = focal_loss
        print("Using Focal Loss!")
    else:
        loss_fn = F.binary_cross_entropy
        print("Using Binary Cross-Entropy Loss!")

    if model_name in ["atdkt", "simplekt", "stablekt", "bakt_time", "sparsekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss1 = loss_fn(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
            
    elif model_name in ["rekt"]:
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double())

    elif model_name in ["rkt","dimkt","LSTM_Template","CTNKT","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double())
        
    elif model_name in ["TCN_ABQR"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double())
        
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        
        # 对两个损失都使用 Focal Loss
        loss = loss_fn(y_next.double(), r_next.double())
        loss_r = loss_fn(y_curr.double(), r_curr.double())
        
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
        
    elif model_name in ["Transformer_Template","akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","dtransformer","BERT","atakt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = loss_fn(y.double(), t.double()) + preloss[0]
        
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        if FOCAL_LOSS:
            # 对于 LPKT，使用自定义的 reduction
            criterion = lambda y_pred, y_true: focal_loss(y_pred, y_true, reduction='sum')
        else:
            criterion = nn.BCELoss(reduction='none')
            
        loss = criterion(y, t).sum() if not FOCAL_LOSS else criterion(y, t)
    
    return loss


def model_forward(model, data, writer: PredictionWriter,opt=None, rel=None,model_config={},data_label=0):
    # print(f"model_config5: {model_config}")
    global ii
    print(f"model forward{ii}")
    ii = ii+1
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "stablekt", "sparsekt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        

        y,_ = model(c.long(), r.long())
       
        # print(f"y shape{y.shape}")
        # print(f"cshft shape{cshft.long().shape}")

        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"y true shape{y.shape}")
        # print(f"y true {y}")
        if writer:
            row_idx = writer.find_matching_row(c)
            if row_idx != -1:
                writer.predictions.append({
                    'csv_row_idx': row_idx,
                    'concept_sequence': c.squeeze(0).detach().cpu().numpy().tolist(),    # ✅ 正确
            'predictions': y.squeeze(0).detach().cpu().numpy().tolist(),         # ✅ 正确
                    'batch_idx': len(writer.predictions)
                })
        ys.append(y) # first: yshft
        
    elif model_name in ["LSTM_Template"]:
        # print(f"cfloat:{c.float().shape}")
        y = model(c.long(), r.long())
        # print(f"y shape{y.shape}")
        # print(f"cshft shape{cshft.long().shape}")
        # print(f"model.num_c{model.num_c}")
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"selete shape{y.shape}")
        # print(f"y{y}")
        
        ys.append(y) # first: yshft
        
    elif model_name in ["CTNKT"]:
        # print(f"cfloat:{c.float().shape}")
        y = model(c.long(), r.long())
        # print(f"y shape{y.shape}")
        # print(f"cshft shape{cshft.long().shape}")
        # print(f"model.num_c{model.num_c}")
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"selete shape{y.shape}")
        # print(f"y{y}")
        
        ys.append(y) # first: yshft
    elif model_name in ["TCN_ABQR"]:
        # print(f"cc{cc.shape}")
        print(f"model_config: {model_config}")
        #扰动数据设计
        d = model_config['emb_size']
        print(f"d.shape: {d}")
        perturb_shape = (matrix.shape[0],d)
        step_size = 3e-2
        step_m = 3
        grad_clip = 15.0
        mm = 0.99
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
        pro_max = model_config['num_q']
        pro_embed = nn.Parameter(torch.ones((pro_max, d)))
        nn.init.xavier_uniform_(pro_embed)
        perturb.requires_grad_()
        print(f"perturb.shape: {perturb.shape}")
        pro_embed_trained = auto_load_BGRL(pro_embed,matrix,perturb,model_config,mm=mm)
        print(f"pro_embed_trained: {pro_embed_trained.shape}")
        # forward =  lambda perturb: model(q.long(), r.long(), c.long(), qshft.long(), cshft.long())
        y =  model(q.long(), r.long(), c.long(), qshft.long(), cshft.long(),pro_embed=pro_embed_trained)

        
        # print(f"bbb emb{emb_size}")
        
        # y, contrast_loss = forward(perturb)
        # y, contrast_loss = forward(perturb)
        
        
        # print("+"*100)
        # print(y.shape)
        # print("+"*100)
        # y = (y * one_hot(qshft.long(), model.num_q)).sum(-1)
        ys.append(y)
        # loss = cal_loss(model, ys, r, rshft, sm, preloss)
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
        # loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        # print("+"*100)
        # print(loss)
        # print("+"*100)
        # loss /= step_m
        # print("+"*100)
        # print(f"loss: {loss}")
        # print("+"*100)
        
        opt.zero_grad()

        # for _ in range(step_m - 1):
        #     loss.backward()
        #     # print(f"perturb.shape2: {perturb.shape}")
        #     # perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        #     # perturb.data = perturb_data.data
        #     # perturb.grad[:] = 0
        #     # perturb.grad.zero_()

        #     # predict, _, contrast_loss, c_reg_loss = forward(perturb)  # akt
        #     # y, contrast_loss = forward(perturb)
        #     pro_embed_trained = auto_load_BGRL(pro_embed,matrix,perturb,model_config,mm=mm)
        #     y =  model(q.long(), r.long(), c.long(), qshft.long(), cshft.long(),pro_embed=pro_embed_trained)
        #     ys.clear()
        #     # ys.append(y)
        #     ys.append(y[:,:])
        #     # loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        #     loss = cal_loss(model, ys, r, rshft, sm, preloss)
        #     loss /= step_m
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        # model.gcl.update_target_network(mm)  # EMA更新对抗学习的参数
        # auto_load_BGRL(pro_embed,matrix,perturb,model_config,mm=mm,update=True)
        # print("传播一次了")
        return loss
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["Transformer_Template","akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","BERT","atakt"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss
    
def save_results_to_file(results, folder_path, file_name="results.txt"):
    """保存结果到文件中"""
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"结果已保存到文件: {file_path}")
    
def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None,emb_sizess=128,model_config={},level1_loader=None,level2_loader=None,level3_loader=None):
    
    if MULTI_LEVEL_TRAIN == 1:
        # 使用多级训练
        return train_model_multilevel_custom(
    model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
    level1_loader,level2_loader, level3_loader, freeze_epoch, 
    test_loader, test_window_loader, save_model, data_config, fold, emb_sizess, model_config
)
    else:
        max_auc, best_epoch = 0, -1
        train_step = 0
        # print(f"model_config3: {model_config}")
        rel = None
        if model.model_name == "rkt":
            dpath = data_config["dpath"]
            dataset_name = dpath.split("/")[-1]
            tmp_folds = set(data_config["folds"]) - {fold}
            folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
            if dataset_name in ["algebra2005", "bridge2algebra2006"]:
                fname = "phi_dict" + folds_str + ".pkl"
                rel = pd.read_pickle(os.path.join(dpath, fname))
            else:
                fname = "phi_array" + folds_str + ".pkl" 
                rel = pd.read_pickle(os.path.join(dpath, fname))

        if model.model_name=='lpkt':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
        
        total_forward_time = 0.0  # 累计所有 forward 的时间
        total_epoch_time = 0.0    # 累计所有 epoch 的时间
        total_forward_count = 0    # 累计所有 forward 的次数

        for i in range(1, num_epochs + 1):
            epoch_start_time = time.time()  # 记录 epoch 开始时间
            loss_mean = []
            epoch_forward_time = 0.0       # 当前 epoch 的 forward 时间
            epoch_forward_count = 0         # 当前 epoch 的 forward 次数
            global ii
            ii = 0

                
                
            for data in train_loader:
                train_step += 1
                if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                    model.model.train()
                else:
                    model.train()
                
                # 记录 forward 开始时间
                forward_start = time.time()
                
                if model.model_name=='rkt':
                    loss = model_forward(model, data, rel)
                elif model.model_name=='TCN_ABQR':
                    print("entered_here!")
                    print(f"model_config4: {model_config}")
                    loss = model_forward(model, data,opt, rel,model_config=model_config)
                else:
                    writer = PredictionWriter("/root/autodl-tmp/pykt_self_version/data/assist2009/train_valid_sequences.csv")
                    loss = model_forward(model, data,writer)
                
                # 记录 forward 结束时间
                forward_end = time.time()
                forward_duration = forward_end - forward_start

                # 更新时间统计
                epoch_forward_time += forward_duration
                epoch_forward_count += 1
                total_forward_time += forward_duration
                total_forward_count += 1
                if model.model_name != "TCN_ABQR":
                    opt.zero_grad()
                    loss.backward()  # 计算梯度
                if model.model_name == "rkt":
                    clip_grad_norm_(model.parameters(), model.grad_clip)
                if model.model_name == "dtransformer":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()  # 更新模型参数

                loss_mean.append(loss.detach().cpu().numpy())
                if model.model_name == "gkt" and train_step % 10 == 0:
                    text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                    debug_print(text=text, fuc_name="train_model")
            
            if model.model_name == 'lpkt':
                scheduler.step()  # 更新学习率

            loss_mean = np.mean(loss_mean)
            
            if model.model_name == 'rkt':
                auc, acc = evaluate(model, valid_loader, model.model_name, rel)
            else:
                auc, acc = evaluate(model, valid_loader, model.model_name)

            if auc > max_auc + 1e-3:
                if save_model:
                    torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + "_model.ckpt"))
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                if not save_model:
                    if test_loader is not None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")
                        testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                    if test_window_loader is not None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                        window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
                validauc, validacc = auc, acc
            
            epoch_end_time = time.time()  # 记录 epoch 结束时间
            epoch_duration = epoch_end_time - epoch_start_time
            total_epoch_time += epoch_duration

            # 计算当前 epoch 的平均 forward 时间
            avg_forward_time_epoch = epoch_forward_time / epoch_forward_count if epoch_forward_count > 0 else 0.0

            print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
            print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")
            print(f"            Avg Forward Time this Epoch: {avg_forward_time_epoch:.6f} seconds")

            # 检查是否提前停止
            if i - best_epoch >= 10:
                break

        # 训练结束后，计算所有 forward 的平均时间和所有 epoch 的平均时间
        overall_avg_forward_time = total_forward_time / total_forward_count if total_forward_count > 0 else 0.0
        avg_epoch_time = total_epoch_time / i if i > 0 else 0.0

        print("\n训练结束！")
        print(f"每个 forward 的总体平均时间: {overall_avg_forward_time:.6f} 秒")
        print(f"每个 epoch 的平均时间: {avg_epoch_time:.2f} 秒")

        # 保存最终结果
        results = {
            "Best Epoch": best_epoch,
            "Best AUC": max_auc,
            "Valid AUC": validauc,
            "Valid Accuracy": validacc,
            "Test AUC": testauc,
            "Test Accuracy": testacc,
            "Window Test AUC": window_testauc,
            "Window Test Accuracy": window_testacc,
            "Overall Avg Forward Time": overall_avg_forward_time,
            "Avg Epoch Time": avg_epoch_time
        }
        save_results_to_file(results, ckpt_path)

        return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch


def auto_load_BGRL(pro_embed, Q_matrix, perb,model_config,mm,force_retrain=False):
    pro_embed = pro_embed.to(device)
    Q_matrix = Q_matrix.to(device)
    print("tranning embedding")
    dataset_path = model_config["dataset_path"]
    pro_embed_path = os.path.join(model_config['dataset_path'],"pro_embed.pt")
    d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2 = (model_config[k] for k in ['emb_size', 'p', 'drop_feat1', 'drop_feat2', 'drop_edge1', 'drop_edge2'])
    print(f"d.shape2: {d}")
    model = BGRL(d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2).to(device)
    # print(f"not os.path.exists(pro_embed_path): {not os.path.exists(pro_embed_path)}")
    if not os.path.exists(pro_embed_path) or force_retrain: #不存在，则训练
        print("不存在嵌入，开始训练")
        
        opt = torch.optim.Adam(model.parameters(),lr=1e-4)
        try:
            # pro_embed, contrast_loss = model(pro_embed, Q_matrix, perb)
            model.train()
            for epoch in range(100):
                _, contrast_loss = model(pro_embed, Q_matrix, perb)
                opt.zero_grad()
                contrast_loss.backward()
                # contrast_loss = 0.1 * contrast_loss
                opt.step()
                model.update_target_network(mm)
                if epoch %10 == 0:
                    print(f"Epoch {epoch:03d} | Loss: {contrast_loss.item():.4f}")
            torch.save(model.state_dict(),os.path.join(model_config['dataset_path'],"Graph.pt"))
            with torch.no_grad():
                pro_embed,_ = model(pro_embed, Q_matrix, perb)
                torch.save(pro_embed,pro_embed_path)
            print(f"模型训练完成保存到{pro_embed_path}")
        except Exception as e:
            # 打印完整错误堆栈
            print(f"\n❌ ABQR嵌入矩阵训练失败，详细错误信息：")
            print(traceback.format_exc())
            
            # 补充关键变量状态检查
            print("\n关键变量状态检查：")
            print(f"model_config 是否存在: {'dataset_path' in model_config}")
            if 'dataset_path' in model_config:
                print(f"保存路径是否存在: {os.path.exists(model_config['dataset_path'])}")
            print(f"pro_embed shape: {pro_embed.shape if isinstance(pro_embed, torch.Tensor) else 'Not Tensor'}")
            print(f"设备是否一致: model_device={next(model.parameters()).device}, perb_device={perb.device}")


    else:
        print("存在嵌入,直接读取")
        pass #存在，直接读取
    
    return torch.load(pro_embed_path)

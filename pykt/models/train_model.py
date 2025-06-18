import os, sys
import torch
import torch.nn as nn
from collections import Counter
from imblearn.over_sampling import SMOTE
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
from pykt.config import FOCAL_LOSS
from pykt.config import MULTI_LEVEL_TRAIN
import pandas as pd
import time  # 导入时间模块
import pykt.models.glo
import traceback
from pykt.models.TCN_ABQR import BGRL
from itertools import zip_longest
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ii = 0
TCN_ABQR = 0
if TCN_ABQR:
    pre_load_gcn = "/share/disk/hzb/dataset/assistment2009/ques_skill_gcn_adj.pt"
    matrix = torch.load(pre_load_gcn).to(device)
    if not matrix.is_sparse:
        matrix = matrix.to_sparse()
def save_results_to_file(results, folder_path, file_name="results.txt"):
    """保存结果到文件中"""
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"结果已保存到文件: {file_path}")


import torch
import torch.nn as nn
import torch.nn.functional as F
def collect_embeddings_from_loader(model, data_loader, device):
    """从数据加载器中收集嵌入向量"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data in data_loader:
            # 解析数据
            dcur = data
            q, c, r = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device)
            qshft, cshft, rshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device)
            m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)
            
            # DKT前向传播获取嵌入
            if model.model_name == "dkt":
                y, xemb = model(c.long(), r.long())
                
                # 收集有效的嵌入向量和对应的标签
                valid_mask = sm.bool()  # 使用shifted mask来确定有效位置
                
                # 展平处理
                xemb_flat = xemb.view(-1, xemb.size(-1))  # [batch_size * seq_len, emb_size]
                rshft_flat = rshft.view(-1)  # [batch_size * seq_len]
                valid_mask_flat = valid_mask.view(-1)  # [batch_size * seq_len]
                
                # 只取有效位置的数据
                valid_embeddings = xemb_flat[valid_mask_flat]
                valid_labels = rshft_flat[valid_mask_flat]
                
                all_embeddings.append(valid_embeddings.cpu().numpy())
                all_labels.append(valid_labels.cpu().numpy())
    
    # 合并所有批次的数据
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)
    
    return all_embeddings, all_labels


def create_smote_dataloader(embeddings1, labels1, embeddings3, labels3, batch_size=32, device='cuda'):
    """使用SMOTE创建新的数据加载器"""
    # 合并level1和level3的数据
    combined_embeddings = np.vstack([embeddings1, embeddings3])
    combined_labels = np.hstack([labels1, labels3])
    
    print(f"SMOTE前数据分布: {Counter(combined_labels)}")
    
    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42, k_neighbors=min(5, len(combined_embeddings) - 1))
    
    try:
        resampled_embeddings, resampled_labels = smote.fit_resample(combined_embeddings, combined_labels)
        print(f"SMOTE后数据分布: {Counter(resampled_labels)}")
        
        # 转换为torch张量
        embeddings_tensor = torch.FloatTensor(resampled_embeddings).to(device)
        labels_tensor = torch.LongTensor(resampled_labels).to(device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(embeddings_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    except Exception as e:
        print(f"SMOTE失败: {e}")
        # 如果SMOTE失败，返回原始数据
        embeddings_tensor = torch.FloatTensor(combined_embeddings).to(device)
        labels_tensor = torch.LongTensor(combined_labels).to(device)
        dataset = TensorDataset(embeddings_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader


def model_forward_with_smote_data(model, smote_data, device):
    """处理SMOTE生成的数据进行训练"""
    embeddings, labels = smote_data
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # 从嵌入直接到LSTM层（跳过embedding层）
    h, _ = model.lstm_layer(embeddings.unsqueeze(1))  # 添加序列维度
    h = model.dropout_layer(h)
    y = model.out_layer(h)
    y = torch.sigmoid(y).squeeze(1)  # 移除序列维度
    
    # 计算损失
    loss = torch.nn.functional.binary_cross_entropy(y, labels.float())
    
    return loss


def train_model_multilevel(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, 
                          level1_loader=None, level3_loader=None, freeze_epoch=10,
                          test_loader=None, test_window_loader=None, save_model=False, 
                          data_config=None, fold=None, emb_sizess=128, model_config={}):
    """改进的多级训练函数"""
    max_auc, best_epoch = 0, -1
    train_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查是否启用多级训练
    if MULTI_LEVEL_TRAIN != 1:
        print("使用原始训练方式")
        return train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                          test_loader, test_window_loader, save_model, data_config, fold,
                          emb_sizess, model_config, level1_loader, level3_loader)
    
    print("使用多级训练方式")
    
    # 检查必要的数据加载器
    if level1_loader is None or level3_loader is None:
        print("警告: level1_loader 或 level3_loader 为空，使用原始训练方式")
        return train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                          test_loader, test_window_loader, save_model, data_config, fold,
                          emb_sizess, model_config, level1_loader, level3_loader)
    
    smote_dataloader = None
    model_frozen = False
    
    for i in range(1, num_epochs + 1):
        loss_mean = []
        
        # 第一阶段：正常训练到指定epoch
        if i <= freeze_epoch:
            model.train()
            for data in train_loader:
                train_step += 1
                
                # DKT前向传播
                loss = model_forward(model, data)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                loss_mean.append(loss.detach().cpu().numpy())
        
        # 在指定epoch后进行SMOTE数据增强
        elif i == freeze_epoch + 1:
            print(f"在第{i}个epoch进行SMOTE数据增强...")
            
            # 冻结模型权重，收集嵌入
            print("冻结模型权重，收集嵌入向量...")
            for param in model.parameters():
                param.requires_grad = False
            model_frozen = True
            
            # 从level2和level3数据中收集嵌入
            print("从level1数据收集嵌入...")
            embeddings2, labels2 = collect_embeddings_from_loader(model, level1_loader, device)
            
            print("从level3数据收集嵌入...")
            embeddings3, labels3 = collect_embeddings_from_loader(model, level3_loader, device)
            
            # 使用SMOTE创建新的数据加载器
            print("使用SMOTE创建增强数据...")
            smote_dataloader = create_smote_dataloader(
                embeddings2, labels2, embeddings3, labels3, 
                batch_size=32, device=device
            )
            
            # 解冻模型权重
            print("解冻模型权重...")
            for param in model.parameters():
                param.requires_grad = True
            model_frozen = False
            
            # 继续训练
            model.train()
            
            # 在原始数据上训练
            for data in train_loader:
                train_step += 1
                loss = model_forward(model, data)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                loss_mean.append(loss.detach().cpu().numpy())
            
            # 在SMOTE数据上训练
            if smote_dataloader is not None:
                for smote_data in smote_dataloader:
                    train_step += 1
                    loss = model_forward_with_smote_data(model, smote_data, device)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    loss_mean.append(loss.detach().cpu().numpy())
        
        # 第三阶段：继续在原始数据和SMOTE数据上训练
        else:
            model.train()
            
            # 在原始数据上训练
            for data in train_loader:
                train_step += 1
                loss = model_forward(model, data)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                loss_mean.append(loss.detach().cpu().numpy())
            
            # 在SMOTE数据上训练（如果存在）
            if smote_dataloader is not None:
                for smote_data in smote_dataloader:
                    train_step += 1
                    loss = model_forward_with_smote_data(model, smote_data, device)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    loss_mean.append(loss.detach().cpu().numpy())
        
        # 计算平均损失
        loss_mean = np.mean(loss_mean)
        
        # 验证模型
        auc, acc = evaluate(model, valid_loader, model.model_name)
        
        # 保存最佳模型
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
        
        # 打印训练信息
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")
        
        if i == freeze_epoch + 1:
            print(f"            已完成SMOTE数据增强，继续训练...")
        
        # 早停机制
        if i - best_epoch >= 10:
            break
    
    # 确保模型权重被解冻
    if model_frozen:
        for param in model.parameters():
            param.requires_grad = True
    
    print("多级训练完成！")
    
    # 保存最终结果
    results = {
        "Training Method": "Multi-Level with SMOTE",
        "Freeze Epoch": freeze_epoch,
        "Best Epoch": best_epoch,
        "Best AUC": max_auc,
        "Valid AUC": validauc,
        "Valid Accuracy": validacc,
        "Test AUC": testauc,
        "Test Accuracy": testacc,
        "Window Test AUC": window_testauc,
        "Window Test Accuracy": window_testacc
    }
    save_results_to_file(results, ckpt_path)
    
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch

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
        
    elif model_name in ["Transformer_Template","akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","dtransformer"]:
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


def model_forward(model, data, opt=None, rel=None,model_config={},data_label=0):
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
    elif model_name in ["Transformer_Template","akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:               
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
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None,emb_sizess=128,model_config={},level1_loader=None,level3_loader=None):
    freeze_epoch = 10
    if MULTI_LEVEL_TRAIN == 1:
        # 使用多级训练
        return train_model_multilevel(
            model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
            level1_loader, level3_loader, freeze_epoch, 
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
                    loss = model_forward(model, data)
                
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

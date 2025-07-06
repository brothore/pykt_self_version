#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori

class KTDataset_uid(Dataset):
    """
    知识跟踪(Knowledge Tracing)数据集类
    
    用于初始化以下类型的数据集（除dkt_forget模型外的所有模型）:
        - 训练数据、验证数据
        - 通用测试数据（概念级别评估）
        - 真实教育场景测试数据（问题级别评估）
    
    Args:
        file_path (str): 训练/验证/测试文件路径
        input_type (list[str]): 数据集的输入类型，取值为 ["questions", "concepts"] 中的一个或多个
        folds (set(int)): 用于生成数据集的折数，测试数据使用-1
        qtest (bool, optional): 是否为问题级别评估。默认为False（概念级别评估）
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        
        # 处理fold信息，生成文件名后缀
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        
        # 根据是否为问题级别测试确定预处理文件名
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        # 检查预处理文件是否存在，如果不存在则进行数据预处理
        if not os.path.exists(processed_data):
            print(f"开始预处理 {file_path} fold: {folds_str}...")
            if self.qtest:
                # 问题级别测试需要额外的测试数据
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                # 常规训练/验证数据
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            # 保存预处理后的数据
            pd.to_pickle(save_data, processed_data)
        else:
            # 从预处理文件加载数据
            print(f"从预处理文件读取数据: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                # 可选：限制数据量用于调试（当前注释掉）
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        
        # 打印数据集基本信息
        print(f"文件路径: {file_path}, "
              f"问题序列长度: {len(self.dori['qseqs']) if 'qseqs' in self.dori else 0}, "
              f"概念序列长度: {len(self.dori['cseqs']) if 'cseqs' in self.dori else 0}, "
              f"回答序列长度: {len(self.dori['rseqs'])}, "
              f"用户ID数量: {len(self.dori['uids'])}")

    def __len__(self):
        """
        返回数据集长度
        
        Returns:
            int: 数据集的长度（学习序列的数量）
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        
        Args:
            index (int): 要获取的数据索引
            
        Returns:
            如果不是问题级别测试(qtest=False):
                dcur (dict): 包含以下键值对的字典
                    - q_seqs (torch.tensor): 第0~seqlen-2次交互的问题ID序列
                    - c_seqs (torch.tensor): 第0~seqlen-2次交互的知识概念ID序列  
                    - r_seqs (torch.tensor): 第0~seqlen-2次交互的回答序列
                    - shft_qseqs (torch.tensor): 第1~seqlen-1次交互的问题ID序列
                    - shft_cseqs (torch.tensor): 第1~seqlen-1次交互的知识概念ID序列
                    - shft_rseqs (torch.tensor): 第1~seqlen-1次交互的回答序列
                    - masks (torch.tensor): 掩码序列，形状为seqlen-1
                    - smasks (torch.tensor): 选择掩码，0表示不选择，1表示选择，形状为seqlen-1
                    - uids (torch.tensor): 用户ID
                    - tseqs/utseqs: 时间戳相关序列（如果存在）
            
            如果是问题级别测试(qtest=True):
                (dcur, dqtest): 元组包含
                    - dcur: 如上所述的字典
                    - dqtest (dict): 问题级别评估所需的额外数据
        """
        dcur = dict()
        # 获取当前序列的掩码
        mseqs = self.dori["masks"][index]
        
        # 处理所有序列数据
        for key in self.dori:
            if key in ["masks", "smasks", "uids"]:  # 跳过特殊处理的键
                continue
            if len(self.dori[key]) == 0:
                # 空序列的处理
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            
            # 应用掩码并创建当前序列和移位序列
            # 当前序列：去掉最后一个元素并应用掩码
            seqs = self.dori[key][index][:-1] * mseqs
            # 移位序列：去掉第一个元素并应用掩码（用于预测下一步）
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        
        # 添加掩码信息
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        
        # 添加用户ID（uid在每个序列中是固定的，不需要移位）
        dcur["uids"] = self.dori["uids"][index]
        
        # 根据测试类型返回不同的数据
        if not self.qtest:
            return dcur
        else:
            # 问题级别测试需要额外的测试数据
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        从CSV文件加载和预处理数据
        
        Args:
            sequence_path (str): 序列文件路径
            folds (list[int]): 要加载的折数列表
            pad_val (int, optional): 填充值。默认为-1
            
        Returns: 
            如果不是问题级别测试:
                dori (dict): 包含以下键的字典
                    - qseqs: 问题ID序列列表
                    - cseqs: 概念ID序列列表  
                    - rseqs: 回答序列列表
                    - uids: 用户ID列表
                    - tseqs: 时间戳序列列表（如果存在）
                    - utseqs: 使用时间序列列表（如果存在）
                    - masks: 计算的掩码序列
                    - smasks: 选择掩码序列
                    
            如果是问题级别测试:
                (dori, dqtest): 元组包含
                    - dori: 如上所述
                    - dqtest: 问题级别评估的额外数据
        """
        # 初始化数据容器
        dori = {
            "qseqs": [],    # 问题序列
            "cseqs": [],    # 概念序列
            "rseqs": [],    # 回答序列
            "tseqs": [],    # 时间戳序列
            "utseqs": [],   # 使用时间序列
            "smasks": [],   # 选择掩码
            "uids": []      # 用户ID
        }

        # 读取CSV文件并过滤指定的fold
        df = pd.read_csv(sequence_path)  # 可选：添加 [0:1000] 用于调试
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        
        # 问题级别测试的额外数据容器
        dqtest = {"qidxs": [], "rests": [], "orirow": []}
        
        # 逐行处理数据
        for i, row in df.iterrows():
            # 根据输入类型加载相应的序列数据
            if "concepts" in self.input_type:
                # 解析概念ID序列
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                # 解析问题ID序列
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            
            # 解析时间相关序列（如果存在）
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
            
            # 解析回答序列和选择掩码
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            
            # 添加用户ID（每行一个值，不是序列）
            dori["uids"].append(int(row["uid"]))

            # 统计交互次数
            interaction_num += dori["smasks"][-1].count(1)

            # 如果是问题级别测试，加载额外数据
            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        
        # 将数据转换为PyTorch张量
        for key in dori:
            if key == "uids":
                # 用户ID是整数列表，转换为LongTensor
                dori[key] = LongTensor(dori[key])
            elif key not in ["rseqs"]:  # 大部分序列数据转换为LongTensor
                if len(dori[key]) > 0:  # 只转换非空列表
                    dori[key] = LongTensor(dori[key])
            else:  # 回答序列转换为FloatTensor
                dori[key] = FloatTensor(dori[key])

        # 计算掩码序列：前一个和后一个位置都不是填充值的位置为True
        if "cseqs" in dori and len(dori["cseqs"]) > 0:
            mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        else:
            # 如果没有概念序列，使用问题序列计算掩码
            mask_seqs = (dori["qseqs"][:,:-1] != pad_val) * (dori["qseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        # 处理选择掩码：移除第一个位置，保留1到最后的位置
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        
        print(f"交互总数: {interaction_num}")

        # 处理问题级别测试的额外数据
        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]  # 移除第一个位置
            return dori, dqtest
        
        return dori
    
class ABQRDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori

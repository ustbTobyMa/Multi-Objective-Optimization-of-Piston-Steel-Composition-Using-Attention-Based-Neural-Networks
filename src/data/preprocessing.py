# -*- coding: utf-8 -*-
"""
数据预处理模块
处理活塞钢成分-性能数据，用于模型训练
参考论文Section 2.1
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans  # 用于缺失值填充
# from sklearn.ensemble import IsolationForest  # 异常值检测


class DataPreprocessor:
    """数据预处理类，处理成分-工艺-性能数据"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # 鲁棒标准化
        self.feature_names = None
        # self.kmeans = None  # 用于缺失值填充的聚类
        
    def load_data(self, data_path):
        """加载原始数据"""
        # 读取CSV文件
        data = pd.read_csv(data_path, encoding='utf-8')
        print(f"数据加载完成，共 {len(data)} 条记录")
        return data
    
    def clean_data(self, data):
        """
        数据清洗：处理缺失值和异常值
        缺失值用k-means聚类后的中位数填充（k=50）
        异常值用IsolationForest检测（contamination=0.05）
        """
        cleaned_data = data.copy()
        
        # 缺失值处理 - 先用中位数填充，后续可以改进
        missing_rate = cleaned_data.isnull().sum() / len(cleaned_data)
        print(f"缺失值比例: {missing_rate[missing_rate > 0]}")
        
        # 简单处理：用中位数填充（实际论文中用的是k-means聚类后的中位数）
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
        
        # 异常值检测（简化版，实际用IsolationForest）
        # 这里只做简单的3-sigma规则
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_val = cleaned_data[col].mean()
            std_val = cleaned_data[col].std()
            # 移除3倍标准差外的值
            cleaned_data = cleaned_data[
                (cleaned_data[col] >= mean_val - 3*std_val) & 
                (cleaned_data[col] <= mean_val + 3*std_val)
            ]
        
        print(f"清洗后数据量: {len(cleaned_data)} 条")
        return cleaned_data
    
    def prepare_features(self, data):
        """
        准备特征矩阵
        成分特征：12个合金元素（C, Cr, Mo, Mn, Si, Ni, P, S, V, Ti, Al, Cu）
        工艺特征：4个热处理参数（淬火温度、回火温度、冷却速率、保温时间）
        """
        # 成分元素
        chengfen = ['C', 'Cr', 'Mo', 'Mn', 'Si', 'Ni', 'P', 'S', 'V', 'Ti', 'Al', 'Cu']
        
        # 工艺参数
        gongyi = ['Quench_Temp', 'Temper_Temp', 'Cooling_Rate', 'Holding_Time']
        
        # 合并特征
        feature_cols = chengfen + gongyi
        X = data[feature_cols].values
        
        # 目标性能（6个）
        mubiao = ['Yield_Strength', 'Tensile_Strength', 'Elongation', 
                 'Impact_Toughness', 'Thermal_Conductivity', 'Thermal_Expansion']
        y = data[mubiao].values
        
        self.feature_names = feature_cols
        return X, y, feature_cols
    
    def scale_features(self, X_train, X_val=None, X_test=None):
        """特征标准化，用RobustScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """
        数据划分：训练集/验证集/测试集
        比例：70/15/15
        """
        # 先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 再从剩余数据中分出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def process_pipeline(self, data_path):
        """完整的数据预处理流程"""
        # 加载数据
        raw_data = self.load_data(data_path)
        
        # 清洗数据
        cleaned_data = self.clean_data(raw_data)
        
        # 准备特征
        X, y, feature_names = self.prepare_features(cleaned_data)
        
        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 特征标准化
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': self.scaler
        }

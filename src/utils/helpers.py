# -*- coding: utf-8 -*-
"""
工具函数
用于模型评估、目标函数计算等
"""

import numpy as np
from typing import Dict, List, Tuple


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算决定系数R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对百分比误差MAPE"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def normalize_objectives(objectives: np.ndarray, 
                        lower_bounds: np.ndarray,
                        upper_bounds: np.ndarray) -> np.ndarray:
    """归一化目标函数值到[0, 1]"""
    normalized = (objectives - lower_bounds) / (upper_bounds - lower_bounds)
    return np.clip(normalized, 0, 1)


def composite_performance_score(objectives: np.ndarray, 
                               weights: np.ndarray) -> np.ndarray:
    """
    计算综合性能分数
    F = w₁·S + w₂·D + w₃·H
    """
    return np.dot(objectives, weights)


def filter_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """筛选Pareto最优解"""
    n_solutions = objectives.shape[0]
    is_pareto = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i != j:
                # 检查j是否支配i
                if np.all(objectives[j] <= objectives[i]) and \
                   np.any(objectives[j] < objectives[i]):
                    is_pareto[i] = False
                    break
    
    return is_pareto


def select_diverse_solutions(solutions: List[Dict], 
                            n_select: int,
                            distance_metric: str = 'euclidean') -> List[Dict]:
    """从Pareto前沿中选择多样性解"""
    # 实现细节省略
    return solutions[:n_select]

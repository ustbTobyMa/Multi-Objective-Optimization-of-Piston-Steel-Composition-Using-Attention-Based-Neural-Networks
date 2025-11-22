# -*- coding: utf-8 -*-
"""
可视化模块
生成论文中的各种图表
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Plotter:
    """绘图工具类"""
    
    def __init__(self, style='seaborn-v0_8', figsize=(10, 6)):
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_correlation_matrix(self, data, save_path=None):
        """绘制元素相关性矩阵（论文Fig. 2b）"""
        # 选择成分列
        elements = ['C', 'Cr', 'Mo', 'Mn', 'Si', 'Ni', 'V', 'Ti', 'Al', 'Cu']
        corr_data = data[elements].corr()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Correlation Matrix of Alloying Elements')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parity_plots(self, y_true, y_pred, property_names, save_path=None):
        """绘制预测值vs真实值对比图（论文Fig. 4）"""
        n_properties = len(property_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, prop_name in enumerate(property_names):
            ax = axes[idx]
            true_vals = y_true[:, idx]
            pred_vals = y_pred[:, idx]
            
            # 散点图
            ax.scatter(true_vals, pred_vals, alpha=0.6, s=50)
            
            # 1:1线
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='1:1 line')
            
            # 计算R²
            r2 = np.corrcoef(true_vals, pred_vals)[0, 1] ** 2
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            ax.set_title(prop_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_heatmap(self, attention_weights, feature_names, 
                               property_names, save_path=None):
        """绘制注意力权重热图（论文Fig. 5a）"""
        # 平均注意力权重（跨样本和头）
        avg_attention = np.mean(attention_weights, axis=(0, 1))  # 占位符
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(avg_attention, xticklabels=property_names, 
                   yticklabels=feature_names, annot=True, 
                   fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Attention Weight Heatmap')
        ax.set_xlabel('Properties')
        ax.set_ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pareto_front(self, objectives, save_path=None):
        """绘制Pareto前沿（论文Fig. 6）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 性能vs成本（Fig. 6a）
        ax1 = axes[0]
        # 假设objectives是[yield_str, tensile_str, impact, cost]
        performance = objectives[:, 0] + objectives[:, 1] + objectives[:, 2]  # 综合性能
        cost = objectives[:, 3]
        
        ax1.scatter(cost, performance, alpha=0.6, s=50, c='red', label='Pareto solutions')
        ax1.axvline(x=1.3, color='k', linestyle='--', label='Cost cap (1.3)')
        ax1.set_xlabel('Cost Index')
        ax1.set_ylabel('Composite Performance')
        ax1.set_title('Performance-Cost Pareto Front')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 强度vs韧性权衡（Fig. 6b）
        ax2 = axes[1]
        strength = objectives[:, 0] + objectives[:, 1]
        ductility = objectives[:, 2]  # 用冲击韧性作为代理
        
        scatter = ax2.scatter(strength, ductility, c=objectives[:, 3], 
                            cmap='viridis', alpha=0.6, s=50)
        ax2.set_xlabel('Strength (Yield + Tensile)')
        ax2.set_ylabel('Impact Toughness')
        ax2.set_title('Strength-Ductility Trade-off')
        plt.colorbar(scatter, ax=ax2, label='Cost Index')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence(self, history, save_path=None):
        """绘制优化收敛曲线（论文Fig. 6c）"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        generations = range(len(history['best']))
        ax.plot(generations, history['best'], label='Best fitness', linewidth=2)
        ax.plot(generations, history['average'], label='Average fitness', linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('NSGA-II Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_property_comparison(self, ai_steels, conventional_steels, 
                                save_path=None):
        """绘制性能对比图（论文Fig. 9）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 强度对比（Fig. 9a）
        ax1 = axes[0]
        steels = ['AI-QT', 'AI-NQT', 'Conv-QT', 'Conv-NQT']
        yield_strength = [
            ai_steels['QT']['yield'],
            ai_steels['NQT']['yield'],
            conventional_steels['QT']['yield'],
            conventional_steels['NQT']['yield']
        ]
        tensile_strength = [
            ai_steels['QT']['tensile'],
            ai_steels['NQT']['tensile'],
            conventional_steels['QT']['tensile'],
            conventional_steels['NQT']['tensile']
        ]
        
        x = np.arange(len(steels))
        width = 0.35
        ax1.bar(x - width/2, yield_strength, width, label='Yield Strength', alpha=0.8)
        ax1.bar(x + width/2, tensile_strength, width, label='Tensile Strength', alpha=0.8)
        ax1.set_xlabel('Steel Type')
        ax1.set_ylabel('Strength (MPa)')
        ax1.set_title('Strength Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(steels)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 塑性和韧性（Fig. 9b）
        ax2 = axes[1]
        elongation = [
            ai_steels['QT']['elongation'],
            ai_steels['NQT']['elongation'],
            conventional_steels['QT']['elongation'],
            conventional_steels['NQT']['elongation']
        ]
        impact = [
            ai_steels['QT']['impact'],
            ai_steels['NQT']['impact'],
            conventional_steels['QT']['impact'],
            conventional_steels['NQT']['impact']
        ]
        
        ax2.bar(x - width/2, elongation, width, label='Elongation (%)', alpha=0.8)
        ax2.bar(x + width/2, impact, width, label='Impact Toughness (J)', alpha=0.8)
        ax2.set_xlabel('Steel Type')
        ax2.set_ylabel('Property Value')
        ax2.set_title('Ductility and Toughness Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(steels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

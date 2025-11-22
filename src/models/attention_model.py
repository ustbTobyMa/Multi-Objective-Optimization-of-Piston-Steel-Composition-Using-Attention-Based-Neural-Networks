# -*- coding: utf-8 -*-
"""
基于注意力机制的多任务神经网络
用于预测钢的性能参数
参考论文Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """多头注意力机制，用于学习特征重要性"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        前向传播
        实现scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        """
        residual = x
        x = self.layer_norm(x)
        
        batch_size, seq_len, embed_dim = x.size()
        
        # 投影到Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到V
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        output = output + residual  # 残差连接
        
        return output, attn_weights


class AttentionBasedModel(nn.Module):
    """
    基于注意力机制的多任务预测模型
    架构：输入嵌入 -> 多头注意力 -> MLP -> 任务特定输出头
    """
    
    def __init__(self, input_dim, embed_dim=128, num_heads=8, 
                 hidden_dims=[256, 128, 64], num_tasks=6, dropout=0.2):
        super().__init__()
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 多头注意力
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 前馈网络
        layers = []
        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 任务特定的输出头（6个性能指标）
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1) for _ in range(num_tasks)
        ])
        
    def forward(self, x, return_attention=False):
        """前向传播"""
        # 嵌入输入
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加序列维度用于attention
        
        # 应用注意力
        x, attn_weights = self.attention(x)
        x = x.squeeze(1)  # 移除序列维度
        
        # 前馈网络
        x = self.mlp(x)
        
        # 任务特定预测
        predictions = torch.cat([head(x) for head in self.task_heads], dim=1)
        
        if return_attention:
            return {
                'predictions': predictions,
                'attention_weights': attn_weights
            }
        return {'predictions': predictions}
    
    def get_attention_weights(self, x):
        """提取注意力权重用于可解释性分析"""
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate=1e-3, weight_decay=1e-5):
        """设置优化器和学习率调度器"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        # 学习率调度器可以后续添加
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(...)
        
    def train_epoch(self, train_loader, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            outputs = self.model(batch_x)
            predictions = outputs['predictions']
            
            # 计算损失（多任务加权MSE）
            loss = criterion(predictions, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                predictions = outputs['predictions']
                
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # 计算评估指标（R², RMSE, MAPE等）
        # 这里简化处理，实际需要详细计算
        
        return {
            'loss': total_loss / len(val_loader),
            'predictions': np.concatenate(all_preds, axis=0),
            'targets': np.concatenate(all_targets, axis=0)
        }

# 基于注意力神经网络的活塞钢成分多目标优化

本项目实现了商用车辆活塞钢成分的多目标优化框架，结合可解释的注意力神经网络和约束遗传算法。

## 概述

本研究提出了一种可解释的、基于注意力机制的深度学习框架，结合NSGA-II多目标优化算法，在可焊性和成本约束下设计可制造的钢成分。该方法能够：

- **可解释的性能预测**：多头注意力机制提供成分-性能关系的特征级可解释性
- **约束多目标优化**：NSGA-II算法在现实工业约束下搜索最优成分
- **实验验证**：通过实验室合成和表征验证框架有效性

## 方法

工作流程包括四个主要阶段：

1. **数据预处理** (Section 2.1)：成分-工艺-性能数据集整理和预处理
2. **模型训练** (Section 2.2)：基于注意力的多任务神经网络进行性能预测
3. **优化** (Section 2.3)：约束NSGA-II进行多目标成分设计
4. **可视化** (Section 3)：结果分析和解释

## 项目结构

```
.
├── src/
│   ├── data/
│   │   └── preprocessing.py          # 数据加载和预处理
│   ├── models/
│   │   └── attention_model.py        # 注意力神经网络
│   ├── optimization/
│   │   └── nsga2_optimizer.py        # NSGA-II多目标优化器
│   ├── visualization/
│   │   └── plotting.py               # 可视化工具
│   └── utils/
│       └── helpers.py                 # 工具函数
├── data/
│   ├── raw/                          # 原始数据文件
│   └── processed/                    # 处理后的数据
├── results/
│   ├── figures/                      # 生成的图表
│   ├── optimization/                 # 优化结果
│   └── predictions/                  # 模型预测
├── main.py                           # 主执行脚本
├── requirements.txt                  # Python依赖
└── README.md                         # 本文件
```

## 主要特性

### 1. 注意力模型架构

模型使用多头注意力捕获上下文相关的元素-性能相互作用：

- **输入**：12个合金元素 + 4个热处理参数
- **架构**：嵌入层 → 多头注意力 → MLP → 任务特定输出头
- **输出**：6个目标性能的预测（强度、塑性、韧性、热性能等）
- **可解释性**：注意力权重揭示每个性能的特征重要性

### 2. 约束多目标优化

NSGA-II优化器，带工业约束：

- **目标**：最大化强度、韧性、高温性能；最小化成本
- **约束**：
  - 碳当量(CEV) ≤ 0.60（可焊性）
  - 相对成本指数 ≤ 1.30
  - 成分边界和总合金含量上限
- **输出**：Pareto最优解集，带权衡分析

### 3. 结果可视化

全面的可视化套件：

- 数据库分析（相关性矩阵、性能分布）
- 模型性能（对比图、注意力热图）
- 优化结果（Pareto前沿、收敛曲线）
- 性能对比（AI设计钢 vs. 传统钢）

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/ustbTobyMa/Multi-Objective-Optimization-of-Piston-Steel-Composition-Using-Attention-Based-Neural-Networks.git
cd Multi-Objective-Optimization-of-Piston-Steel-Composition-Using-Attention-Based-Neural-Networks
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用

### 基本流程

运行完整流程：

```bash
python main.py
```

### 单独使用各模块

**数据预处理：**
```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
processed_data = preprocessor.process_pipeline("data/raw/steel_data.csv")
```

**模型训练：**
```python
from src.models.attention_model import AttentionBasedModel, ModelTrainer

model = AttentionBasedModel(input_dim=16, num_tasks=6)
trainer = ModelTrainer(model)
trainer.setup_training()
# 训练循环...
```

**优化：**
```python
from src.optimization.nsga2_optimizer import NSGA2Optimizer, ConstraintHandler

constraint_handler = ConstraintHandler(cev_limit=0.60, cost_limit=1.30)
optimizer = NSGA2Optimizer(model, constraint_handler)
results = optimizer.optimize()
```

## 结果

框架成功实现：

- 所有目标性能的R² > 0.95
- 在约束下生成36个Pareto最优解
- 验证AI设计钢显示18-25%强度提升
- 在600°C下表现出改善的抗氧化性

## 引用

如果使用本代码，请引用：

```bibtex
@article{ma2024multi,
  title={Multi-Objective Optimization of Commercial Vehicle Piston Steel Composition Using Attention-Based Neural Networks},
  author={Ma, Weitao and Rao, Yanjun and Zhang, Zheyue and others},
  journal={[期刊名称]},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 联系方式

如有问题或合作意向，请联系：
- 通讯作者：Renbo Song (songrb@mater.ustb.edu.cn)

## 致谢

本研究得到以下项目支持：
- 国家自然科学基金 (No. 52074033)
- 河钢集团重点研发项目 (HG2023242)

## 说明

**重要提示**：本仓库提供了方法和流程的演示。完整实现细节请参考完整论文。部分实现细节已简化或省略，以突出整体框架和可解释性。

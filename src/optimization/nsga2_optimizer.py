# -*- coding: utf-8 -*-
"""
NSGA-II多目标优化算法
用于活塞钢成分优化设计
参考论文Section 2.3
"""

import numpy as np
from typing import List, Tuple, Dict, Callable


class ConstraintHandler:
    """
    约束处理类
    处理成分边界、碳当量(CEV)、成本等约束
    """
    
    def __init__(self, cev_limit=0.60, cost_limit=1.30):
        self.cev_limit = cev_limit  # 碳当量上限（可焊性约束）
        self.cost_limit = cost_limit  # 成本指数上限（相对成本≤1.3）
        
        # 成分边界（wt%）
        self.bounds = {
            'C': (0.15, 0.60),
            'Cr': (0.50, 2.00),
            'Mo': (0.10, 0.50),
            'V': (0.02, 0.30),
            'Mn': (0.50, 1.50),
            'Si': (0.20, 1.20),
            'Ni': (0.10, 1.50),
            'P': (0.0, 0.03),
            'S': (0.0, 0.02)
        }
        
        # 工艺参数边界
        self.process_bounds = {
            'Quench_Temp': (820, 1050),  # 淬火温度 °C
            'Temper_Temp': (150, 680),   # 回火温度 °C
            'Cooling_Rate': (5, 100)      # 冷却速率 °C/min
        }
    
    def calculate_cev(self, composition: Dict[str, float]) -> float:
        """
        计算碳当量CEV（IIW公式）
        CEV = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15
        """
        C = composition.get('C', 0)
        Mn = composition.get('Mn', 0)
        Cr = composition.get('Cr', 0)
        Mo = composition.get('Mo', 0)
        V = composition.get('V', 0)
        Ni = composition.get('Ni', 0)
        Cu = composition.get('Cu', 0)  # Cu不是决策变量，从数据中获取
        
        cev = C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15
        return cev
    
    def calculate_cost_index(self, composition: Dict[str, float], 
                            process_params: Dict[str, float]) -> float:
        """
        计算相对成本指数
        Cost = Σ(wt%_i × price_i) + E_Q + E_T
        CI = Cost / Cost_ref
        """
        # 成本计算（简化版，实际需要元素价格和热处理能耗）
        base_cost = sum(composition.values()) * 1.0  # 占位符
        process_cost = 0.1  # 占位符
        
        total_cost = base_cost + process_cost
        cost_ref = 1.0  # 参考钢（4140）的成本
        
        return total_cost / cost_ref
    
    def check_constraints(self, composition: Dict[str, float],
                         process_params: Dict[str, float]) -> Tuple[bool, List[str]]:
        """检查解是否满足所有约束"""
        violations = []
        
        # 检查成分边界
        for element, value in composition.items():
            if element in self.bounds:
                min_val, max_val = self.bounds[element]
                if value < min_val or value > max_val:
                    violations.append(f"{element}_bound")
        
        # 检查总合金含量上限
        zhuyao_yuansu = ['C', 'Cr', 'Mo', 'V', 'Mn', 'Si', 'Ni']
        total_alloy = sum(composition.get(elem, 0) for elem in zhuyao_yuansu)
        if total_alloy > 8.0:
            violations.append("alloy_cap")
        
        # 检查碳当量
        cev = self.calculate_cev(composition)
        if cev > self.cev_limit:
            violations.append("cev_limit")
        
        # 检查成本指数
        cost_idx = self.calculate_cost_index(composition, process_params)
        if cost_idx > self.cost_limit:
            violations.append("cost_limit")
        
        return len(violations) == 0, violations
    
    def repair_solution(self, composition: Dict[str, float],
                       process_params: Dict[str, float]) -> Tuple[Dict, Dict]:
        """修复不可行解，使其满足约束"""
        # 修复成分边界
        repaired_comp = {}
        for element, value in composition.items():
            if element in self.bounds:
                min_val, max_val = self.bounds[element]
                repaired_comp[element] = np.clip(value, min_val, max_val)
            else:
                repaired_comp[element] = value
        
        # 修复总合金含量
        zhuyao_yuansu = ['C', 'Cr', 'Mo', 'V', 'Mn', 'Si', 'Ni']
        total_alloy = sum(repaired_comp.get(elem, 0) for elem in zhuyao_yuansu)
        if total_alloy > 8.0:
            scale_factor = 8.0 / total_alloy
            for elem in zhuyao_yuansu:
                if elem in repaired_comp:
                    repaired_comp[elem] *= scale_factor
        
        # 修复CEV
        cev = self.calculate_cev(repaired_comp)
        if cev > self.cev_limit:
            # 按比例缩小
            scale_factor = self.cev_limit / cev
            for elem in ['C', 'Mn', 'Cr', 'Mo', 'V', 'Ni']:
                if elem in repaired_comp:
                    repaired_comp[elem] *= scale_factor
        
        return repaired_comp, process_params


class NSGA2Optimizer:
    """
    NSGA-II多目标优化器
    用于钢成分优化设计
    """
    
    def __init__(self, model, constraint_handler, 
                 population_size=200, max_generations=500):
        self.model = model  # 代理模型（用于预测性能）
        self.constraint_handler = constraint_handler
        self.population_size = population_size
        self.max_generations = max_generations
        
    def initialize_population(self) -> List[Dict]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            # 随机初始化
            candidate = self._random_candidate()
            # 修复约束
            comp, process = self.constraint_handler.repair_solution(
                candidate['composition'], candidate['process']
            )
            candidate['composition'] = comp
            candidate['process'] = process
            population.append(candidate)
        
        return population
    
    def _random_candidate(self) -> Dict:
        """生成随机候选解"""
        composition = {}
        for element, (min_val, max_val) in self.constraint_handler.bounds.items():
            composition[element] = np.random.uniform(min_val, max_val)
        
        process = {}
        for param, (min_val, max_val) in self.constraint_handler.process_bounds.items():
            process[param] = np.random.uniform(min_val, max_val)
        
        return {'composition': composition, 'process': process}
    
    def evaluate_objectives(self, candidate: Dict) -> np.ndarray:
        """
        评估目标函数
        目标：最大化强度、韧性、高温性能；最小化成本
        """
        # 准备模型输入（简化处理）
        # 实际需要将composition和process转换为特征向量
        
        # 用模型预测性能
        # predictions = self.model.predict(...)
        
        # 计算目标值
        # objectives = [
        #     -predictions['yield_strength'],  # 负号因为要最小化
        #     -predictions['tensile_strength'],
        #     -predictions['impact_toughness'],
        #     cost_index
        # ]
        
        # 占位符返回值
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    def fast_non_dominated_sort(self, population: List[Dict], 
                               objectives: np.ndarray) -> List[List[int]]:
        """快速非支配排序"""
        # NSGA-II的核心算法
        # 实现细节省略（实际需要完整实现）
        fronts = []
        fronts.append(list(range(len(population))))  # 占位符
        return fronts
    
    def crowding_distance(self, front: List[int], objectives: np.ndarray) -> np.ndarray:
        """计算拥挤距离（用于保持多样性）"""
        # 实现细节省略
        return np.ones(len(front))
    
    def selection(self, population: List[Dict], objectives: np.ndarray) -> List[Dict]:
        """锦标赛选择"""
        # 实现细节省略
        return population[:len(population)//2]
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """模拟二进制交叉(SBX)"""
        # 实现细节省略
        return parent1.copy(), parent2.copy()
    
    def mutation(self, individual: Dict) -> Dict:
        """多项式变异"""
        # 实现细节省略
        return individual
    
    def optimize(self) -> Dict:
        """运行NSGA-II优化"""
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        objectives = np.array([self.evaluate_objectives(ind) for ind in population])
        
        for generation in range(self.max_generations):
            # 快速非支配排序
            fronts = self.fast_non_dominated_sort(population, objectives)
            
            # 计算拥挤距离
            for front in fronts:
                front_indices = front
                distances = self.crowding_distance(front_indices, objectives)
            
            # 选择、交叉、变异
            selected = self.selection(population, objectives)
            
            offspring = []
            for i in range(0, len(selected)-1, 2):
                child1, child2 = self.crossover(selected[i], selected[i+1])
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                # 修复约束
                comp1, proc1 = self.constraint_handler.repair_solution(
                    child1['composition'], child1['process']
                )
                comp2, proc2 = self.constraint_handler.repair_solution(
                    child2['composition'], child2['process']
                )
                
                child1['composition'] = comp1
                child1['process'] = proc1
                child2['composition'] = comp2
                child2['process'] = proc2
                
                offspring.extend([child1, child2])
            
            # 合并父代和子代，选择下一代
            combined = population + offspring
            combined_objectives = np.vstack([
                objectives,
                np.array([self.evaluate_objectives(ind) for ind in offspring])
            ])
            
            # 更新种群（实现细节省略）
            population = combined[:self.population_size]
            objectives = combined_objectives[:self.population_size]
        
        # 提取Pareto前沿
        fronts = self.fast_non_dominated_sort(population, objectives)
        pareto_front = [population[i] for i in fronts[0]]
        
        return {
            'pareto_front': pareto_front,
            'final_population': population,
            'objectives': objectives
        }

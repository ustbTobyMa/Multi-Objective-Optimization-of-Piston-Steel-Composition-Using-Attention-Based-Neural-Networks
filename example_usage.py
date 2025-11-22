"""
Example Usage
=============
This script demonstrates how to use the framework components individually.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data.preprocessing import DataPreprocessor
from src.models.attention_model import AttentionBasedModel, ModelTrainer
from src.optimization.nsga2_optimizer import NSGA2Optimizer, ConstraintHandler
from src.visualization.plotting import Plotter


def example_data_preprocessing():
    """Example: Data preprocessing workflow."""
    print("=" * 60)
    print("Example 1: Data Preprocessing")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    
    # In actual usage, provide path to your data file
    # processed_data = preprocessor.process_pipeline("data/raw/steel_data.csv")
    
    print("Data preprocessing pipeline includes:")
    print("  - Data loading and validation")
    print("  - Missing value imputation")
    print("  - Outlier detection and removal")
    print("  - Feature scaling")
    print("  - Train/validation/test splitting")
    print("\nNote: Provide actual data file path for real execution")


def example_model_training():
    """Example: Model training workflow."""
    print("\n" + "=" * 60)
    print("Example 2: Model Training")
    print("=" * 60)
    
    # Initialize model
    model = AttentionBasedModel(
        input_dim=16,
        embed_dim=128,
        num_heads=8,
        hidden_dims=[256, 128, 64],
        num_tasks=6,
        dropout=0.2
    )
    
    print(f"Model initialized:")
    print(f"  - Input dimension: {model.embedding.in_features}")
    print(f"  - Embedding dimension: 128")
    print(f"  - Attention heads: 8")
    print(f"  - Number of tasks: 6")
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    trainer.setup_training(learning_rate=1e-3, weight_decay=1e-5)
    
    print("\nTraining configuration:")
    print("  - Optimizer: Adam")
    print("  - Learning rate: 1e-3")
    print("  - Weight decay: 1e-5")
    print("\nNote: Actual training requires data loaders and training loop")


def example_optimization():
    """Example: Optimization workflow."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Objective Optimization")
    print("=" * 60)
    
    # Initialize constraint handler
    constraint_handler = ConstraintHandler(cev_limit=0.60, cost_limit=1.30)
    
    print("Constraint handler initialized:")
    print(f"  - CEV limit: {constraint_handler.cev_limit}")
    print(f"  - Cost limit: {constraint_handler.cost_limit}")
    print(f"  - Composition bounds: {len(constraint_handler.bounds)} elements")
    print(f"  - Process bounds: {len(constraint_handler.process_bounds)} parameters")
    
    # Example: Check constraints for a candidate
    example_composition = {
        'C': 0.35, 'Cr': 1.2, 'Mo': 0.3, 'V': 0.1,
        'Mn': 0.8, 'Si': 0.5, 'Ni': 0.5, 'P': 0.02, 'S': 0.01
    }
    example_process = {
        'Quench_Temp': 900,
        'Temper_Temp': 550,
        'Cooling_Rate': 50
    }
    
    is_feasible, violations = constraint_handler.check_constraints(
        example_composition, example_process
    )
    
    print(f"\nExample candidate constraint check:")
    print(f"  - Feasible: {is_feasible}")
    if violations:
        print(f"  - Violations: {violations}")
    else:
        print("  - All constraints satisfied")
    
    print("\nNote: Full optimization requires trained model")


def example_visualization():
    """Example: Visualization workflow."""
    print("\n" + "=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)
    
    plotter = Plotter()
    
    print("Available visualization functions:")
    print("  - plot_correlation_matrix(): Element correlation analysis")
    print("  - plot_parity_plots(): Model prediction vs. measurements")
    print("  - plot_attention_heatmap(): Feature importance visualization")
    print("  - plot_pareto_front(): Optimization results")
    print("  - plot_convergence(): Optimization convergence")
    print("  - plot_property_comparison(): AI vs. conventional steels")
    
    print("\nNote: Visualization requires actual data/results")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Framework Usage Examples")
    print("=" * 60)
    
    example_data_preprocessing()
    example_model_training()
    example_optimization()
    example_visualization()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor full workflow, see main.py")
    print("For configuration options, see config_example.py")


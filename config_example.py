"""
Configuration Example
=====================
Example configuration file for the optimization framework.
Copy this file to config.py and modify according to your needs.
"""

# Data paths
DATA_CONFIG = {
    'raw_data_path': 'data/raw/steel_data.csv',
    'processed_data_path': 'data/processed/processed_data.pkl',
    'train_test_split': {
        'test_size': 0.15,
        'val_size': 0.15,
        'random_state': 42
    }
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': 16,  # 12 elements + 4 process parameters
    'embed_dim': 128,
    'num_heads': 8,
    'hidden_dims': [256, 128, 64],
    'num_tasks': 6,  # Number of target properties
    'dropout': 0.2
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'early_stopping_patience': 20,
    'gradient_clip_norm': 5.0
}

# Optimization configuration
OPTIMIZATION_CONFIG = {
    'population_size': 200,
    'max_generations': 500,
    'crossover_prob': 0.9,
    'mutation_prob': 0.1,
    'tournament_size': 2
}

# Constraint configuration
CONSTRAINT_CONFIG = {
    'cev_limit': 0.60,  # Carbon equivalent limit for weldability
    'cost_limit': 1.30,  # Relative cost index limit (30% increase)
    'aggregate_alloy_cap': 8.0  # Total alloy content limit (wt%)
}

# Composition bounds (wt%)
COMPOSITION_BOUNDS = {
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

# Process bounds
PROCESS_BOUNDS = {
    'Quench_Temp': (820, 1050),  # °C
    'Temper_Temp': (150, 680),   # °C
    'Cooling_Rate': (5, 100)      # °C/min
}

# Output paths
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'figures_dir': 'results/figures',
    'optimization_dir': 'results/optimization',
    'predictions_dir': 'results/predictions'
}


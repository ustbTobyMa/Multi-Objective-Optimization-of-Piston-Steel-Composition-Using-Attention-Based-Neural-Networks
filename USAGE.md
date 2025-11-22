# Usage Guide

This guide provides detailed instructions on how to use the multi-objective optimization framework for piston steel composition design.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the main workflow:**
```bash
python main.py
```

3. **Or run individual examples:**
```bash
python example_usage.py
```

## Detailed Usage

### 1. Data Preprocessing

The preprocessing module handles data loading, cleaning, and preparation:

```python
from src.data.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Run complete preprocessing pipeline
processed_data = preprocessor.process_pipeline("data/raw/steel_data.csv")

# Access processed data
X_train = processed_data['X_train']
y_train = processed_data['y_train']
X_test = processed_data['X_test']
y_test = processed_data['y_test']
```

**Data Format Requirements:**
- CSV file with columns for:
  - Composition: C, Cr, Mo, Mn, Si, Ni, P, S, V, Ti, Al, Cu (wt%)
  - Process: Quench_Temp, Temper_Temp, Cooling_Rate, Holding_Time
  - Properties: Yield_Strength, Tensile_Strength, Elongation, Impact_Toughness, Thermal_Conductivity, Thermal_Expansion

### 2. Model Training

Train the attention-based neural network:

```python
from src.models.attention_model import AttentionBasedModel, ModelTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Initialize model
model = AttentionBasedModel(
    input_dim=16,
    embed_dim=128,
    num_heads=8,
    hidden_dims=[256, 128, 64],
    num_tasks=6,
    dropout=0.2
)

# Initialize trainer
trainer = ModelTrainer(model)
trainer.setup_training(learning_rate=1e-3, weight_decay=1e-5)

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(500):
    loss = trainer.train_epoch(train_loader, criterion)
    # Validation and early stopping...
```

### 3. Model Evaluation

Evaluate model performance and extract attention weights:

```python
# Get predictions
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test), return_attention=True)
    predictions = outputs['predictions']
    attention_weights = outputs['attention_weights']

# Calculate metrics
from src.utils.helpers import calculate_r2_score, calculate_rmse

r2 = calculate_r2_score(y_test, predictions.numpy())
rmse = calculate_rmse(y_test, predictions.numpy())
```

### 4. Multi-Objective Optimization

Run NSGA-II optimization to find optimal compositions:

```python
from src.optimization.nsga2_optimizer import NSGA2Optimizer, ConstraintHandler

# Initialize constraint handler
constraint_handler = ConstraintHandler(cev_limit=0.60, cost_limit=1.30)

# Initialize optimizer
optimizer = NSGA2Optimizer(
    model=model,
    constraint_handler=constraint_handler,
    population_size=200,
    max_generations=500
)

# Run optimization
results = optimizer.optimize()

# Extract Pareto front
pareto_solutions = results['pareto_front']
```

### 5. Visualization

Generate visualizations for analysis:

```python
from src.visualization.plotting import Plotter

plotter = Plotter()

# Correlation matrix
plotter.plot_correlation_matrix(data, save_path='results/figures/correlation.png')

# Parity plots
plotter.plot_parity_plots(y_test, predictions, property_names, 
                         save_path='results/figures/parity.png')

# Attention heatmap
plotter.plot_attention_heatmap(attention_weights, feature_names, 
                               property_names, 
                               save_path='results/figures/attention.png')

# Pareto front
plotter.plot_pareto_front(objectives, 
                         save_path='results/figures/pareto.png')
```

## Configuration

Copy `config_example.py` to `config.py` and modify according to your needs:

```python
# Model parameters
MODEL_CONFIG = {
    'embed_dim': 128,
    'num_heads': 8,
    'hidden_dims': [256, 128, 64],
    ...
}

# Optimization parameters
OPTIMIZATION_CONFIG = {
    'population_size': 200,
    'max_generations': 500,
    ...
}
```

## Workflow Summary

The complete workflow follows these steps:

1. **Data Preparation**: Load and preprocess composition-property data
2. **Model Training**: Train attention-based surrogate model
3. **Model Evaluation**: Validate predictions and extract attention weights
4. **Optimization**: Run NSGA-II to find Pareto-optimal solutions
5. **Analysis**: Visualize results and select candidates
6. **Validation**: Experimental validation of selected compositions

## Notes

- This is a demonstration framework. For complete implementation, refer to the full paper.
- Some implementation details are simplified or omitted to focus on the overall methodology.
- Actual execution requires real data files and complete implementation of optimization algorithms.
- Model training and optimization are computationally intensive and may require GPU acceleration.

## Troubleshooting

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're running from the project root directory

**Data format issues:**
- Verify CSV file has required columns
- Check data types and units match expected format

**Model training issues:**
- Adjust batch size if memory errors occur
- Reduce model size if needed
- Use GPU if available for faster training

## Citation

If you use this framework, please cite the associated paper.


# Multi-Objective Optimization of Piston Steel Composition Using Attention-Based Neural Networks

This repository contains the implementation of a multi-objective optimization framework for commercial vehicle piston steel composition design, combining interpretable attention-based neural networks with constrained genetic algorithms.

## Overview

This work presents an interpretable, attention-based deep learning framework coupled with NSGA-II multi-objective optimization to design manufacturable steel compositions under weldability and cost constraints. The methodology enables:

- **Interpretable Property Prediction**: Multi-head attention mechanism provides feature-level interpretability for composition-property relationships
- **Constrained Multi-Objective Optimization**: NSGA-II algorithm searches for optimal compositions under realistic industrial constraints
- **Experimental Validation**: Framework validated through laboratory synthesis and characterization

## Methodology

The workflow consists of four main stages:

1. **Data Preprocessing** (Section 2.1): Harmonized composition-process-property dataset compilation and preprocessing
2. **Model Training** (Section 2.2): Attention-based multi-task neural network for property prediction
3. **Optimization** (Section 2.3): Constrained NSGA-II for multi-objective composition design
4. **Visualization** (Section 3): Results analysis and interpretation

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── preprocessing.py          # Data loading and preprocessing
│   ├── models/
│   │   └── attention_model.py        # Attention-based neural network
│   ├── optimization/
│   │   └── nsga2_optimizer.py        # NSGA-II multi-objective optimizer
│   ├── visualization/
│   │   └── plotting.py               # Visualization utilities
│   └── utils/
│       └── helpers.py                 # Helper functions
├── data/
│   ├── raw/                          # Raw data files
│   └── processed/                    # Processed data
├── results/
│   ├── figures/                      # Generated figures
│   ├── optimization/                 # Optimization results
│   └── predictions/                  # Model predictions
├── main.py                           # Main execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Key Features

### 1. Attention-Based Model Architecture

The model uses multi-head attention to capture context-dependent element-property interactions:

- **Input**: 12 alloying elements + 4 heat-treatment parameters
- **Architecture**: Embedding → Multi-head attention → MLP → Task-specific heads
- **Output**: Predictions for 6 target properties (strength, ductility, toughness, thermal properties)
- **Interpretability**: Attention weights reveal feature importance for each property

### 2. Constrained Multi-Objective Optimization

NSGA-II optimizer with industrial constraints:

- **Objectives**: Maximize strength, toughness, high-temperature performance; minimize cost
- **Constraints**: 
  - Carbon equivalent (CEV) ≤ 0.60 for weldability
  - Relative cost index ≤ 1.30
  - Composition bounds and aggregate alloy cap
- **Output**: Pareto-optimal solution set with trade-off analysis

### 3. Results Visualization

Comprehensive visualization suite:

- Database analysis (correlation matrices, property distributions)
- Model performance (parity plots, attention heatmaps)
- Optimization results (Pareto fronts, convergence plots)
- Property comparisons (AI-designed vs. conventional steels)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ustbTobyMa/Multi-Objective-Optimization-of-Piston-Steel-Composition-Using-Attention-Based-Neural-Networks.git
cd Multi-Objective-Optimization-of-Piston-Steel-Composition-Using-Attention-Based-Neural-Networks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Workflow

Run the complete pipeline:

```bash
python main.py
```

### Individual Components

**Data Preprocessing:**
```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
processed_data = preprocessor.process_pipeline("data/raw/steel_data.csv")
```

**Model Training:**
```python
from src.models.attention_model import AttentionBasedModel, ModelTrainer

model = AttentionBasedModel(input_dim=16, num_tasks=6)
trainer = ModelTrainer(model)
trainer.setup_training()
# Training loop...
```

**Optimization:**
```python
from src.optimization.nsga2_optimizer import NSGA2Optimizer, ConstraintHandler

constraint_handler = ConstraintHandler(cev_limit=0.60, cost_limit=1.30)
optimizer = NSGA2Optimizer(model, constraint_handler)
results = optimizer.optimize()
```

For detailed usage instructions, see [USAGE.md](USAGE.md).

## Results

The framework successfully:

- Achieved R² > 0.95 for all target properties
- Generated 36 Pareto-optimal solutions under constraints
- Validated AI-designed steels showing 18-25% strength improvements
- Demonstrated improved oxidation resistance at 600°C

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ma2024multi,
  title={Multi-Objective Optimization of Commercial Vehicle Piston Steel Composition Using Attention-Based Neural Networks},
  author={Ma, Weitao and Rao, Yanjun and Zhang, Zheyue and others},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact:
- Corresponding author: Renbo Song (songrb@mater.ustb.edu.cn)

## Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (No. 52074033)
- Hebei Steel Group Key Research and Development Project (HG2023242)

## Notes

**Important**: This repository provides a demonstration of the methodology and workflow. For complete implementation details, please refer to the full paper. Some implementation details are simplified or omitted to focus on the overall framework and interpretability.

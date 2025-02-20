# KMNIST Optimizer Analysis

This project is designed to analyze the performance of different optimizers on the KMNIST dataset using PyTorch. It includes hyperparameter tuning, cross-validation, training, and testing, with results saved for further analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running Experiments](#running-experiments)
6. [Results](#results)
7. [Supported Optimizers](#supported-optimizers)
8. [Dependencies](#dependencies)
9. [License](#license)

---

## Overview

This project conducts experiments on the KMNIST dataset using multiple optimizers. It aims to evaluate the performance of each optimizer through hyperparameter tuning, cross-validation, and model testing. The results are saved in CSV format for easy analysis.

---

## Features

- Hyperparameter tuning using Optuna
- Cross-validation for model evaluation
- Training and testing on the KMNIST dataset
- Support for multiple optimizers
- Results are saved as CSV files for easy analysis

---

## Installation

Ensure you have Python 3.x and conda installed. Then, create and activate a conda environment:

```bash
conda create -n torch_gpu python=3.x
conda activate torch_gpu
```

Install the required dependencies:

```bash
pip install torch torchvision numpy pandas optuna
```

---

## Configuration

Configuration settings are managed in `configs.py`, where different optimizer configurations are defined.

Available optimizers:
- `AdamConfig`
- `AdamWConfig`
- `RMSPropConfig`
- `SAMConfig`
- `LAMBConfig`
- `NovoGradConfig`
- `AdoptConfig`

Example configuration for Adam Optimizer:
```python
class AdamConfig(OptimizerConfig):
    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 0.0,
            ranges = {
                "lr_range": [1e-5, 1e-2],
                "beta1_range": [0.7, 0.99],
                "beta2_range": [0.8, 0.9999],
                "weight_decay_range": [4e-4, 4e-2],
            }
        )
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
```

---

## Running Experiments

To run experiments, use the `experiments.py` script. The following arguments are supported:
- `--optimizer`: Choose optimizers (e.g., adam, adamw, rmsprop, sam, lamb, novograd, adopt)
- `--batch_size`: Batch size for training
- `--experiment_name`: Name of the experiment for saving results

Example:
```bash
python experiments.py --optimizer adam,adamw,rmsprop --batch_size 128 --experiment_name exp1
```

For multiple optimizers:
```bash
python experiments.py --optimizer adam,adamw,rmsprop,sam,lamb,novograd,adopt --batch_size 16384 --experiment_name exp4
```

---

## Results

Results are saved in the following directories:
- `./results/{experiment_name}/status.csv`: Epoch logs including training and validation metrics
- `./results/{experiment_name}/metrics.csv`: Test results including accuracy, precision, and loss
- `./results/{experiment_name}/cv_results.csv`: Cross-validation results

---

## Supported Optimizers

This project supports the following optimizers:

1. **Adam**: Adaptive Moment Estimation ([Adam Paper](https://arxiv.org/abs/1412.6980))
2. **AdamW**: Adam with decoupled weight decay ([AdamW Paper](https://arxiv.org/abs/1711.05101))
3. **RMSProp**: Root Mean Square Propagation ([RMSProp Reference](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))
4. **SAM**: Sharpness-Aware Minimization ([SAM Paper](https://arxiv.org/abs/2010.01412))
5. **LAMB**: Layer-wise Adaptive Moments ([LAMB Paper](https://arxiv.org/abs/1904.00962))
6. **NovoGrad**: Combines gradient normalization with Adam-like momentum ([NovoGrad Paper](https://arxiv.org/abs/1905.11286))
7. **Adopt**: Custom optimizer implementation ([Adopt Paper](https://arxiv.org/abs/2411.02853))

Each optimizer is configurable with its own set of hyperparameters and ranges for learning rate, beta values, and weight decay.

---

## Dependencies

The following dependencies are required to run this project:

- `torch`: PyTorch framework
- `torchvision`: Dataset and model utilities for PyTorch
- `numpy`: Numerical computation
- `pandas`: Data manipulation and analysis
- `optuna`: Hyperparameter tuning framework

Install them with:

```bash
pip install torch torchvision numpy pandas optuna
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [PyTorch](https://pytorch.org) - Deep learning framework
- [Optuna](https://optuna.org) - Hyperparameter optimization framework

---

This documentation provides a comprehensive overview and usage guide for the KMNIST Optimizer Analysis project.


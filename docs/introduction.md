# Introduction

## Motivation

Atomistic Foundation Models have emerged as powerful tools in molecular and materials science. However, the diverse implementations of these open-source models, with their varying architectures and interfaces, create significant barriers for customized fine-tuning and downstream applications.

MatterTune is a comprehensive platform that addresses these challenges through systematic yet general abstraction of Atomistic Foundation Model architectures. By adopting a modular design philosophy, MatterTune provides flexible and concise user interfaces that enable intuitive and efficient fine-tuning workflows.

## Key Features

### Pre-trained Model Support
Seamlessly work with multiple state-of-the-art pre-trained models including:
- JMP
- EquiformerV2
- M3GNet
- ORB

### Flexible Property Predictions
Support for various molecular and materials properties:
- Energy prediction
- Force prediction (both conservative and non-conservative)
- Stress tensor prediction
- Custom system-level property predictions

### Data Processing
Built-in support for multiple data formats:
- XYZ files
- ASE databases
- Materials Project database
- Matbench datasets
- Custom datasets

### Training Features
- Automated train/validation splitting
- Multiple loss functions (MAE, MSE, Huber, L2-MAE)
- Property normalization and scaling
- Early stopping and model checkpointing
- Comprehensive logging with WandB, TensorBoard, and CSV support

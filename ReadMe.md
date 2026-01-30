# Anomaly Detection

A comprehensive framework for image-based anomaly detection using multiple vision models and techniques. This project provides both supervised (full-shot) and zero-shot learning approaches for classification and object detection tasks.

## Features

- **Multiple Detection Approaches**
  - Full-shot learning: Train custom models with labeled data
  - Zero-shot learning: Use pre-trained models without training data
  
- **Vision Backbones**
  - CLIP: Vision-language models
  - DINO: Self-supervised vision models
  - DINOv2: Improved DINO architecture
  - EfficientNetV2: Lightweight efficient models
  - OWLv2: Object detection
  - ViT: Vision Transformers
  
- **Applications**
  - Image classification (anomaly vs. normal)
  - Object detection for anomalies
  - Memory bank-based anomaly detection
  
- **Interactive Demo**
  - Streamlit-based web interface for easy exploration
  - Real-time inference and visualization
  - Support for multiple model types

## Installation

### Requirements
- Python >= 3.9
- PyTorch 2.1.2
- TorchVision 0.16.2

### Setup with Pixi (Recommended)
```bash
pixi install
```

### Setup with pip
```bash
pip install -e .
```

## Project Structure

```
src/
├── benchmark/          # Performance benchmarking scripts
│   ├── classification/ # Classification benchmarks
│   ├── detection/      # Detection benchmarks
│   └── memory_bank/    # Memory bank evaluation
├── data/              # Data loading and preprocessing
│   ├── datasets/      # Dataset loaders (MVTec, etc.)
│   └── splits/        # Data split utilities
├── demo/              # Streamlit web application
│   ├── pages/         # App pages for different tasks
│   ├── adapters/      # Model adapters and configurations
│   └── core/          # Core services and interfaces
└── models/            # Model implementations
    ├── adapters/      # Vision backbone adapters
    └── core/          # Core model abstractions
```

## Usage

### Run the Interactive Demo
```bash
cd src/demo
streamlit run 01_Home.py
```

The demo provides pages for:
- **Classification**: Image classification using CLIP or linear adapters
- **Memory Bank Anomaly Detection**: Using memory-based approaches
- **OWLv2 Detection**: Zero-shot object detection

### Training (Full-Shot)
See the notebooks in `src/benchmark/classification/full_shot/` for training examples:
- `training.ipynb`: Training scripts
- `inference_time.ipynb`: Performance evaluation

### Zero-Shot Inference
See the notebooks in `src/benchmark/classification/zero_shot/` for zero-shot examples:
- `zero_shot_demo.ipynb`: Basic usage examples

## Datasets

- **MVTec**: Anomaly detection benchmark dataset for industrial applications
- Custom datasets supported through the data loading utilities

## Benchmarking

Performance benchmarks are available for:
- Model inference time
- Memory usage
- Accuracy metrics
- Hardware configurations (e.g., Apple M4)

Run benchmark scripts:
```bash
python src/benchmark/classification/full_shot/training.ipynb
python src/benchmark/detection/full_shot/training.ipynb
```

## Architecture

### Core Components
- **Vision Backbones**: Pre-trained feature extractors (CLIP, DINO, etc.)
- **Adapters**: Lightweight modules to adapt backbones for specific tasks
- **Models**: High-level model implementations combining backbones and adapters
- **Memory Bank**: Efficient storage and retrieval for anomaly detection

### Support for Multiple Frameworks
- Image classification adapters
- Zero-shot classifiers
- Memory bank models
- Detection models

## Configuration

Hardware and model configurations are stored in YAML files:
- `src/benchmark/hardware_configs/`: Hardware specifications for different devices
- `src/demo/infrastructure/config/`: Application configuration files

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is part of an internship at OCTO Technology.

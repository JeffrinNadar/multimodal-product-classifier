# Large-Scale Multimodal Product Classifier with Active Learning

A production-ready machine learning system that classifies products using both text and image data, featuring active learning to optimize labeling efficiency.

## 🎯 Project Overview

This project implements a multimodal deep learning classifier trained on the Amazon Products dataset (3M+ samples), combining CNN-based image embeddings with transformer-based text embeddings to classify items into 500+ categories.

**Key Features:**
- Multimodal architecture (text + images)
- Multiple ML approaches (classical → deep learning)
- Active learning loop for efficient training
- Comprehensive experiment tracking
- Framework comparison (PyTorch, TensorFlow, scikit-learn)

## 🏗️ Architecture

### Data Pipeline
- Parallel preprocessing for large-scale data
- HuggingFace tokenization (BERT-based)
- Image augmentation and normalization
- Stratified train/val/test splits

### Models Implemented

#### 1. Baseline Models (scikit-learn)
- Logistic Regression
- Linear SVM
- Random Forest
- Naive Bayes

#### 2. Deep Learning Models
- **Text Encoder**: DistilBERT (768-d embeddings)
- **Image Encoder**: ResNet50 (2048-d embeddings)
- **Fusion Model**: Concatenated multimodal MLP

#### 3. Active Learning
- Uncertainty sampling (entropy-based)
- Iterative retraining pipeline
- 40% reduction in labeling cost


## AI-Assisted Development

This project was developed using modern AI coding tools:

**Tools Used:**
- **Claude Code**: Generated data pipeline, model architectures, and training loops
- **GitHub Copilot**: Assisted with boilerplate code and utility functions

**Learnings:**
- AI tools excel at generating standard ML patterns and boilerplate
- Manual verification crucial for model architectures and loss functions
- Human oversight needed for data preprocessing edge cases
- Significant productivity boost (~40% faster development)

## 🧪 Experiment Tracking

Experiments tracked using:
- **TensorBoard**: Training metrics, loss curves
- **Weights & Biases**: Hyperparameter tuning, model comparison

View experiments:
```bash
tensorboard --logdir=runs/
```


## 📝 Citation

```bibtex
@misc{multimodal-classifier-2024,
  author = {Jeffrin Nadar},
  title = {Large-Scale Multimodal Product Classifier},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/JeffrinNadar/multimodal-product-classifier}
}
```

## 📄 License

MIT License

## 📧 Contact

Jeffrin Nadar - anthony.jeffrin.b@gmail.com

Project Link: [https://github.com/JeffrinNadar/multimodal-product-classifier](https://github.com/JeffrinNadar/multimodal-product-classifier)

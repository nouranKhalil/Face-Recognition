
# Face Recognition using Dimensionality Reduction and Unsupervised Learning

This project investigates the effectiveness of various dimensionality reduction and clustering techniques for face recognition. Specifically, we evaluate Principal Component Analysis (PCA), K-Means Clustering, Gaussian Mixture Models (GMM), and Convolutional Autoencoders using the ORL face dataset. The study aims to compare performance across models and analyze how dimensionality reduction impacts clustering quality.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Implemented Techniques](#implemented-techniques)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [K-Means Clustering](#k-means-clustering)
  - [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
  - [Convolutional Autoencoders](#convolutional-autoencoders)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Authors](#authors)

## Project Overview

Face recognition aims to identify subjects from images. This project implements and compares several approaches for dimensionality reduction and unsupervised clustering to recognize individuals from grayscale face images. The results provide insights into how linear and nonlinear feature reduction techniques interact with clustering algorithms.

## Dataset Description

- **Dataset**: ORL Face Dataset  
- **Subjects**: 40 individuals  
- **Images per Subject**: 10 grayscale images (92×112 pixels)  
- **Train/Test Split**: Odd-numbered samples used for training; even-numbered samples used for testing

## Implemented Techniques

### Principal Component Analysis (PCA)

- **Purpose**: Linear dimensionality reduction by retaining variance thresholds (80%, 85%, 90%, 95%)
- **Process**:
  - Mean normalization and covariance matrix computation
  - Eigen decomposition and sorting
  - Retain minimal number of components for given variance
- **Visualization**: Eigenfaces and image reconstructions
- **Insight**: Higher thresholds yield better reconstructions but increase dimensionality

### K-Means Clustering

- **Objective**: Partition data into *k* clusters using unsupervised learning
- **Initialization**: KMeans++ for centroid selection
- **Assignment**: Nearest-centroid labeling
- **Update**: Centroid recalculation and convergence check
- **Observations**:
  - Accuracy increases with higher *k*
  - Best performance at *k = 60*, *α = 0.9*

### Gaussian Mixture Models (GMM)

- **Objective**: Probabilistic clustering using Expectation-Maximization
- **Initialization**:
  - Uniform weights
  - KMeans-based mean estimation
  - Identity covariance matrices
- **Optimization**:
  - Log-probability for numerical stability
  - NumPy broadcasting for efficiency
- **Observation**:
  - Optimal performance at *k = 60*, *α = 0.85*
  - More sensitive to overfitting than K-Means

### Convolutional Autoencoders

- **Purpose**: Nonlinear dimensionality reduction via deep learning
- **Architecture**:
  - Encoder: 4 convolutional layers → fully connected layer (latent space of size 128)
  - Decoder: 4 transposed convolutional layers
- **Training**:
  - Optimizer: Adam (learning rate = 1e-3, weight decay = 1e-5)
  - Epochs: 100
  - Loss: Mean Squared Error (MSE)
- **Limitation**: Underperformance due to limited dataset size and training complexity

## Evaluation Metrics

- **Accuracy**: Correct cluster assignment relative to ground truth
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Maps true vs. predicted labels to assess model behavior

## Results Summary

| Method                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| K-Means + PCA          | 0.84     | 0.83     |
| K-Means + Autoencoder  | 0.81     | 0.78     |
| GMM + PCA              | 0.78     | 0.757    |
| GMM + Autoencoder      | 0.68     | 0.655    |

- **Best overall model**: K-Means with PCA at α = 0.9 and k = 60
- **Key insight**: Linear PCA outperformed Autoencoders due to its simplicity, deterministic nature, and better generalization on small datasets

## Installation & Requirements

Make sure Python 3.x is installed, then install the required packages:

```bash
pip install numpy matplotlib pandas torch scikit-learn scipy tabulate
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/nouranKhalil/Face-Recognition.git
   cd Face-Recognition
   ```
3. Outputs will include:
   - Reconstruction visualizations
   - Accuracy and F1 score reports
   - Confusion matrices for detailed analysis

## Conclusion

This study demonstrates that classical PCA combined with K-Means provides strong performance for face recognition tasks involving small datasets. While Autoencoders offer potential for nonlinear feature extraction, their effectiveness is constrained by dataset size and training complexity. PCA remains a reliable and efficient choice in such settings.

## Authors

- Hager Ashraf Mohamed Melook  
- Nouran Ashraf Yousef  
- Rana Mohamed Ali Attia

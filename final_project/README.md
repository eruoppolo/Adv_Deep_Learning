# Project: "Extremes vs. Variance: A Comparative Study of Deep Archetypal Analysis and Kernel PCA for Non-Linear Manifold Uncovering"

## 📎 **Project Objectives**

### **Primary Objectives:**
1. **Demonstrate fundamental philosophical differences** between variance-based (Kernel PCA) and extremal-point-based (Deep Archetypal Analysis) approaches to non-linear dimensionality reduction on facial expression data

2. **Quantitatively evaluate** which approach better captures semantically meaningful features for emotion recognition tasks

3. **Provide practical insights** on when to use each method based on empirical evidence from reconstruction quality, interpretability, and downstream task performance

### **Secondary Objectives:**
1. **Bridge classical and modern techniques** by comparing kernel methods with deep learning approaches on the same problem

2. **Analyze computational trade-offs** between the methods in terms of training time, inference speed, and scalability

3. **Explore interpretability** of learned representations through visualization and feature analysis

### **Learning Objectives:**
1. Gain deep understanding of kernel methods and their connection to neural architectures
2. Master implementation of both classical (Kernel PCA) and modern (DAA) unsupervised learning techniques
3. Develop skills in comparative analysis and scientific evaluation of ML methods

---

## 🔧 **Techniques**

### **Core Techniques:**

#### **1. Deep Archetypal Analysis (DAA)**
- **Architecture**: CNN encoder → Softmax weights → Convex combination with learnable archetypes → CNN decoder
- **Training**: End-to-end gradient descent with MSE reconstruction loss
- **Regularization**: Sparsity penalty on archetypal weights, diversity constraint on archetypes
- **Output**: K learnable archetypes representing data extremes + convex weights for each sample

#### **2. Kernel Principal Component Analysis (Kernel PCA)**
- **Kernels to test**:
  - RBF: k(x,y) = exp(-γ||x-y||²) with γ ∈ {0.001, 0.01, 0.1}
  - Polynomial: k(x,y) = (⟨x,y⟩ + c)^d with d ∈ {2, 3, 4}
- **Implementation**: Eigendecomposition of centered kernel matrix
- **Reconstruction**: Pre-image problem approximation using fixed-point iteration
- **Output**: Principal components in feature space + projection coefficients

### **Evaluation Techniques:**

#### **3. Reconstruction Analysis**
- **Metrics**: MSE, PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)
- **Variable component analysis**: Performance with k ∈ {3, 5, 7, 10} components/archetypes

#### **4. Feature Quality Assessment**
- **Downstream classification**:
  - Classifiers: SVM (RBF kernel), 3-layer MLP, Random Forest
  - Cross-validation: 5-fold stratified CV
  - Metrics: Accuracy, F1-score (macro), confusion matrices
- **Feature visualization**:
  - t-SNE/UMAP on learned representations
  - Interpolation studies (for DAA)

#### **5. Comparative Visualization**
- **Direct comparison**: Side-by-side archetype vs eigenface visualization
- **Reconstruction progression**: Quality vs number of components
- **Feature importance**: Which components/archetypes are most used

### **Supporting Techniques:**
- **Data preprocessing**: Normalization, augmentation (horizontal flips only)
- **Hyperparameter optimization**: Grid search with validation set
- **Statistical testing**: Paired t-tests for significance of performance differences

---

```
final_project/
├── README.md                    # Project overview and instructions
├── requirements.txt             # Dependencies
├── data/
│   └── jaffe/                   # Dataset
├── src/
│   ├── deep_aa_model.py
│   ├── kernel_pca.py
│   └── comparison_suite.py
├── notebooks/                   # Jupyter notebooks
├── models/                      # Saved models
├── features/                    # Extracted features
├── figures/                     # Generated visualizations
├── results/                     # Tables and metrics
└── report/
    ├── technical_report.pdf
    ├── executive_summary.pdf
    └── presentation.pptx
```

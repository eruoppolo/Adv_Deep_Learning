# Project Definition: "Extremes vs. Variance: A Comparative Study of Deep Archetypal Analysis and Kernel PCA for Non-Linear Manifold Uncovering"

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

## 📦 **Deliverables and Output**

### **Code Deliverables:**

1. **`deep_aa_model.py`**
   - Complete DAA implementation with configurable architecture
   - Training script with logging and checkpointing
   - Archetype extraction and visualization functions

2. **`kernel_pca.py`**
   - Kernel PCA implementation supporting multiple kernels
   - Pre-image reconstruction solver
   - Eigenface visualization utilities

3. **`comparison_suite.py`**
   - Unified evaluation framework
   - Feature extraction for both methods
   - Classification pipeline
   - Visualization generation scripts

4. **`notebooks/`** directory:
   - `01_data_exploration.ipynb`: JAFFE dataset analysis
   - `02_daa_training.ipynb`: DAA experiments and ablations
   - `03_kernel_pca_analysis.ipynb`: Kernel selection and component analysis
   - `04_comparative_study.ipynb`: Head-to-head comparisons
   - `05_results_visualization.ipynb`: Final results and figures

### **Model/Data Outputs:**

1. **Trained Models:**
   - `models/daa_best.pth`: Best DAA model checkpoint
   - `models/kernel_pca_rbf.pkl`: Fitted Kernel PCA models (one per kernel type)
   - `models/classifiers/`: Trained downstream classifiers

2. **Extracted Features:**
   - `features/archetypes.npy`: Learned DAA archetypes (7×latent_dim)
   - `features/eigenfaces.npy`: Top Kernel PCA components
   - `features/daa_weights.npy`: Archetypal coefficients for all samples
   - `features/kpca_projections.npy`: Kernel PCA projections

### **Visualization Outputs:**

1. **Core Visualizations:**
   - `figures/archetypes_grid.png`: 7 learned archetypes visualized
   - `figures/eigenfaces_grid.png`: Top 7 kernel principal components
   - `figures/comparison_grid.png`: Side-by-side comparison
   - `figures/reconstruction_quality.png`: MSE/SSIM vs number of components
   - `figures/interpolation_study.png`: Smooth transitions between archetypes

2. **Analysis Plots:**
   - `figures/tsne_comparison.png`: t-SNE of both feature spaces
   - `figures/confusion_matrices.png`: Classification performance comparison
   - `figures/ablation_results.png`: Effect of hyperparameters
   - `figures/computational_comparison.png`: Time/memory requirements

### **Written Deliverables:**

1. **Technical Report (8-10 pages):**
   - **Abstract**: 200-word summary
   - **Introduction**: Motivation and research questions
   - **Background**: Mathematical foundations of AA and Kernel PCA
   - **Methodology**: Implementation details and experimental setup
   - **Results**: Comprehensive evaluation with tables and figures
   - **Discussion**: Interpretation and practical implications
   - **Conclusion**: Key findings and future directions
   - **References**: Academic citations

2. **Executive Summary (1 page):**
   - Key findings in bullet points
   - Practical recommendations table
   - When to use each method (decision tree/flowchart)

3. **Presentation Slides (15 slides):**
   - Visual-heavy presentation for exam defense
   - Live demo section showing interpolations
   - Clear takeaway messages

### **Quantitative Results Tables:**

1. **Table 1: Reconstruction Performance**
   ```
   Method          | MSE    | PSNR   | SSIM   | Time(ms)
   DAA (K=7)       | 0.023  | 28.4   | 0.89   | 12.3
   Kernel PCA-RBF  | 0.031  | 26.1   | 0.85   | 8.7
   Kernel PCA-Poly | 0.029  | 26.8   | 0.86   | 9.1
   ```

2. **Table 2: Classification Accuracy**
   ```
   Features        | SVM    | MLP    | RF     | Avg
   DAA Weights     | 78.3%  | 81.2%  | 76.5%  | 78.7%
   Kernel PCA-RBF  | 74.6%  | 77.3%  | 73.2%  | 75.0%
   Raw Pixels      | 68.2%  | 71.5%  | 65.3%  | 68.3%
   ```

3. **Table 3: Computational Requirements**
   ```
   Method          | Training | Inference | Memory | Scalability
   DAA             | 45 min   | 12 ms     | 120MB  | O(n)
   Kernel PCA      | 8 min    | 9 ms      | n²×8B  | O(n²)
   ```

### **Repository Structure:**
```
project/
├── README.md                    # Project overview and instructions
├── requirements.txt             # Dependencies
├── data/
│   └── jaffe/                  # Dataset
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

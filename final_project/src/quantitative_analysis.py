import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# Import our models and helpers
from deep_aa_model import DeepAA
from sklearn.decomposition import KernelPCA
from comparative_analysis import load_daa_model

# --- Configuration ---
TARGET_DIGIT = 4
N_COMPONENTS_KPCA = 3
BEST_GAMMA = 0.02

# --- Paths ---
DAA_MODEL_PATH = f'../models/daa_mnist_digit_{TARGET_DIGIT}/checkpoints/best_model.pth'
DATA_PATH = f'../data/mnist_digit_{TARGET_DIGIT}_flat.npy'
OUTPUT_DIR = f'../features/results_digit_{TARGET_DIGIT}'
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Load Data and Models ---
    print("Loading data and models for quantitative analysis...")
    X_data_flat = np.load(DATA_PATH)
    X_data_img = X_data_flat.reshape(-1, 32, 32)
    daa_model = load_daa_model(DAA_MODEL_PATH, device)
    kpca = KernelPCA(n_components=N_COMPONENTS_KPCA, kernel="rbf", gamma=BEST_GAMMA, fit_inverse_transform=True, random_state=42)
    kpca.fit(X_data_flat)
    print("Data and models loaded.")

    # --- Reconstruction Error Analysis (No Changes) ---
    print("\n--- Running Reconstruction Error Analysis ---")
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_data_flat).view(-1, 1, 32, 32).float().to(device)
        daa_recons_list = [daa_model(batch)[0].cpu().numpy() for batch in torch.split(X_tensor, 128)]
        daa_recons = np.concatenate(daa_recons_list, axis=0).squeeze()
    kpca_latent = kpca.transform(X_data_flat)
    kpca_recons = kpca.inverse_transform(kpca_latent).reshape(-1, 32, 32)
    mse_daa = mean_squared_error(X_data_flat, daa_recons.reshape(-1, 1024))
    mse_kpca = mean_squared_error(X_data_flat, kpca_recons.reshape(-1, 1024))
    X_data_ssim = (X_data_img + 1) / 2
    daa_recons_ssim = (daa_recons + 1) / 2
    kpca_recons_ssim = (kpca_recons + 1) / 2
    ssim_daa = np.mean([ssim(X_data_ssim[i], np.clip(daa_recons_ssim[i],0,1), data_range=1.0) for i in range(len(X_data_ssim))])
    ssim_kpca = np.mean([ssim(X_data_ssim[i], np.clip(kpca_recons_ssim[i],0,1), data_range=1.0) for i in range(len(X_data_ssim))])
    recon_results = pd.DataFrame({'Method': ['DAA', 'KPCA'], 'MSE (Lower is Better)': [mse_daa, mse_kpca], 'SSIM (Higher is Better)': [ssim_daa, ssim_kpca]})
    print("\nReconstruction Metrics:"); print(recon_results)
    
    # --- Clustering Evaluation (No Changes) ---
    print("\n--- Running Clustering Evaluation ---")
    with torch.no_grad():
        alpha_list = [daa_model.encode(batch)[0].cpu().numpy() for batch in torch.split(X_tensor, 128)]
        daa_latent_alphas = np.concatenate(alpha_list, axis=0)
    n_clusters = 3
    kmeans_daa = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(daa_latent_alphas)
    silhouette_daa = silhouette_score(daa_latent_alphas, kmeans_daa.labels_)
    kmeans_kpca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(kpca_latent)
    silhouette_kpca = silhouette_score(kpca_latent, kmeans_kpca.labels_)
    cluster_results = pd.DataFrame({'Method': ['DAA (alpha weights)', 'KPCA (projections)'], 'Silhouette Score (Higher is Better)': [silhouette_daa, silhouette_kpca]})
    print("\nClustering Quality Metrics:"); print(cluster_results)

    # --- Latent Space Visualization (REVISED CODE) ---
    print("\n--- Generating Final Latent Space Visualizations ---")
    
    # --- FIX: Ensure labels have the same size as data points ---
    if len(kmeans_daa.labels_) != len(daa_latent_alphas):
        raise ValueError("Mismatch in DAA data and labels size. This should not happen.")
    if len(kmeans_kpca.labels_) != len(kpca_latent):
        raise ValueError("Mismatch in KPCA data and labels size. This should not happen.")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    palette = plt.get_cmap('viridis')

    # 3.1: DAA Latent Space Visualization
    ax1 = axes[0]
    with torch.no_grad():
        latent_archetypes = daa_model.archetypes.cpu().numpy()
        daa_latent_z = daa_latent_alphas @ latent_archetypes
    
    pca_viz = PCA(n_components=2)
    # FIT PCA on Z and Archetypes together for a consistent projection
    combined_daa_latents = np.vstack([daa_latent_z, latent_archetypes])
    projected_latents = pca_viz.fit_transform(combined_daa_latents)
    
    projected_z = projected_latents[:len(daa_latent_z)]
    projected_archetypes = projected_latents[len(daa_latent_z):]
    
    scatter1 = ax1.scatter(projected_z[:, 0], projected_z[:, 1], c=kmeans_daa.labels_, cmap=palette, alpha=0.5, s=20)
    ax1.scatter(projected_archetypes[:, 0], projected_archetypes[:, 1], marker='D', s=250, c='red', edgecolor='black', zorder=10)

    try:
        hull = ConvexHull(projected_archetypes)
        for simplex in hull.simplices:
            ax1.plot(projected_archetypes[simplex, 0], projected_archetypes[simplex, 1], 'r--', lw=2)
        ax1.plot([projected_archetypes[hull.vertices[-1], 0], projected_archetypes[hull.vertices[0], 0]],
                 [projected_archetypes[hull.vertices[-1], 1], projected_archetypes[hull.vertices[0], 1]], 'r--', lw=2)
    except Exception as e:
        print(f"Could not compute convex hull for DAA: {e}")

    ax1.set_title("DAA Latent Space (Clusters & Convex Hull)", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Principal Component 1 (for visualization)", fontsize=12)
    ax1.set_ylabel("Principal Component 2 (for visualization)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    legend_labels = [f'Cluster {i}' for i in range(n_clusters)]
    ax1.legend(handles=scatter1.legend_elements(num=n_clusters)[0], labels=legend_labels, title="Clusters")


    # 3.2: KPCA Latent Space Visualization
    ax2 = axes[1]
    scatter2 = ax2.scatter(kpca_latent[:, 0], kpca_latent[:, 1], c=kmeans_kpca.labels_, cmap=palette, alpha=0.5, s=20)
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_title("KPCA Latent Space (Clusters around Origin)", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Principal Component 1 (Variance Axis)", fontsize=12)
    ax2.set_ylabel("Principal Component 2 (Variance Axis)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(handles=scatter2.legend_elements(num=n_clusters)[0], labels=legend_labels, title="Clusters")
    ax2.axis('equal')

    plt.tight_layout()
    final_plot_path = os.path.join(OUTPUT_DIR, 'latent_space_cluster_comparison.png')
    plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Final latent space comparison saved to {final_plot_path}")
    plt.show()
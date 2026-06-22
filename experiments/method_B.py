# method_B.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import sys
sys.path.append('..')
from utils import vectorize_text_tfidf

# ========================================================
# Configuration
# ========================================================

DATASET_PATH = '../data/processed/dataset_7labels_clean.csv'
FIGURES_DIR = '../figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ========================================================
# Load and preprocess dataset
# ========================================================

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {len(df)} instances")

# Vectorize text using TF-IDF
tfidf_matrix, tfidf_vectorizer, selected_data = vectorize_text_tfidf(
    df, 
    precomputed_vectorizer=None, 
    max_features=5000, 
    min_df=3, 
    max_df=0.85
)

# Standardize features for better K-means performance
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(tfidf_matrix).toarray()
print(f"Feature matrix shape: {X.shape}")

# Get true labels for evaluation
true_labels = selected_data['label'].tolist()
emotion_labels = sorted(selected_data['label'].unique())


# ========================================================
# Helper Functions
# ========================================================

def get_sample_data(X, sample_size=21000, random_seed=42, iteration=0):
    """Draw a random sample from the dataset."""
    np.random.seed(random_seed + iteration)
    sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Standardize the sample
    scaler_sample = StandardScaler()
    X_scaled = scaler_sample.fit_transform(X_sample)
    
    return X_scaled, sample_indices


def compute_cluster_metrics(X_scaled, labels, true_labels_sample, k):
    """Compute evaluation metrics for clustering."""
    # Skip if not enough clusters or samples
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return {'ari': np.nan, 'nmi': np.nan, 'silhouette': np.nan}
    
    # Compute metrics
    ari = adjusted_rand_score(true_labels_sample, labels)
    nmi = normalized_mutual_info_score(true_labels_sample, labels)
    
    # Silhouette requires at least 2 clusters and each cluster has >1 sample
    if n_clusters >= 2 and np.min(np.bincount(labels)) > 1:
        sil = silhouette_score(X_scaled, labels, metric='euclidean')
    else:
        sil = np.nan
    
    return {'ari': ari, 'nmi': nmi, 'silhouette': sil}


# ========================================================
# Run 1: Finding Optimal K
# ========================================================

def run_optimal_k_analysis(X, n_iterations=10, sample_size=21000, k_range=range(2, 15), random_seed=42):
    """
    Run K-means with different k values across multiple iterations to find optimal k.
    """
    print("\n" + "=" * 80)
    print("RUN 1: FINDING OPTIMAL K (10 iterations, k=2..14)")
    print("=" * 80)
    
    # Store results
    all_silhouette_scores = {k: [] for k in k_range}
    all_ari_scores = {k: [] for k in k_range}
    all_nmi_scores = {k: [] for k in k_range}
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}:")
        
        # Get sample
        X_scaled, sample_indices = get_sample_data(X, sample_size, random_seed, iteration)
        true_labels_sample = [true_labels[i] for i in sample_indices]
        
        for k in k_range:
            # Run K-means
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, 
                           random_state=random_seed + iteration + k, algorithm='lloyd')
            labels = kmeans.fit_predict(X_scaled)
            
            # Compute metrics
            metrics = compute_cluster_metrics(X_scaled, labels, true_labels_sample, k)
            
            if not np.isnan(metrics['ari']):
                all_ari_scores[k].append(metrics['ari'])
            if not np.isnan(metrics['nmi']):
                all_nmi_scores[k].append(metrics['nmi'])
            if not np.isnan(metrics['silhouette']):
                all_silhouette_scores[k].append(metrics['silhouette'])
        
        # Print progress
        progress = [f"k={k}: ARI={np.mean(all_ari_scores[k]):.3f}" if all_ari_scores[k] else f"k={k}: N/A" 
                   for k in k_range[:3]]
        print(f"  Progress: {', '.join(progress)}...")
    
    # Compute means and stds
    results = {
        'k_range': list(k_range),
        'ari_mean': [np.mean(all_ari_scores[k]) if all_ari_scores[k] else np.nan for k in k_range],
        'ari_std': [np.std(all_ari_scores[k]) if all_ari_scores[k] else np.nan for k in k_range],
        'nmi_mean': [np.mean(all_nmi_scores[k]) if all_nmi_scores[k] else np.nan for k in k_range],
        'nmi_std': [np.std(all_nmi_scores[k]) if all_nmi_scores[k] else np.nan for k in k_range],
        'silhouette_mean': [np.mean(all_silhouette_scores[k]) if all_silhouette_scores[k] else np.nan for k in k_range],
        'silhouette_std': [np.std(all_silhouette_scores[k]) if all_silhouette_scores[k] else np.nan for k in k_range],
        'all_ari': all_ari_scores,
        'all_nmi': all_nmi_scores,
        'all_silhouette': all_silhouette_scores
    }
    
    # Find optimal k
    valid_indices = [i for i, v in enumerate(results['ari_mean']) if not np.isnan(v)]
    if valid_indices:
        best_idx = valid_indices[np.argmax([results['ari_mean'][i] for i in valid_indices])]
        results['best_k'] = results['k_range'][best_idx]
        results['best_k_ari'] = results['ari_mean'][best_idx]
    else:
        results['best_k'] = 7
        results['best_k_ari'] = np.nan
    
    print(f"\nOptimal K: {results['best_k']} (mean ARI: {results['best_k_ari']:.4f})")
    
    return results


# ========================================================
# Run 2: Fixed K=7 Stability Analysis
# ========================================================

def run_fixed_k_stability(X, k_fixed=7, n_iterations=10, sample_size=21000, random_seed=42):
    """
    Run K-means with fixed k across multiple iterations for stability analysis.
    """
    print("\n" + "=" * 80)
    print(f"RUN 2: FIXED K={k_fixed} STABILITY ANALYSIS (10 iterations)")
    print("=" * 80)
    
    results = {
        'labels': [],
        'centroids': [],
        'inertia': [],
        'cluster_sizes': [],
        'sample_indices': [],
        'ari': [],
        'nmi': [],
        'silhouette': []
    }
    
    print(f"\nRunning {n_iterations} iterations of K-means (k={k_fixed}) on {sample_size:,}-instance samples...")
    print("-" * 80)
    
    for iteration in range(n_iterations):
        # Get sample
        X_scaled, sample_indices = get_sample_data(X, sample_size, random_seed, iteration)
        true_labels_sample = [true_labels[i] for i in sample_indices]
        
        # Run K-means
        kmeans = KMeans(n_clusters=k_fixed, n_init=10, max_iter=300,
                       random_state=random_seed + iteration, algorithm='lloyd')
        kmeans.fit(X_scaled)
        labels = kmeans.labels_
        
        # Store results
        results['labels'].append(labels)
        results['centroids'].append(kmeans.cluster_centers_)
        results['inertia'].append(kmeans.inertia_)
        results['sample_indices'].append(sample_indices)
        
        # Cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        results['cluster_sizes'].append(cluster_sizes)
        
        # Compute metrics
        metrics = compute_cluster_metrics(X_scaled, labels, true_labels_sample, k_fixed)
        results['ari'].append(metrics['ari'])
        results['nmi'].append(metrics['nmi'])
        results['silhouette'].append(metrics['silhouette'])
        
        print(f"Iter {iteration+1:2d}: ARI={metrics['ari']:.4f}, NMI={metrics['nmi']:.4f}, "
              f"Silhouette={metrics['silhouette']:.4f}, Sizes={dict(cluster_sizes)}")
    
    print("-" * 80)
    
    # Compute stability metrics
    stability = compute_stability_metrics(results)
    results.update(stability)
    
    return results


def compute_stability_metrics(results):
    """Compute detailed stability metrics."""
    n_iter = len(results['centroids'])
    k_fixed = results['centroids'][0].shape[0]
    
    # Centroid similarity
    centroid_similarities = []
    for i in range(n_iter):
        for j in range(i+1, n_iter):
            dist_matrix = cdist(results['centroids'][i], results['centroids'][j], 'euclidean')
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            mean_dist = np.mean(dist_matrix[row_ind, col_ind])
            centroid_similarities.append(1 / (1 + mean_dist))
    
    # Cluster size stability (CV)
    sizes_matrix = np.array([list(dict(sizes).values()) for sizes in results['cluster_sizes']])
    if sizes_matrix.shape[1] < k_fixed:
        padded = np.zeros((sizes_matrix.shape[0], k_fixed))
        padded[:, :sizes_matrix.shape[1]] = sizes_matrix
        sizes_matrix = padded
    
    cv_per_cluster = sizes_matrix.std(axis=0) / (sizes_matrix.mean(axis=0) + 1e-10)
    
    # ARI stability (pairwise agreement between iterations)
    from sklearn.metrics import adjusted_rand_score
    ari_pairwise = []
    for i in range(n_iter):
        for j in range(i+1, n_iter):
            ari = adjusted_rand_score(results['labels'][i], results['labels'][j])
            ari_pairwise.append(ari)
    
    return {
        'centroid_similarity_mean': np.mean(centroid_similarities),
        'centroid_similarity_std': np.std(centroid_similarities),
        'cv_per_cluster': cv_per_cluster,
        'mean_cv': np.mean(cv_per_cluster),
        'ari_pairwise_mean': np.mean(ari_pairwise),
        'ari_pairwise_std': np.std(ari_pairwise)
    }


# ========================================================
# Plotting Functions
# ========================================================

def plot_optimal_k_results(opt_results, save_path=None):
    """Plot optimal k analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    k_range = opt_results['k_range']
    
    # ARI
    axes[0].plot(k_range, opt_results['ari_mean'], 'o-', color='royalblue', linewidth=2)
    axes[0].fill_between(k_range, 
                        np.array(opt_results['ari_mean']) - np.array(opt_results['ari_std']),
                        np.array(opt_results['ari_mean']) + np.array(opt_results['ari_std']),
                        alpha=0.2, color='royalblue')
    axes[0].axvline(x=opt_results['best_k'], color='red', linestyle='--', 
                    label=f'Best K={opt_results["best_k"]}')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Adjusted Rand Index (ARI)')
    axes[0].set_title('ARI vs. Number of Clusters')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # NMI
    axes[1].plot(k_range, opt_results['nmi_mean'], 's-', color='forestgreen', linewidth=2)
    axes[1].fill_between(k_range,
                        np.array(opt_results['nmi_mean']) - np.array(opt_results['nmi_std']),
                        np.array(opt_results['nmi_mean']) + np.array(opt_results['nmi_std']),
                        alpha=0.2, color='forestgreen')
    axes[1].axvline(x=opt_results['best_k'], color='red', linestyle='--')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Normalized Mutual Information (NMI)')
    axes[1].set_title('NMI vs. Number of Clusters')
    axes[1].grid(True, alpha=0.3)
    
    # Silhouette
    axes[2].plot(k_range, opt_results['silhouette_mean'], '^-', color='darkorange', linewidth=2)
    axes[2].fill_between(k_range,
                        np.array(opt_results['silhouette_mean']) - np.array(opt_results['silhouette_std']),
                        np.array(opt_results['silhouette_mean']) + np.array(opt_results['silhouette_std']),
                        alpha=0.2, color='darkorange')
    axes[2].axvline(x=opt_results['best_k'], color='red', linestyle='--')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Silhouette Score')
    axes[2].set_title('Silhouette Score vs. Number of Clusters')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Method B: Optimal K Analysis (10 iterations)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    return fig


def plot_fixed_k_stability(stab_results, k_fixed=7, save_path=None):
    """Plot fixed k stability analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    n_iter = len(stab_results['inertia'])
    iterations = range(1, n_iter + 1)
    
    # 1. ARI per iteration
    axes[0, 0].plot(iterations, stab_results['ari'], 'o-', color='royalblue', linewidth=2)
    axes[0, 0].axhline(y=np.mean(stab_results['ari']), color='red', linestyle='--',
                       label=f"Mean: {np.mean(stab_results['ari']):.4f}")
    axes[0, 0].fill_between(iterations,
                           np.mean(stab_results['ari']) - np.std(stab_results['ari']),
                           np.mean(stab_results['ari']) + np.std(stab_results['ari']),
                           alpha=0.2, color='red')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Adjusted Rand Index (ARI)')
    axes[0, 0].set_title(f'ARI Across Iterations (k={k_fixed})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. NMI per iteration
    axes[0, 1].plot(iterations, stab_results['nmi'], 's-', color='forestgreen', linewidth=2)
    axes[0, 1].axhline(y=np.mean(stab_results['nmi']), color='red', linestyle='--',
                       label=f"Mean: {np.mean(stab_results['nmi']):.4f}")
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Normalized Mutual Information (NMI)')
    axes[0, 1].set_title(f'NMI Across Iterations (k={k_fixed})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Silhouette per iteration
    axes[0, 2].plot(iterations, stab_results['silhouette'], '^-', color='darkorange', linewidth=2)
    valid_sil = [s for s in stab_results['silhouette'] if not np.isnan(s)]
    if valid_sil:
        axes[0, 2].axhline(y=np.mean(valid_sil), color='red', linestyle='--',
                           label=f"Mean: {np.mean(valid_sil):.4f}")
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Silhouette Score')
    axes[0, 2].set_title(f'Silhouette Score Across Iterations (k={k_fixed})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cluster sizes heatmap
    sizes_matrix = np.array([list(dict(sizes).values()) for sizes in stab_results['cluster_sizes']])
    if sizes_matrix.shape[1] < k_fixed:
        padded = np.zeros((sizes_matrix.shape[0], k_fixed))
        padded[:, :sizes_matrix.shape[1]] = sizes_matrix
        sizes_matrix = padded
    
    im = axes[1, 0].imshow(sizes_matrix, aspect='auto', cmap='Blues')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Iteration')
    axes[1, 0].set_title('Cluster Sizes Heatmap')
    plt.colorbar(im, ax=axes[1, 0], label='Size')
    for i in range(sizes_matrix.shape[0]):
        for j in range(sizes_matrix.shape[1]):
            axes[1, 0].text(j, i, f'{int(sizes_matrix[i, j])}', 
                           ha='center', va='center', fontsize=7,
                           color='white' if sizes_matrix[i, j] > sizes_matrix.max()/2 else 'black')
    
    # 5. Cluster size distributions (boxplot)
    axes[1, 1].boxplot(sizes_matrix.T)
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Size')
    axes[1, 1].set_title('Cluster Size Distributions Across Iterations')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Stability metrics summary
    metrics = [
        ('Centroid\nSimilarity', stab_results['centroid_similarity_mean']),
        ('Pairwise\nARI', stab_results['ari_pairwise_mean']),
        ('Cluster Size\nCV', stab_results['mean_cv'])
    ]
    names, values = zip(*metrics)
    colors = ['royalblue' if v > 0.7 else 'orange' if v > 0.4 else 'red' 
              for v in values]
    bars = axes[1, 2].bar(names, values, color=colors)
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Stability Metrics Summary')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add threshold lines
    axes[1, 2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    axes[1, 2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    axes[1, 2].legend()
    
    plt.suptitle(f'Method B: Fixed K={k_fixed} Stability Analysis (10 iterations)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    return fig


# ========================================================
# Main Execution
# ========================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("METHOD B: ENHANCED STABILITY ASSESSMENT")
    print("=" * 80)
    print(f"Dataset size: {len(X):,} instances")
    print(f"Sample size: 21,000 instances")
    print(f"Number of iterations: 10")
    print("=" * 80)
    
    # ====================================================
    # RUN 1: FINDING OPTIMAL K
    # ====================================================
    opt_results = run_optimal_k_analysis(
        X, 
        n_iterations=10, 
        sample_size=21000, 
        k_range=range(2, 15), 
        random_seed=42
    )
    
    # Save optimal K figure
    fig1 = plot_optimal_k_results(
        opt_results, 
        save_path=os.path.join(FIGURES_DIR, 'method_B_optimal_k.png')
    )
    
    # ====================================================
    # RUN 2: FIXED K=7 STABILITY ANALYSIS
    # ====================================================
    stab_results = run_fixed_k_stability(
        X, 
        k_fixed=7, 
        n_iterations=10, 
        sample_size=21000, 
        random_seed=42
    )
    
    # Save fixed K figure
    fig2 = plot_fixed_k_stability(
        stab_results, 
        k_fixed=7,
        save_path=os.path.join(FIGURES_DIR, 'method_B_fixed_k_stability.png')
    )
    
    # ====================================================
    # PRINT FINAL SUMMARY
    # ====================================================
    print("\n" + "=" * 80)
    print("METHOD B: FINAL SUMMARY")
    print("=" * 80)
    
    print("\n--- RUN 1: OPTIMAL K ---")
    print(f"  Best K: {opt_results['best_k']}")
    print(f"  Best ARI: {opt_results['best_k_ari']:.4f}")
    
    print("\n--- RUN 2: FIXED K=7 STABILITY ---")
    print(f"  Mean ARI: {np.mean(stab_results['ari']):.4f} ± {np.std(stab_results['ari']):.4f}")
    print(f"  Mean NMI: {np.mean(stab_results['nmi']):.4f} ± {np.std(stab_results['nmi']):.4f}")
    print(f"  Mean Silhouette: {np.nanmean(stab_results['silhouette']):.4f} ± {np.nanstd(stab_results['silhouette']):.4f}")
    print(f"  Centroid Similarity: {stab_results['centroid_similarity_mean']:.4f} ± {stab_results['centroid_similarity_std']:.4f}")
    print(f"  Pairwise ARI (stability): {stab_results['ari_pairwise_mean']:.4f} ± {stab_results['ari_pairwise_std']:.4f}")
    print(f"  Cluster Size CV: {stab_results['mean_cv']:.4f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    
    if opt_results['best_k'] == 7:
        print(f"✓ Optimal K = 7 matches the true number of emotion labels")
    else:
        print(f"⚠ Optimal K = {opt_results['best_k']} (true labels: 7)")
    
    if np.mean(stab_results['ari']) < 0.05:
        print("✗ Low ARI - poor agreement with ground truth labels")
    elif np.mean(stab_results['ari']) < 0.2:
        print("◯ Moderate ARI - some agreement with ground truth")
    else:
        print("✓ High ARI - good agreement with ground truth")
    
    if stab_results['centroid_similarity_mean'] < 0.5:
        print("✗ Low centroid stability - cluster centers are inconsistent across samples")
    elif stab_results['centroid_similarity_mean'] < 0.7:
        print("◯ Moderate centroid stability - some consistency in cluster positions")
    else:
        print("✓ High centroid stability - cluster centers are consistent")
    
    if stab_results['mean_cv'] > 1.5:
        print(f"✗ High cluster size variation (CV={stab_results['mean_cv']:.3f}) - cluster sizes are unstable")
        print("  This suggests the data may not have 7 well-separated, equally-sized clusters.")
    elif stab_results['mean_cv'] > 0.8:
        print(f"◯ Moderate cluster size variation (CV={stab_results['mean_cv']:.3f})")
    else:
        print(f"✓ Cluster sizes are stable (CV={stab_results['mean_cv']:.3f})")
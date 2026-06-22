# method_D.py
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from scipy.sparse import vstack
import warnings
warnings.filterwarnings('ignore')

from utils import vectorize_text_tfidf

DATASET_PATH = '../data/processed/dataset_7labels_clean.csv'
MIN_DF = 3
MAX_DF = 0.85
MAX_FEATURES = 5000

# =============================================================================
# METHOD D: PURPOSEULLY BALANCED SAMPLING (3,000 per label = 21,000 total)
# =============================================================================

def method_d_balanced_sampling(df, samples_per_label=3000, random_seed=42):
    """
    Method D: Purposefully balanced sampling
    Exactly samples_per_label drawn from each of the 7 emotion labels.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Drop rows with NaN in clean_text_transf before sampling
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    print(f"Dataset size after dropping NaN: {len(df_clean)} (dropped {len(df) - len(df_clean)} rows)")
    
    unique_labels = df_clean['label'].unique()
    print(f"Found {len(unique_labels)} emotion labels: {unique_labels}")
    
    balanced_samples = []
    
    for label in unique_labels:
        label_data = df_clean[df_clean['label'] == label]
        
        if len(label_data) == 0:
            print(f"  Warning: No samples for label '{label}'")
            continue
        
        if len(label_data) < samples_per_label:
            print(f"  Warning: Label '{label}' has only {len(label_data)} samples (< {samples_per_label})")
            # Sample with replacement to reach target
            sampled = label_data.sample(n=samples_per_label, replace=True, random_state=random_seed)
        else:
            sampled = label_data.sample(n=samples_per_label, random_state=random_seed)
        
        balanced_samples.append(sampled)
        print(f"  {label}: {len(sampled)} samples")
    
    df_balanced = pd.concat(balanced_samples, ignore_index=True)
    
    print(f"\nTotal balanced sample size: {len(df_balanced)}")
    print("Class distribution in balanced sample:")
    for label, count in df_balanced['label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(df_balanced)*100:.1f}%)")
    
    return df_balanced


# =============================================================================
# TEST 1: CENTROID STABILITY ANALYSIS
# =============================================================================

def test_centroid_stability(df, samples_per_label=3000, n_runs=5, n_clusters=7, random_seed=42):
    """
    Test 1: Centroid stability across multiple balanced samples.
    Uses a fixed vectorizer on a combined corpus to ensure consistent feature space.
    """
    print("\n" + "="*60)
    print("TEST 1: CENTROID STABILITY ANALYSIS")
    print("="*60)
    
    # Step 1: Clean data first
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    
    # Step 2: Create a combined corpus from all balanced samples
    print("\nCreating combined corpus for consistent feature space...")
    all_texts = []
    
    for run in range(n_runs):
        run_seed = run * 42
        df_balanced = method_d_balanced_sampling(df_clean, samples_per_label, run_seed)
        all_texts.extend(df_balanced['clean_text_transf'].tolist())
    
    # Fit vectorizer on combined corpus
    from sklearn.feature_extraction.text import TfidfVectorizer
    fixed_vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=MAX_FEATURES, 
        min_df=MIN_DF, 
        max_df=MAX_DF
    )
    fixed_vectorizer.fit(all_texts)
    print(f"Fixed vectorizer vocabulary size: {len(fixed_vectorizer.get_feature_names_out())}")
    
    all_centroids = []
    silhouette_scores = []
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}:")
        run_seed = run * 42
        
        # Get balanced sample
        df_balanced = method_d_balanced_sampling(df_clean, samples_per_label, run_seed)
        
        # Vectorize using fixed vectorizer (consistent feature space)
        X, _, data = vectorize_text_tfidf(df_balanced, precomputed_vectorizer=fixed_vectorizer)
        
        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=run_seed, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Store centroids
        all_centroids.append(kmeans.cluster_centers_)
        silhouette_scores.append(silhouette_score(X, labels))
        
        print(f"  Silhouette score: {silhouette_scores[-1]:.4f}")
    
    # Now all centroids have the same shape -> can be stacked
    all_centroids_array = np.stack(all_centroids)  # Shape: (n_runs, n_clusters, n_features)
    
    # Calculate centroid stability
    centroid_stability = []
    for cluster_idx in range(n_clusters):
        all_centroids_for_cluster = all_centroids_array[:, cluster_idx, :]
        
        pairwise_distances = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                dist = np.linalg.norm(all_centroids_for_cluster[i] - all_centroids_for_cluster[j])
                pairwise_distances.append(dist)
        
        centroid_stability.append(np.mean(pairwise_distances))
    
    results = {
        'centroid_stability': centroid_stability,
        'mean_centroid_distance': np.mean(centroid_stability),
        'std_centroid_distance': np.std(centroid_stability),
        'silhouette_scores': silhouette_scores,
        'mean_silhouette': np.mean(silhouette_scores),
        'std_silhouette': np.std(silhouette_scores)
    }
    
    print(f"\nCentroid stability (mean pairwise distance across runs):")
    for i, dist in enumerate(centroid_stability):
        print(f"  Cluster {i}: {dist:.4f}")
    print(f"  Overall mean: {results['mean_centroid_distance']:.4f} ± {results['std_centroid_distance']:.4f}")
    print(f"  Mean silhouette: {results['mean_silhouette']:.4f} ± {results['std_silhouette']:.4f}")
    
    return results


# =============================================================================
# TEST 2: LEARNING CURVES
# =============================================================================

def test_learning_curves(df, sample_sizes=[500, 1000, 1500, 2000, 2500, 3000], 
                         n_clusters=7, random_seed=42):
    """
    Test 2: Learning curves - performance with increasing samples per label.
    Uses consistent feature space across all sample sizes.
    """
    print("\n" + "="*60)
    print("TEST 2: LEARNING CURVES")
    print("="*60)
    
    # Clean data first
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    
    # Step 1: Create combined corpus from all sample sizes for consistent feature space
    print("\nCreating combined corpus for consistent feature space...")
    all_texts = []
    
    for size in sample_sizes:
        df_balanced = method_d_balanced_sampling(df_clean, size, random_seed)
        all_texts.extend(df_balanced['clean_text_transf'].tolist())
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    fixed_vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=MAX_FEATURES, 
        min_df=MIN_DF, 
        max_df=MAX_DF
    )
    fixed_vectorizer.fit(all_texts)
    print(f"Fixed vectorizer vocabulary size: {len(fixed_vectorizer.get_feature_names_out())}")
    
    results = []
    
    for size in sample_sizes:
        print(f"\n---Samples per label: {size}---")
        
        # Get balanced sample
        df_balanced = method_d_balanced_sampling(df_clean, size, random_seed)
        
        # Vectorize using fixed vectorizer
        X, _, data = vectorize_text_tfidf(df_balanced, precomputed_vectorizer=fixed_vectorizer)
        
        true_labels = data['label'].tolist()
        
        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        
        # Compute metrics
        silhouette = silhouette_score(X, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        results.append({
            'samples_per_label': size,
            'total_samples': size * 7,
            'silhouette': silhouette,
            'ari': ari,
            'nmi': nmi
        })
        
        print(f"  ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {silhouette:.4f}")
    
    ari_values = [r['ari'] for r in results]
    improvements = [ari_values[i] - ari_values[i-1] for i in range(1, len(ari_values))]
    
    final = {
        'results': results,
        'ari_values': ari_values,
        'improvements': improvements,
        'convergence_threshold': results[-1]['ari'] - results[-2]['ari'] if len(results) >= 2 else None
    }
    
    print(f"\nLearning curve summary:")
    print(f"  ARI at {sample_sizes[0]}: {ari_values[0]:.4f}")
    print(f"  ARI at {sample_sizes[-1]}: {ari_values[-1]:.4f}")
    print(f"  Total improvement: {ari_values[-1] - ari_values[0]:.4f}")
    if final['convergence_threshold'] is not None:
        print(f"  Improvement from {sample_sizes[-2]} to {sample_sizes[-1]}: {final['convergence_threshold']:.4f}")
    
    return final


# =============================================================================
# TEST 3: OUT-OF-SAMPLE VALIDATION
# =============================================================================

def test_out_of_sample_validation(df, samples_per_label=3000, test_size=10000, 
                                  n_clusters=7, random_seed=42):
    """
    Test 3: Out-of-sample validation - train on balanced sample, test on unseen data.
    """
    print("\n" + "="*60)
    print("TEST 3: OUT-OF-SAMPLE VALIDATION")
    print("="*60)
    
    # Clean data first
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    print(f"Dataset size after cleaning: {len(df_clean)}")
    
    # Step 1: Get balanced training sample
    print(f"\nGenerating balanced training sample ({samples_per_label} per label)...")
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    train_samples = []
    unique_labels = df_clean['label'].unique()
    
    for label in unique_labels:
        label_data = df_clean[df_clean['label'] == label]
        n_samples = min(samples_per_label, len(label_data))
        sampled = label_data.sample(n=n_samples, random_state=random_seed)
        train_samples.append(sampled)
        print(f"  {label}: {len(sampled)} samples")
    
    df_train = pd.concat(train_samples, ignore_index=True)
    train_indices = set(df_train.index)
    
    # Vectorize training data
    X_train, vectorizer, train_data = vectorize_text_tfidf(df_train)
    y_train_true = train_data['label'].tolist()
    
    # Train K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    y_train_pred = kmeans.fit_predict(X_train)
    
    train_ari = adjusted_rand_score(y_train_true, y_train_pred)
    train_nmi = normalized_mutual_info_score(y_train_true, y_train_pred)
    train_sil = silhouette_score(X_train, y_train_pred)
    
    print(f"\nTraining metrics (balanced {samples_per_label} per label):")
    print(f"  ARI: {train_ari:.4f}")
    print(f"  NMI: {train_nmi:.4f}")
    print(f"  Silhouette: {train_sil:.4f}")
    
    # Step 2: Get unseen test samples
    print(f"\nGetting {test_size} unseen test samples...")
    
    remaining_indices = [i for i in df_clean.index if i not in train_indices]
    
    if len(remaining_indices) < test_size:
        test_indices = remaining_indices
        print(f"  Warning: Only {len(remaining_indices)} samples available for testing")
    else:
        test_indices = np.random.choice(remaining_indices, size=test_size, replace=False)
    
    df_test = df_clean.loc[test_indices].reset_index(drop=True)
    
    test_selected = df_test[['sample_id', 'original_id', 'clean_text_transf', 'label', 'method']].copy()
    test_selected = test_selected.dropna(subset=['clean_text_transf'])
    test_selected['clean_text_transf'] = test_selected['clean_text_transf'].astype(str)
    X_test = vectorizer.transform(test_selected['clean_text_transf'])
    y_test_true = test_selected['label'].tolist()
    
    print(f"  Test set size: {X_test.shape[0]}")
    
    # Step 3: Predict on test set
    y_test_pred = kmeans.predict(X_test)
    
    test_ari = adjusted_rand_score(y_test_true, y_test_pred)
    test_nmi = normalized_mutual_info_score(y_test_true, y_test_pred)
    test_sil = silhouette_score(X_test, y_test_pred)
    
    print(f"\nTest metrics (unseen data):")
    print(f"  ARI: {test_ari:.4f}")
    print(f"  NMI: {test_nmi:.4f}")
    print(f"  Silhouette: {test_sil:.4f}")
    
    gap_ari = train_ari - test_ari
    gap_nmi = train_nmi - test_nmi
    gap_sil = train_sil - test_sil
    
    print(f"\nTrain-Test Gap:")
    print(f"  ARI gap: {gap_ari:.4f}")
    print(f"  NMI gap: {gap_nmi:.4f}")
    print(f"  Silhouette gap: {gap_sil:.4f}")
    
    viable = gap_ari < 0.05
    print(f"\nScalability Viability: {'✅ VIABLE' if viable else '⚠️ NEEDS MORE DATA'}")
    print(f"  (Gap < 0.05 indicates strong generalization)")
    
    return {
        'train': {'ari': train_ari, 'nmi': train_nmi, 'silhouette': train_sil},
        'test': {'ari': test_ari, 'nmi': test_nmi, 'silhouette': test_sil},
        'gap': {'ari': gap_ari, 'nmi': gap_nmi, 'silhouette': gap_sil},
        'viable': viable,
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0]
    }


# =============================================================================
# METHOD D: COMPLETE EXECUTION
# =============================================================================

def run_method_d_full_analysis(df, samples_per_label=3000, n_clusters=7):
    """
    Run complete Method D analysis including all three scalability tests.
    """
    print("="*60)
    print("METHOD D: PURPOSEULLY BALANCED SAMPLING WITH SCALABILITY ANALYSIS")
    print("="*60)
    print(f"Samples per label: {samples_per_label}")
    print(f"Total training samples: {samples_per_label * 7}")
    print(f"Number of clusters: {n_clusters}")
    print("="*60)
    
    # Clean data first
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    print(f"\nDataset size after cleaning: {len(df_clean)}")
    
    print("\nGenerating final balanced model...")
    df_balanced = method_d_balanced_sampling(df_clean, samples_per_label, 42)
    X_final, vectorizer, data_final = vectorize_text_tfidf(df_balanced)
    y_final_true = data_final['label'].tolist()
    
    final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_final_pred = final_kmeans.fit_predict(X_final)
    
    final_ari = adjusted_rand_score(y_final_true, y_final_pred)
    final_nmi = normalized_mutual_info_score(y_final_true, y_final_pred)
    final_sil = silhouette_score(X_final, y_final_pred)
    
    print(f"\nFinal balanced model metrics:")
    print(f"  ARI: {final_ari:.4f}")
    print(f"  NMI: {final_nmi:.4f}")
    print(f"  Silhouette: {final_sil:.4f}")
    
    results = {
        'final_balanced_model': {
            'ari': final_ari,
            'nmi': final_nmi,
            'silhouette': final_sil,
            'sample_size': X_final.shape[0]
        },
        'centroid_stability': test_centroid_stability(df_clean, samples_per_label, n_runs=5, n_clusters=n_clusters),
        'learning_curves': test_learning_curves(df_clean, n_clusters=n_clusters),
        'out_of_sample': test_out_of_sample_validation(df_clean, samples_per_label, test_size=10000, n_clusters=n_clusters)
    }
    
    print("\n" + "="*60)
    print("METHOD D SUMMARY")
    print("="*60)
    
    print("\n✅ Scalability Verdict:")
    if results['out_of_sample']['viable']:
        print("   VIABLE: Model generalizes well (train-test gap < 0.05)")
        print(f"   The balanced sample of {samples_per_label} instances per label is sufficient.")
        print(f"   Processing the full {len(df_clean):,} dataset would yield minimal improvement.")
    else:
        print("   WARNING: Model may need more data (train-test gap >= 0.05)")
        print(f"   Consider increasing samples per label beyond {samples_per_label}.")
    
    return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    dataset = pd.read_csv(DATASET_PATH)
    
    # Run Method D complete analysis
    results = run_method_d_full_analysis(dataset, samples_per_label=3000, n_clusters=7)
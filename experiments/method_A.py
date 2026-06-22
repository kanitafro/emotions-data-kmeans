# method_A.py
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

from utils import vectorize_text_tfidf

DATASET_PATH = '../data/processed/dataset_7labels_clean.csv'

# =============================================================================
# METHOD A: SINGLE RANDOM SAMPLE (21,000 instances)
# =============================================================================

def method_a_single_sample(df, sample_size=21000, random_seed=42):
    """
    Method A: Single random sample using reservoir sampling.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    
    print(f"Dataset size after cleaning: {len(df_clean)}")
    
    df_sample = df_clean.sample(n=sample_size, random_state=random_seed)
    
    print(f"\nSampled {len(df_sample)} instances")
    print("Class distribution in sample:")
    for label, count in df_sample['label'].value_counts().items():
        pct = count / len(df_sample) * 100
        full_pct = len(df_clean[df_clean['label'] == label]) / len(df_clean) * 100
        print(f"  {label}: {count} ({pct:.1f}%) [full: {full_pct:.1f}%]")
    
    return df_sample


def create_cluster_label_confusion(kmeans_labels, true_labels, emotion_labels):
    """
    Creates a confusion matrix showing cluster composition by true emotion labels.
    
    IMPORTANT: This properly handles the case where cluster labels don't align
    with true labels by showing percentages within each cluster.
    """
    n_clusters = len(np.unique(kmeans_labels))
    n_emotions = len(emotion_labels)
    
    # Initialize confusion matrix
    confusion = np.zeros((n_clusters, n_emotions))
    
    # Fill confusion matrix with raw counts
    for i, true_label in enumerate(emotion_labels):
        for j in range(n_clusters):
            mask = (kmeans_labels == j) & (np.array(true_labels) == true_label)
            confusion[j, i] = np.sum(mask)
    
    # Convert to percentages within each cluster (row-wise)
    row_sums = confusion.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_pct = confusion / row_sums * 100
    
    # Create DataFrame
    df_confusion = pd.DataFrame(
        confusion_pct,
        index=[f"Cluster {i}" for i in range(n_clusters)],
        columns=emotion_labels
    )
    
    return df_confusion, confusion


def evaluate_clustering(X, true_labels, predicted_labels, emotion_labels):
    """
    Evaluate clustering performance with multiple metrics.
    """
    # Compute metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    sil = silhouette_score(X, predicted_labels)
    
    # Create confusion matrix
    df_confusion, raw_confusion = create_cluster_label_confusion(
        predicted_labels, true_labels, emotion_labels
    )
    
    # For each cluster, find the dominant emotion and its purity
    cluster_purities = []
    for cluster_idx in range(len(df_confusion)):
        cluster_row = df_confusion.iloc[cluster_idx]
        dominant_emotion = cluster_row.idxmax()
        dominant_purity = cluster_row.max()
        cluster_purities.append({
            'cluster': cluster_idx,
            'dominant_emotion': dominant_emotion,
            'purity': dominant_purity
        })
    
    # Overall purity (macro average)
    overall_purity = np.mean([p['purity'] for p in cluster_purities])
    
    # Also compute per-emotion recall (how much of each emotion is captured)
    per_emotion_recall = {}
    for i, emotion in enumerate(emotion_labels):
        emotion_mask = np.array(true_labels) == emotion
        if np.sum(emotion_mask) > 0:
            # Find which cluster this emotion mostly falls into
            emotion_clusters = predicted_labels[emotion_mask]
            if len(emotion_clusters) > 0:
                most_common_cluster = np.bincount(emotion_clusters).argmax()
                recall = np.sum(emotion_clusters == most_common_cluster) / np.sum(emotion_mask)
                per_emotion_recall[emotion] = recall
    
    return {
        'ari': ari,
        'nmi': nmi,
        'silhouette': sil,
        'confusion_matrix': df_confusion,
        'raw_confusion': raw_confusion,
        'cluster_purities': cluster_purities,
        'overall_purity': overall_purity,
        'per_emotion_recall': per_emotion_recall
    }


def run_method_a(df, sample_size=21000, n_clusters=7, random_seed=42):
    """
    Run complete Method A analysis.
    """
    print("="*60)
    print("METHOD A: SINGLE RANDOM SAMPLE")
    print("="*60)
    print(f"Sample size: {sample_size}")
    print(f"Number of clusters: {n_clusters}")
    print("="*60)
    
    # Step 1: Get random sample
    print("\n" + "-"*40)
    print("STEP 1: SAMPLING")
    print("-"*40)
    df_sample = method_a_single_sample(df, sample_size, random_seed)
    
    # Step 2: Vectorize
    print("\n" + "-"*40)
    print("STEP 2: TF-IDF VECTORIZATION")
    print("-"*40)
    X, vectorizer, data = vectorize_text_tfidf(df_sample)
    
    # Step 3: Get true labels
    true_labels = data['label'].tolist()
    emotion_labels = sorted(data['label'].unique())
    print(f"Emotion labels: {emotion_labels}")
    
    # Step 4: Find optimal K (optional - can be fixed at 7)
    print("\n" + "-"*40)
    print("STEP 3: FINDING OPTIMAL K")
    print("-"*40)
    
    k_range = range(2, 15)
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal K (by silhouette): {best_k}")
    print(f"Using K = {n_clusters} (fixed)")
    
    # Step 5: Run K-means with fixed n_clusters
    print("\n" + "-"*40)
    print("STEP 4: K-MEANS CLUSTERING")
    print("-"*40)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    predicted_labels = kmeans.fit_predict(X)
    
    # Print cluster sizes
    print("Cluster sizes:")
    for i in range(n_clusters):
        print(f"  Cluster {i}: {np.sum(predicted_labels == i)} samples")
    
    # Step 6: Evaluate
    print("\n" + "-"*40)
    print("STEP 5: EVALUATION")
    print("-"*40)
    results = evaluate_clustering(X, true_labels, predicted_labels, emotion_labels)
    
    print(f"\nEvaluation Metrics:")
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    
    # Step 7: Display confusion matrix
    print("\n" + "-"*40)
    print("STEP 6: CLUSTER COMPOSITION (Confusion Matrix)")
    print("-"*40)
    print("Rows = Clusters, Columns = True Emotion Labels")
    print("Values = Percentage of samples in each cluster belonging to each emotion")
    print()
    print(results['confusion_matrix'].round(2).to_string())
    
    # Step 8: Show dominant emotion per cluster
    print("\n" + "-"*40)
    print("STEP 7: DOMINANT EMOTION PER CLUSTER")
    print("-"*40)
    for p in results['cluster_purities']:
        print(f"  Cluster {p['cluster']}: {p['dominant_emotion']} ({p['purity']:.1f}%)")
    
    # Step 9: Show per-emotion recall
    print("\n" + "-"*40)
    print("STEP 8: PER-EMOTION RECALL")
    print("-"*40)
    for emotion, recall in results['per_emotion_recall'].items():
        print(f"  {emotion}: {recall:.2%}")
    
    # Step 10: Full results
    full_results = {
        'sample_size': sample_size,
        'n_clusters': n_clusters,
        'sample': df_sample,
        'X': X,
        'vectorizer': vectorizer,
        'data': data,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'emotion_labels': emotion_labels,
        'kmeans_model': kmeans,
        'evaluation': results,
        'best_k': best_k
    }
    
    print("\n" + "="*60)
    print("METHOD A SUMMARY")
    print("="*60)
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    print(f"  Best K (silhouette): {best_k}")
    print(f"  Used K: {n_clusters}")
    
    return full_results


# =============================================================================
# ADDITIONAL: SILHOUETTE PLOT DATA
# =============================================================================

def get_silhouette_curve(df, sample_size=21000, k_range=range(2, 15), random_seed=42):
    """
    Get silhouette scores for different k values (for plotting).
    """
    df_sample = method_a_single_sample(df, sample_size, random_seed)
    X, _, _ = vectorize_text_tfidf(df_sample)
    
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    
    return list(k_range), silhouette_scores


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv(DATASET_PATH)
    
    # Run Method A
    results = run_method_a(
        df=dataset,
        sample_size=21000,
        n_clusters=7,
        random_seed=42
    )
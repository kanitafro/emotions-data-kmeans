# method_C.py
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
# METHOD C: STRATIFIED SAMPLING (Preserving Natural Class Distribution)
# =============================================================================

def method_c_stratified_sampling(df, sample_size=21000, random_seed=42):
    """
    Method C: Stratified sampling preserving natural class distribution.
    
    For each emotion label l, the number of samples drawn is proportional
    to its percentage in the full dataset:
        n_l = round(21000 * count_l / total)
    
    Args:
        df (pd.DataFrame): Full dataset with 'clean_text_transf' and 'label' columns
        sample_size (int): Total number of samples to draw (default=21000)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Stratified sample preserving natural class distribution
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Clean data first
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    
    print(f"Dataset size after cleaning: {len(df_clean)}")
    
    # Get class distribution
    total = len(df_clean)
    label_counts = df_clean['label'].value_counts()
    
    # Calculate proportional samples per label
    samples_per_label = {}
    for label, count in label_counts.items():
        n_l = int(round(sample_size * count / total))
        samples_per_label[label] = n_l
    
    # Adjust to ensure exact sample_size
    diff = sample_size - sum(samples_per_label.values())
    if diff > 0:
        # Add remaining samples to largest class
        largest_label = max(samples_per_label, key=samples_per_label.get)
        samples_per_label[largest_label] += diff
    elif diff < 0:
        # Remove from smallest class if overshoot
        smallest_label = min(samples_per_label, key=samples_per_label.get)
        samples_per_label[smallest_label] += diff  # diff is negative
    
    print("\nStratified sampling plan:")
    for label in sorted(samples_per_label.keys()):
        n_l = samples_per_label[label]
        full_pct = label_counts[label] / total * 100
        sample_pct = n_l / sample_size * 100
        print(f"  {label}: {n_l} samples ({sample_pct:.1f}%) [full: {full_pct:.1f}%]")
    
    # Draw samples from each label
    stratified_samples = []
    for label, n_l in samples_per_label.items():
        label_data = df_clean[df_clean['label'] == label]
        
        if len(label_data) < n_l:
            print(f"  Warning: Label '{label}' has only {len(label_data)} samples (< {n_l})")
            sampled = label_data.sample(n=min(n_l, len(label_data)), random_state=random_seed)
        else:
            sampled = label_data.sample(n=n_l, random_state=random_seed)
        
        stratified_samples.append(sampled)
    
    # Combine all samples
    df_sample = pd.concat(stratified_samples, ignore_index=True)
    
    print(f"\nTotal stratified sample size: {len(df_sample)}")
    print("Class distribution in stratified sample:")
    for label, count in df_sample['label'].value_counts().items():
        pct = count / len(df_sample) * 100
        full_pct = label_counts[label] / total * 100
        print(f"  {label}: {count} ({pct:.1f}%) [full: {full_pct:.1f}%]")
    
    return df_sample


def create_cluster_label_confusion(kmeans_labels, true_labels, emotion_labels):
    """
    Creates a confusion matrix showing cluster composition by true emotion labels.
    
    Args:
        kmeans_labels (array): Cluster assignments from K-means
        true_labels (array): Ground truth emotion labels
        emotion_labels (list): List of unique emotion labels
    
    Returns:
        pd.DataFrame: Confusion matrix (rows=clusters, columns=true labels)
    """
    n_clusters = len(np.unique(kmeans_labels))
    n_emotions = len(emotion_labels)
    
    # Initialize confusion matrix with counts
    confusion_counts = np.zeros((n_clusters, n_emotions))
    
    # Fill confusion matrix with raw counts
    for i, true_label in enumerate(emotion_labels):
        for j in range(n_clusters):
            mask = (kmeans_labels == j) & (np.array(true_labels) == true_label)
            confusion_counts[j, i] = np.sum(mask)
    
    # Convert to percentages within each cluster (row-wise)
    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_pct = confusion_counts / row_sums * 100
    
    # Create DataFrames
    df_confusion_pct = pd.DataFrame(
        confusion_pct,
        index=[f"Cluster {i}" for i in range(n_clusters)],
        columns=emotion_labels
    )
    
    df_confusion_counts = pd.DataFrame(
        confusion_counts,
        index=[f"Cluster {i}" for i in range(n_clusters)],
        columns=emotion_labels
    )
    
    return df_confusion_pct, df_confusion_counts


def evaluate_clustering(X, true_labels, predicted_labels, emotion_labels):
    """
    Evaluate clustering performance with multiple metrics.
    """
    # Compute metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    # Silhouette requires at least 2 clusters
    n_clusters = len(np.unique(predicted_labels))
    if n_clusters >= 2:
        sil = silhouette_score(X, predicted_labels)
    else:
        sil = np.nan
    
    # Create confusion matrices
    df_confusion_pct, df_confusion_counts = create_cluster_label_confusion(
        predicted_labels, true_labels, emotion_labels
    )
    
    # For each cluster, find the dominant emotion and its purity
    cluster_purities = []
    for cluster_idx in range(len(df_confusion_pct)):
        cluster_row = df_confusion_pct.iloc[cluster_idx]
        dominant_emotion = cluster_row.idxmax()
        dominant_purity = cluster_row.max()
        cluster_purities.append({
            'cluster': cluster_idx,
            'dominant_emotion': dominant_emotion,
            'purity': dominant_purity,
            'size': df_confusion_counts.iloc[cluster_idx].sum()
        })
    
    # Overall purity (macro average)
    overall_purity = np.mean([p['purity'] for p in cluster_purities])
    
    # Per-emotion recall (how much of each emotion is captured by its dominant cluster)
    per_emotion_recall = {}
    for emotion in emotion_labels:
        emotion_mask = np.array(true_labels) == emotion
        if np.sum(emotion_mask) > 0:
            emotion_clusters = predicted_labels[emotion_mask]
            if len(emotion_clusters) > 0:
                most_common_cluster = np.bincount(emotion_clusters).argmax()
                recall = np.sum(emotion_clusters == most_common_cluster) / np.sum(emotion_mask)
                per_emotion_recall[emotion] = recall
    
    return {
        'ari': ari,
        'nmi': nmi,
        'silhouette': sil,
        'confusion_pct': df_confusion_pct,
        'confusion_counts': df_confusion_counts,
        'cluster_purities': cluster_purities,
        'overall_purity': overall_purity,
        'per_emotion_recall': per_emotion_recall
    }


def run_method_c(df, sample_size=21000, n_clusters=7, random_seed=42):
    """
    Run complete Method C analysis.
    """
    print("="*60)
    print("METHOD C: STRATIFIED SAMPLING (Natural Class Distribution)")
    print("="*60)
    print(f"Sample size: {sample_size}")
    print(f"Number of clusters: {n_clusters}")
    print("="*60)
    
    # Step 1: Get stratified sample
    print("\n" + "-"*40)
    print("STEP 1: STRATIFIED SAMPLING")
    print("-"*40)
    df_sample = method_c_stratified_sampling(df, sample_size, random_seed)
    
    # Step 2: Vectorize
    print("\n" + "-"*40)
    print("STEP 2: TF-IDF VECTORIZATION")
    print("-"*40)
    X, vectorizer, data = vectorize_text_tfidf(df_sample)
    
    # Step 3: Get true labels
    true_labels = data['label'].tolist()
    emotion_labels = sorted(data['label'].unique())
    print(f"Emotion labels: {emotion_labels}")
    
    # Step 4: Find optimal K
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
    
    # Step 7: Display confusion matrix (percentages)
    print("\n" + "-"*40)
    print("STEP 6: CLUSTER COMPOSITION (Confusion Matrix - Percentages)")
    print("-"*40)
    print("Rows = Clusters, Columns = True Emotion Labels")
    print("Values = Percentage of samples in each cluster belonging to each emotion")
    print()
    print(results['confusion_pct'].round(2).to_string())
    
    # Step 8: Display confusion matrix (counts)
    print("\n" + "-"*40)
    print("STEP 7: CLUSTER COMPOSITION (Confusion Matrix - Counts)")
    print("-"*40)
    print("Rows = Clusters, Columns = True Emotion Labels")
    print("Values = Number of samples in each cluster belonging to each emotion")
    print()
    print(results['confusion_counts'].to_string())
    
    # Step 9: Show dominant emotion per cluster
    print("\n" + "-"*40)
    print("STEP 8: DOMINANT EMOTION PER CLUSTER")
    print("-"*40)
    for p in results['cluster_purities']:
        print(f"  Cluster {p['cluster']} (size={p['size']}): {p['dominant_emotion']} ({p['purity']:.1f}%)")
    
    # Step 10: Show per-emotion recall
    print("\n" + "-"*40)
    print("STEP 9: PER-EMOTION RECALL")
    print("-"*40)
    for emotion, recall in results['per_emotion_recall'].items():
        print(f"  {emotion}: {recall:.2%}")
    
    # Step 11: Full results
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
    print("METHOD C SUMMARY")
    print("="*60)
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    print(f"  Best K (silhouette): {best_k}")
    print(f"  Used K: {n_clusters}")
    
    return full_results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv(DATASET_PATH)
    
    # Run Method C
    results = run_method_c(
        df=dataset,
        sample_size=21000,
        n_clusters=7,
        random_seed=42
    )
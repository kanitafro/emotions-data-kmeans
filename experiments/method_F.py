# method_F.py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import umap
import warnings
import time
warnings.filterwarnings('ignore')

from utils import vectorize_text_tfidf

DATASET_PATH = '../data/processed/dataset_7labels_clean.csv'

def create_cluster_label_confusion(labels, true_labels, emotion_labels):
    """
    Creates a confusion matrix showing cluster composition by true emotion labels.
    """
    n_clusters = len(np.unique(labels))
    n_emotions = len(emotion_labels)
    
    confusion = np.zeros((n_clusters, n_emotions))
    
    for i, true_label in enumerate(emotion_labels):
        for j in range(n_clusters):
            mask = (labels == j) & (np.array(true_labels) == true_label)
            confusion[j, i] = np.sum(mask)
    
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_pct = confusion / row_sums * 100
    
    df_confusion = pd.DataFrame(
        confusion_pct,
        index=[f"Cluster {i}" for i in range(n_clusters)],
        columns=emotion_labels
    )
    
    return df_confusion


def evaluate_clustering(X, true_labels, predicted_labels, emotion_labels):
    """
    Evaluate clustering performance with multiple metrics.
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    # Silhouette: filter out noise points (-1)
    mask = predicted_labels != -1
    if np.sum(mask) > 0 and len(np.unique(predicted_labels[mask])) > 1:
        sil = silhouette_score(X[mask], predicted_labels[mask])
    else:
        sil = np.nan
    
    df_confusion_pct = create_cluster_label_confusion(
        predicted_labels, true_labels, emotion_labels
    )
    
    cluster_purities = []
    for cluster_idx in range(len(df_confusion_pct)):
        cluster_row = df_confusion_pct.iloc[cluster_idx]
        dominant_emotion = cluster_row.idxmax()
        dominant_purity = cluster_row.max()
        cluster_purities.append({
            'cluster': cluster_idx,
            'dominant_emotion': dominant_emotion,
            'purity': dominant_purity
        })
    
    overall_purity = np.mean([p['purity'] for p in cluster_purities])
    
    # Per-emotion recall
    per_emotion_recall = {}
    for emotion in emotion_labels:
        emotion_mask = np.array(true_labels) == emotion
        if np.sum(emotion_mask) > 0:
            emotion_clusters = predicted_labels[emotion_mask]
            # Exclude noise
            emotion_clusters = emotion_clusters[emotion_clusters != -1]
            if len(emotion_clusters) > 0:
                most_common_cluster = np.bincount(emotion_clusters).argmax()
                recall = np.sum(emotion_clusters == most_common_cluster) / np.sum(emotion_mask)
                per_emotion_recall[emotion] = recall
    
    return {
        'ari': ari,
        'nmi': nmi,
        'silhouette': sil,
        'confusion_pct': df_confusion_pct,
        'cluster_purities': cluster_purities,
        'overall_purity': overall_purity,
        'per_emotion_recall': per_emotion_recall
    }


def run_method_f(df, sample_size=21000, random_seed=42):
    """
    Run Method F: DBSCAN on TF-IDF features with UMAP dimensionality reduction.
    """
    start_time = time.time()
    
    print("="*60)
    print("METHOD F: DBSCAN + TF-IDF + UMAP")
    print("="*60)
    print(f"Sample size: {sample_size}")
    print("="*60)
    
    # Step 1: Sample data
    print("\n" + "-"*40)
    print("STEP 1: SAMPLING")
    print("-"*40)
    df_clean = df.dropna(subset=['clean_text_transf']).copy()
    df_clean['clean_text_transf'] = df_clean['clean_text_transf'].astype(str)
    df_sample = df_clean.sample(n=sample_size, random_state=random_seed)
    
    print("Class distribution in sample:")
    for label, count in df_sample['label'].value_counts().items():
        pct = count / len(df_sample) * 100
        full_pct = len(df_clean[df_clean['label'] == label]) / len(df_clean) * 100
        print(f"  {label}: {count} ({pct:.1f}%) [full: {full_pct:.1f}%]")
    
    # Step 2: TF-IDF vectorization
    print("\n" + "-"*40)
    print("STEP 2: TF-IDF VECTORIZATION")
    print("-"*40)
    X, vectorizer, data = vectorize_text_tfidf(df_sample)
    
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)
    
    print(f"TF-IDF shape: {X_dense.shape}")
    
    # Step 3: Dimensionality reduction with UMAP
    print("\n" + "-"*40)
    print("STEP 3: UMAP DIMENSIONALITY REDUCTION")
    print("-"*40)
    reducer = umap.UMAP(n_components=50, random_state=random_seed, n_neighbors=15)
    X_umap = reducer.fit_transform(X_dense)
    print(f"UMAP reduced shape: {X_umap.shape}")
    
    # Step 4: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_umap)
    
    # Step 5: Get true labels
    true_labels = data['label'].tolist()
    emotion_labels = sorted(data['label'].unique())
    print(f"Emotion labels: {emotion_labels}")
    
    # Step 6: Find optimal DBSCAN parameters
    print("\n" + "-"*40)
    print("STEP 4: DBSCAN PARAMETER TUNING")
    print("-"*40)
    
    best_ari = -1
    best_eps = 0.5
    best_min_samples = 5
    best_labels = None
    best_n_clusters = 0
    
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for min_samples in [3, 5, 10]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                ari = adjusted_rand_score(true_labels, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
                    best_n_clusters = n_clusters
    
    print(f"Best eps: {best_eps}")
    print(f"Best min_samples: {best_min_samples}")
    print(f"Number of clusters: {best_n_clusters}")
    print(f"Noise points: {np.sum(best_labels == -1)}")
    
    # Step 7: Evaluate
    print("\n" + "-"*40)
    print("STEP 5: EVALUATION")
    print("-"*40)
    results = evaluate_clustering(X_scaled, true_labels, best_labels, emotion_labels)
    
    print(f"\nEvaluation Metrics:")
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    
    # Step 8: Confusion matrix
    print("\n" + "-"*40)
    print("STEP 6: CLUSTER COMPOSITION (Confusion Matrix - Percentages)")
    print("-"*40)
    print("Rows = Clusters, Columns = True Emotion Labels")
    print("(Noise points excluded)")
    print()
    print(results['confusion_pct'].round(2).to_string())
    
    # Step 9: Dominant emotion per cluster
    print("\n" + "-"*40)
    print("STEP 7: DOMINANT EMOTION PER CLUSTER")
    print("-"*40)
    for p in results['cluster_purities']:
        print(f"  Cluster {p['cluster']}: {p['dominant_emotion']} ({p['purity']:.1f}%)")
    
    # Step 10: Per-emotion recall
    print("\n" + "-"*40)
    print("STEP 8: PER-EMOTION RECALL")
    print("-"*40)
    for emotion, recall in results['per_emotion_recall'].items():
        print(f"  {emotion}: {recall:.2%}")
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("METHOD F SUMMARY")
    print("="*60)
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    print(f"  Best eps: {best_eps}")
    print(f"  Best min_samples: {best_min_samples}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    
    return results


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(dataset)} instances")
    
    results = run_method_f(
        df=dataset,
        sample_size=21000,
        random_seed=42
    )
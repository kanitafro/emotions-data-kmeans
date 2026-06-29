# method_E.py
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import warnings
import time
warnings.filterwarnings('ignore')

DATASET_PATH = '../data/processed/dataset_7labels_clean.csv'


def load_word2vec_model():
    """Load pre-trained Word2Vec model."""
    print("Loading Word2Vec model (this may take a few minutes)...")
    model = api.load("word2vec-google-news-300")
    print("Model loaded successfully!")
    return model


def get_sentence_embedding(text, model, embedding_dim=300):
    """
    Get sentence embedding by averaging word vectors.
    """
    words = text.split()
    vectors = []
    for word in words:
        try:
            vectors.append(model[word])
        except KeyError:
            continue
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)


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
    
    df_confusion_counts = pd.DataFrame(
        confusion,
        index=[f"Cluster {i}" for i in range(n_clusters)],
        columns=emotion_labels
    )
    
    return df_confusion, df_confusion_counts


def evaluate_clustering(X, true_labels, predicted_labels, emotion_labels):
    """
    Evaluate clustering performance with multiple metrics.
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    sil = silhouette_score(X, predicted_labels)
    
    df_confusion_pct, df_confusion_counts = create_cluster_label_confusion(
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
            'purity': dominant_purity,
            'size': df_confusion_counts.iloc[cluster_idx].sum()
        })
    
    overall_purity = np.mean([p['purity'] for p in cluster_purities])
    
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


def run_method_e(df, sample_size=21000, n_clusters=7, random_seed=42):
    """
    Run Method E: K-means on Word2Vec sentence embeddings.
    """
    start_time = time.time()
    
    print("="*60)
    print("METHOD E: K-MEANS + WORD2VEC EMBEDDINGS")
    print("="*60)
    print(f"Sample size: {sample_size}")
    print(f"Number of clusters: {n_clusters}")
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
    
    # Step 2: Load Word2Vec and generate embeddings
    print("\n" + "-"*40)
    print("STEP 2: GENERATING WORD2VEC EMBEDDINGS")
    print("-"*40)
    model = load_word2vec_model()
    
    print("Generating sentence embeddings...")
    embeddings = []
    for text in df_sample['clean_text_transf']:
        emb = get_sentence_embedding(text, model)
        embeddings.append(emb)
    
    X = np.array(embeddings)
    print(f"Embeddings shape: {X.shape}")
    
    # Step 3: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 4: Get true labels
    true_labels = df_sample['label'].tolist()
    emotion_labels = sorted(df_sample['label'].unique())
    print(f"Emotion labels: {emotion_labels}")
    
    # Step 5: Run K-means
    print("\n" + "-"*40)
    print("STEP 3: K-MEANS CLUSTERING")
    print("-"*40)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    predicted_labels = kmeans.fit_predict(X_scaled)
    
    print("Cluster sizes:")
    for i in range(n_clusters):
        print(f"  Cluster {i}: {np.sum(predicted_labels == i)} samples")
    
    # Step 6: Evaluate
    print("\n" + "-"*40)
    print("STEP 4: EVALUATION")
    print("-"*40)
    results = evaluate_clustering(X_scaled, true_labels, predicted_labels, emotion_labels)
    
    print(f"\nEvaluation Metrics:")
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    
    # Step 7: Confusion matrix
    print("\n" + "-"*40)
    print("STEP 5: CLUSTER COMPOSITION (Confusion Matrix - Percentages)")
    print("-"*40)
    print("Rows = Clusters, Columns = True Emotion Labels")
    print()
    print(results['confusion_pct'].round(2).to_string())
    
    # Step 8: Dominant emotion per cluster
    print("\n" + "-"*40)
    print("STEP 6: DOMINANT EMOTION PER CLUSTER")
    print("-"*40)
    for p in results['cluster_purities']:
        print(f"  Cluster {p['cluster']} (size={p['size']}): {p['dominant_emotion']} ({p['purity']:.1f}%)")
    
    # Step 9: Per-emotion recall
    print("\n" + "-"*40)
    print("STEP 7: PER-EMOTION RECALL")
    print("-"*40)
    for emotion, recall in results['per_emotion_recall'].items():
        print(f"  {emotion}: {recall:.2%}")
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("METHOD E SUMMARY")
    print("="*60)
    print(f"  ARI: {results['ari']:.4f}")
    print(f"  NMI: {results['nmi']:.4f}")
    print(f"  Silhouette: {results['silhouette']:.4f}")
    print(f"  Overall Purity: {results['overall_purity']:.2f}%")
    print(f"  Time: {elapsed/60:.1f} minutes")
    
    return results


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(dataset)} instances")
    
    results = run_method_e(
        df=dataset,
        sample_size=21000,
        n_clusters=7,
        random_seed=42
    )
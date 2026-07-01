# K-Means Algorithm on the 7-label emotions Dataset

### Comparative Analysis of Sampling Strategies and Scalable Clustering Algorithms for Emotion Discovery on Imbalanced Text Data

### **Author**: Kanita Tafro, *University of Sarajevo*

*Seminar paper for the course Machine Learning: Unsupervised techniques*

# Instructions

1. **Install dependencies**:
```
pip install -r requirements.txt
```

2. **Data acquisition and preparation**: Follow instructions given in *`data/ReadMe.md`*

3. **Run experiments**
```
python -m experiments.method_A
```
```
python -m experiments.method_B
```
```
python -m experiments.method_C
```
```
python -m experiments.method_D
```
```
python -m experiments.method_E
```
```
python -m experiments.method_F
```


# Experiments (6 methods)

A-D = TF-IDF + K-Means (4 different sampling strategies)
* **Method A**: single random sampling (21000 random samples)
* **Method B**: repeated sampling (10 iterations of 21000 random samples)
* **Method C**: stratified sampling (21000 randomsamples with natural distribution across 7 true labels)
* **Method D**: balanced sampling (7 x 3000 samples)

E-F = comparison methods
* **Method E**: Word2Vec + KMeans
* **Method F**: TF-IDF + UMAP + DBSCAN

# Results

## Experimental Setup

- **Hardware**: 16 GB RAM, 8-core processor
- **Preprocessing**: Lowercasing, punctuation/number removal, stopword removal, stemming (NLTK)
- **TF-IDF Parameters**: `max_features=5000`, `min_df=3`, `max_df=0.85`
- **K-means**: Fixed at `k=7` unless otherwise noted
- **Implementation**: Python with scikit-learn

## K-Means Clustering Experiments (Methods A–D)

### Method A: Single Random Sample (Baseline)

**Sample Distribution**: Joy 30.2%, Sadness 25.6%, Disgust 4.2%, Surprise 4.8%

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | **-0.0056** | Worse than random |
| NMI | 0.0134 | Near-zero mutual info |
| Silhouette | 0.0084 | No cluster separation |
| Overall Purity | 31.72% | Low purity |
| Optimal K | 14 | Double true k=7 |

**Per-Emotion Recall**:

| Emotion | Recall | Note |
|---------|--------|------|
| Disgust | 85.01% | Distinct lexical markers |
| Love | 50.94% | Spread across clusters |
| Surprise | 50.19% | Poorly captured |
| Joy | 49.92% | Distributed |
| Fear | 46.74% | Poor |
| Anger | 44.66% | Poor |
| Sadness | 44.59% | Poor |

All clusters dominated by joy/sadness. Largest cluster (49.2% data) contains 30.5% joy + 23.3% sadness. Disgust is uniquely separable.

---

### Method B: Repeated Sampling (Stability Analysis)

Two runs: (1) optimal k search, (2) fixed k=7 stability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean ARI | 0.0046 ± 0.0047 | Near-random |
| Mean NMI | 0.0199 ± 0.0169 | Very low |
| Mean Silhouette | 0.0091 ± 0.0156 | No separation |
| Centroid Similarity | 0.0144 ± 0.0041 | Very low stability |
| Pairwise ARI | 0.0002 ± 0.0050 | Assignments random |
| Cluster Size CV | 1.9762 | Extreme variation |
| Optimal K | 12 | Not 7 |

One cluster captures 90-99% of data in every iteration. Dominant cluster changes randomly. Negative Silhouette scores indicate worse than random assignment.

---

### Method C: Stratified Sampling

**Sample**: Perfectly matches full dataset distribution (disgust 4.2%, surprise 4.8%)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | 0.0040 | Near-random |
| NMI | 0.0331 | Very low |
| Silhouette | 0.0086 | No separation |
| Overall Purity | 34.29% | Low |
| Optimal K | 14 | Double true k=7 |

 Two clusters capture 72% of data (50.1% + 21.9%). Joy and sadness dominate every cluster. Disgust recall: 88.75%, Love recall: 40.00% (lowest).

---

### Method D: Balanced Sampling (3,000 per label)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | 0.0231 | Near-random |
| NMI | 0.0515 | Very low |
| Silhouette | 0.0081 | No separation |
| Centroid Stability | 0.2432 ± 0.0234 | Moderate instability |
| Train-Test Gap (ARI) | 0.0178 | Generalizes but poorly |

Minimal improvement from 500→3,000 samples per label (+0.0074 ARI). Data quantity is NOT the bottleneck.

---

## Summary of K-Means Experiments

| Metric | Method A | Method B | Method C | Method D |
|--------|----------|----------|----------|----------|
| ARI | -0.0056 | 0.0046 ± 0.0047 | 0.0040 | 0.0231 |
| NMI | 0.0134 | 0.0199 ± 0.0169 | 0.0331 | 0.0515 |
| Silhouette | 0.0084 | 0.0091 ± 0.0156 | 0.0086 | 0.0081 |
| Optimal K | 14 | 12 | 14 | 7 (fixed) |
| Disgust Recall | 85.01% | N/A | 88.75% | N/A |

### Consistent Findings

1. **All methods failed**: ARI near zero (-0.0056 to 0.0231)
2. **Optimal K ≠ 7**: Methods found k=12–14 (lexical patterns, not emotions)
3. **Cluster collapse**: 50–99% data in 1–2 clusters
4. **Disgust is separable**: 85–89% recall (unique lexical markers)
5. **Other emotions overlap**: Recall below 52% for all others
6. **Stability is poor**: Centroid positions, assignments, sizes all vary wildly

### Why K-Means + TF-IDF Fails

1. **TF-IDF cannot capture semantics** ("I feel down" vs "I'm feeling blue" → no overlap)
2. **K-means' spherical assumption** fails for overlapping, non-spherical emotion distributions
3. **Curse of dimensionality**: With 5000 features, Euclidean distance is meaningless

### Implication for full dataset

Flat learning curve proves **data quantity is NOT the bottleneck**. More data won't help—the problem is representation (TF-IDF) and algorithm (K-means).

---

## Alternative Approaches (Methods E–F)

### Method E: Word2Vec + K-Means

Each text → 300-dim dense vector (average of Word2Vec word embeddings)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | 0.0199 | Near-random |
| NMI | 0.0322 | Very low |
| Silhouette | 0.0047 | No separation |
| Overall Purity | 31.90% | Low |
| Largest Cluster | 28.3% | More balanced than TF-IDF |
| Disgust Recall | 55.03% | Dropped from 85-89% |

More balanced clusters (largest: 28.3% vs 49-50% for TF-IDF), but:
- ARI remains near zero
- Disgust separability lost (85% → 55%) — semantic averaging dilutes emotion-specific markers
- Love shows slight improvement (21.2% in one cluster)

Semantic averaging alone is insufficient.

---

### Method F: DBSCAN + TF-IDF + UMAP

TF-IDF (5000-dim) → UMAP → 50-dim → DBSCAN

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | **-0.0047** | Worse than random |
| NMI | **0.1334** | Highest among all methods |
| Silhouette | **-0.0622** | Negative — poor separation |
| Overall Purity | **87.12%** | High but misleading |
| # Clusters | **226** | Extreme fragmentation |
| Noise Points | 4,141 (19.7%) | Significant outliers |

**Why 226 clusters?**
- DBSCAN finds dense regions wherever they exist
- High intra-class variation → multiple clusters per emotion
- Overlapping emotions form separate clusters
- UMAP's local structure preservation amplifies small differences

**Fragmentation by Emotion**:

| Emotion | Pure Clusters (≥90%) | Recall |
|---------|---------------------|--------|
| Joy | 27 | 42.95% |
| Sadness | 32 | 38.31% |
| Fear | 12 | 41.77% |
| Anger | 10 | 38.74% |
| Love | 8 | 44.98% |
| Surprise | 3 | 44.50% |
| Disgust | 0 | 52.97% |

Many 100% pure clusters (27 joy, 32 sadness), but extreme fragmentation means each pure cluster captures only a fraction of each emotion. Disgust recall remains highest (52.97%).

---

## Summary of All Methods

| Method | Algorithm | Features | ARI | NMI | Silhouette | Purity |
|--------|-----------|----------|-----|-----|------------|--------|
| A | K-means | TF-IDF | -0.0056 | 0.0134 | 0.0084 | 31.72% |
| B | K-means | TF-IDF | 0.0046 | 0.0199 | 0.0091 | N/A |
| C | K-means | TF-IDF | 0.0040 | 0.0331 | 0.0086 | 34.29% |
| D | K-means | TF-IDF | 0.0231 | 0.0515 | 0.0081 | N/A |
| E | K-means | Word2Vec | 0.0199 | 0.0322 | 0.0047 | 31.90% |
| F | DBSCAN | TF-IDF+UMAP | -0.0047 | **0.1334** | -0.0622 | **87.12%** |

### Key Takeaways

1. **TF-IDF + K-means unsuitable** — all K-means methods produced near-zero ARI
2. **Class imbalance not the primary issue** — random, stratified, and balanced all gave similar results
3. **Word2Vec improves balance, not quality** — largest cluster: 28.3% vs 49–50%, but ARI unchanged; disgust separability lost (85% → 55%)
4. **DBSCAN captures structure but fragments** — highest NMI (0.1334) and purity (87.12%) but extreme fragmentation (226 clusters)
5. **Fundamental challenge remains** — regardless of representation or algorithm, ground truth emotion labels do not align with natural structures in the data. Emotions are expressed in highly diverse ways that do not form clean, separable clusters.
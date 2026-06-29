# K-Means Algorithm on the 7-label emotions dataset


# Instructions

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Data acquisition and preparation: Follow instructions given in *data/ReadMe.md*

3. Experiment
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

### Method A: Single Random Sample (Baseline)

A single 21,000-instance random sample was drawn preserving the natural class distribution.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | -0.0056 | Worse than random chance |
| NMI | 0.0134 | Near-zero mutual information |
| Silhouette | 0.0084 | Almost no cluster separation |
| Overall Purity | 31.72% | Low cluster purity |
| Optimal K | 14 | Double true k=7 |

**Per-Emotion Recall:**

| Emotion | Recall | Interpretation |
|---------|--------|----------------|
| Disgust | 85.01% | Distinct linguistic patterns |
| Love | 50.94% | Spread across clusters |
| Surprise | 50.19% | Poorly captured |
| Joy | 49.92% | Distributed across clusters |
| Fear | 46.74% | Poorly captured |
| Anger | 44.66% | Poorly captured |
| Sadness | 44.59% | Poorly captured |

**Key Finding:** All clusters were dominated by either joy or sadness. Disgust achieved 85% recall, suggesting it has unique lexical markers, while all other emotions had recall below 51%.

---

### Method B: Repeated Sampling (Stability Analysis)

Ten independent 21,000-instance samples were clustered to assess stability.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean ARI | -0.0004 ± 0.0038 | Near-random agreement |
| Mean Silhouette | 0.0115 ± 0.0557 | Almost no cluster separation |
| Centroid Similarity | 0.0119 ± 0.0025 | Very low stability |
| Cluster Size CV | 2.3519 | Extreme size variation |
| Optimal K | 12 | Not 7 |

**Key Finding:** In every iteration, one cluster captured 90-99% of the data. The dominant cluster changed randomly across iterations, and cluster assignments were effectively random (pairwise ARI ~0).

---

### Method C: Stratified Sampling (Natural Distribution)

Stratified sampling preserved the natural class distribution from the full dataset.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | 0.0040 | Near-random agreement |
| NMI | 0.0331 | Very low mutual information |
| Silhouette | 0.0086 | Almost no cluster separation |
| Overall Purity | 34.29% | Low cluster purity |
| Optimal K | 14 | Double true k=7 |

**Per-Emotion Recall:**

| Emotion | Recall |
|---------|--------|
| Disgust | 88.75% |
| Surprise | 51.94% |
| Joy | 51.59% |
| Fear | 49.52% |
| Anger | 48.75% |
| Sadness | 47.23% |
| Love | 40.00% |

**Key Finding:** Joy and sadness dominated every cluster. Two clusters captured 72% of the data. Disgust remained highly separable, while love had the lowest recall (40%).

---

### Method D: Balanced Sampling (Scalability Test)

Balanced sampling removed the class imbalance confounder with 3,000 instances per emotion label.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ARI | 0.0231 | Near-random agreement |
| NMI | 0.0515 | Very low mutual information |
| Silhouette | 0.0081 | Almost no cluster separation |
| Centroid Stability | 0.2432 ± 0.0234 | Moderate instability |

**Learning Curve (ARI vs. Samples per Label):**

| Samples/Label | ARI |
|---------------|-----|
| 500 | 0.1016 |
| 1,000 | 0.1085 |
| 1,500 | 0.1003 |
| 2,000 | 0.0976 |
| 2,500 | 0.1054 |
| 3,000 | 0.1090 |

**Key Finding:** Minimal improvement from 500 to 3,000 samples per label (+0.0074 ARI). Data quantity is not the bottleneck.

**Out-of-Sample Validation:**

| Set | ARI | NMI | Silhouette |
|-----|-----|-----|------------|
| Training | 0.0231 | 0.0515 | 0.0081 |
| Test | 0.0053 | 0.0313 | 0.0092 |
| **Gap** | **0.0178** | **0.0202** | **-0.0011** |

**Scalability Verdict:** ✅ VIABLE
- Train-test gap < 0.05 indicates strong generalization
- Processing the full 473,247 dataset would yield minimal improvement
- **But performance is near-random (ARI ~0.02)**

---

## Summary: All K-Means Methods

| Metric | Method A | Method B | Method C | Method D |
|--------|----------|----------|----------|----------|
| ARI | -0.0056 | -0.0004 ± 0.0038 | 0.0040 | 0.0231 |
| NMI | 0.0134 | - | 0.0331 | 0.0515 |
| Silhouette | 0.0084 | 0.0115 ± 0.0557 | 0.0086 | 0.0081 |
| Optimal K | 14 | 12 | 14 | 7 (fixed) |
| Disgust Recall | 85.01% | - | 88.75% | - |
| Scaling Viable | N/A | N/A | N/A | Yes (poor) |

### Key Findings Across All Methods

1. **All methods failed** — ARI remained near zero across all four methods (-0.0056 to 0.0231)
2. **Optimal K is not 7** — Methods A and C found k=14, Method B found k=12
3. **Cluster collapse** — 50-99% of data collapsed into one or two clusters in all methods
4. **Disgust is separable** — Consistently achieved 85-89% recall (unique lexical markers)
5. **Other emotions overlap** — Recall below 52% for love, anger, fear, sadness, surprise
6. **Stability is poor** — Cluster assignments are effectively random across iterations

### Why K-Means + TF-IDF Fails

1. **TF-IDF cannot capture semantics** — "I feel down" and "I'm feeling blue" share no TF-IDF overlap but convey the same emotion
2. **K-means' spherical cluster assumption** — Emotions form highly overlapping, non-spherical distributions
3. **Curse of dimensionality** — With 5,000 features, Euclidean distance becomes meaningless

### Conclusion from K-Means Experiments

> **K-means on TF-IDF features is unsuitable for emotion clustering.** The problem is not data quantity (flat learning curve), not class imbalance (consistent across A-D), and not sampling variation (stable across iterations). **The fundamental issue is the representation (TF-IDF) and algorithm (K-means).**
# 🚀 SMILES-2026 Hallucination Detection: Final Solution

> **TL;DR:** 
> Building a hallucination detector on a highly constrained dataset (< 700 samples) with high-dimensional hidden states (896 dimensions per layer) inherently risks severe overfitting. To tackle this, my solution replaces the baseline MLP with a **Gradient Boosting framework (CatBoost)** operating on a carefully engineered, low-dimensional feature space. 
>
> The feature space is built upon two equally important pillars:
> 1. **Representation Projections:** Extracting the signal directly from the embeddings using class-separating directional vectors (computed strictly without data leakage).
> 2. **Geometric & Statistical Features:** Capturing the model's internal uncertainty and representation dynamics using advanced metrics like token entropy, Kirchhoff index, effective rank, and cosine drift.
>
> To ensure robustness, the final feature subset was not picked arbitrarily but was derived from a **rigorous optimization pipeline** designed to balance raw predictive accuracy with cross-fold stability. Combined with a **Repeated Stratified K-Fold** validation strategy and a **Dual-Model approach** (optimizing separate pipelines for Accuracy and ROC-AUC), this solution delivers highly stable and accurate predictions on unseen data.

## 📑 Table of Contents
-[🚀 SMILES-2026 Hallucination Detection: Final Solution](#-smiles-2026-hallucination-detection-final-solution)
- [🚀 SMILES-2026 Hallucination Detection: Final Solution](#-smiles-2026-hallucination-detection-final-solution)
  - [📑 Table of Contents](#-table-of-contents)
  - [1. Reproducibility](#1-reproducibility)
    - [Environment Setup](#environment-setup)
    - [Running the Solution](#running-the-solution)
  - [2. The Core Challenge \& Architecture Rationale](#2-the-core-challenge--architecture-rationale)
    - [2.1. Why the Baseline MLP Fails](#21-why-the-baseline-mlp-fails)
    - [2.2. The Shift to Gradient Boosting (CatBoost)](#22-the-shift-to-gradient-boosting-catboost)
    - [2.3. Strict Validation Strategy](#23-strict-validation-strategy)
    - [2.4. Feature Engineering: Projections \& Geometry](#24-feature-engineering-projections--geometry)
    - [2.5. The Optimization Pipeline](#25-the-optimization-pipeline)
    - [2.6. The Dual-Model Classifier](#26-the-dual-model-classifier)
  - [3. Results \& Metrics](#3-results--metrics)
    - [🎯 The Importance of Standard Deviation](#-the-importance-of-standard-deviation)
    - [Key Contributors to Metric Improvement](#key-contributors-to-metric-improvement)
  - [4. Experiments \& Failed Attempts](#4-experiments--failed-attempts)
    - [4.1. Tuning the Baseline MLP](#41-tuning-the-baseline-mlp)
    - [4.2. Unsupervised Dimensionality Reduction (PCA)](#42-unsupervised-dimensionality-reduction-pca)
    - [4.3. Mean-Pooling Across All Tokens](#43-mean-pooling-across-all-tokens)
  - [5. References](#5-references)

---

## 1. Reproducibility

The generated predictions.csv file is also publicly available here: [link](https://drive.google.com/file/d/1smX0nF0gHB5xto11libvDzldDvGG0pyB/view?usp=sharing)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

# Set up a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies including CatBoost
pip install -r requirements.txt
```

### Running the Solution

```bash
python solution.py
```

---

## 2. The Core Challenge & Architecture Rationale

The main challenge in this task is the small dataset (fewer than 700 samples) combined with huge feature dimensionality (896 dimensions per layer). This creates a classic "Small $N$, Large $D$" problem.

### 2.1. Why the Baseline MLP Fails
A basic rule in machine learning is that to avoid overfitting, the number of model parameters should be much smaller than the number of training samples [1]. The baseline approach uses a Multi-Layer Perceptron (MLP) trained directly on the embeddings. However, this approach has two big issues:
- **Too Many Parameters:** To learn useful patterns, an MLP needs hidden layers. With 896-dimensional inputs, the model quickly gets hundreds of thousands of parameters. This is simply too much for ~600 rows of data.
- **Instability:** Training such a large network on a tiny dataset causes it to just memorize the training data. Trying to fix this with standard deep learning techniques (like heavy Dropout or weight decay) only made the training unstable and led to bad results on unseen data.

### 2.2. The Shift to Gradient Boosting (CatBoost)
To solve this problem, I decided to replace the neural network with **CatBoost**. I chose it for three main reasons:
1. **Better on Small Data:** Gradient boosting usually performs much better than deep learning on tabular data when the dataset is highly restricted and the features are heterogeneous [2].
2. **Protection from Overfitting:** CatBoost has strong built-in parameters to prevent overfitting, such as L2 leaf regularization and column subsampling.
3. **Fast Experiments:** CatBoost trains much faster than an MLP on a CPU. This allowed me to quickly test different feature ideas and run large hyperparameter searches.

### 2.3. Strict Validation Strategy
On a small dataset, a standard train/test split (or a simple 5-fold CV) depends too much on the random seed. A "lucky" split can give you a false sense of high accuracy. 
To make sure my model actually generalizes to unseen data, I used a **Repeated Stratified K-Fold** strategy (6 splits × 3 repeats = 18 independent folds). Every feature and hyperparameter was evaluated across all 18 folds to guarantee stability.

> *To use CatBoost, I had to transform the raw 3D embedding tensors into flat tabular data. I did this by compressing the LLM representations into a small set of informative features using **Representation Projections** and **Geometric/Statistical Features**.*

### 2.4. Feature Engineering: Projections & Geometry
**Files modified:** `aggregation.py` & `probe.py`

**Layer Selection Rationale:** 
Instead of extracting features from all 25 layers, I focused mostly on the middle and late layers (typically layers 12 to 24). Early layers usually process basic grammar and syntax, while deeper layers handle factual knowledge, semantics, and the final token prediction. The "hallucination signal" is much stronger in these deeper layers [3].

Below are the details of the extracted features. Click to expand:

<details>
<summary><b>1. Representation Projections (Mean Difference Projection)</b></summary>
<br>
Inspired by <i>Representation Engineering</i> [4], I found the specific "direction" in the embedding space that separates truthful answers from hallucinations.

*   **The Math:** For a specific layer, I calculate the mean vector of all hallucinated samples and subtract the mean vector of truthful samples to get the direction $D$:
    $$ D = \mu_{hallucinated} - \mu_{truthful} $$
*   **The Projection:** The raw hidden state $x$ of a new sample is projected onto this normalized direction:
    $$ Feature = x \cdot \frac{D}{||D||} $$

This compresses 896 dimensions into just **one highly informative number** per layer. 
> **Strict Validation:** To avoid data leakage, $D$ is calculated *only on the training fold* inside the `fit()` method. 

**Insight:** These projection features (especially `proj_L12` and `proj_L13`) were the absolute best predictors for the ROC-AUC model.
</details>

<details>
<summary><b>2. Uncertainty: Entropy & Kurtosis</b></summary>
<br>
When a model hallucinates, its internal activations often show signs of uncertainty [5]. I calculated several statistical metrics on the last token and across all tokens:

*   **Softmax Entropy:** Measures how "unsure" the model is.
    $$ H = - \sum p_i \log(p_i), \quad \text{where } p_i = \text{softmax}(x_i) $$
*   **Energy Entropy:** Similar to softmax, but uses the squared values of activations as probabilities.
*   **Kurtosis:** Measures the "tailedness" of the activation distribution (how many extreme outlier values exist).
    $$ K = \frac{1}{N} \sum \left( \frac{x_i - \mu}{\sigma} \right)^4 - 3 $$

**Insight:** Statistical features performed incredibly well for strict classification. For example, `all_tokens_max_kurtosis_layer17` was the #1 most important feature for the Accuracy model.
</details>

<details>
<summary><b>3. Layer Dynamics: Cosine Drift & Norms</b></summary>
<br>
I also looked at how the representation changes as it passes through the network [6].

*   **Cosine Drift:** The cosine similarity between the last-token embeddings of adjacent layers (e.g., Layer 22 vs Layer 23). Rapid shifts (low similarity) often indicate that the model is abruptly changing context or "guessing".
    $$ Drift_{i, i+1} = \frac{L_i \cdot L_{i+1}}{||L_i|| \times ||L_{i+1}||} $$
*   **Layer Norms:** The standard L2 norm ($||x||_2$) of the embedding.

**Insight:** Features like `drift_18_19` and `norm_L12` consistently stayed in the top 5 most important features for the Accuracy model.
</details>

<details>
<summary><b>4. Topology: Kirchhoff Index & Effective Rank</b></summary>
<br>
To capture the shape and "focus" of the model's attention, I used Singular Value Decomposition (SVD) and graph metrics:

*   **Effective Rank:** Uses singular values to measure how many independent dimensions the activations actually use.
*   **Kirchhoff Index:** Treats the token similarity matrix as a graph and calculates its total resistance. It measures how "diffuse" or spread out the information is among the tokens.
</details>

<br>

> **Note on Feature Importance:** 
> While my logs show clear top features, gradient boosting models often share or "dilute" feature importance among highly correlated variables. Because features from adjacent layers (e.g., Layer 16 and 17) are highly correlated, the exact ranking order is slightly blurred. However, the overall trend between groups of features (Projections vs. Geometry) remains very clear.

### 2.5. The Optimization Pipeline
Having generated over 100 potential features, using all of them at once would cause CatBoost to overfit. To select the optimal subset, I built a two-stage optimization pipeline (run entirely offline before final submission). (As an implementation detail: to drastically speed up hypothesis testing and feature generation, I pre-cached all LLM activations in fp16, reducing the feature extraction bottleneck to zero)

1. **Hyperparameter Search (Optuna):** I used the TPE Sampler to find the best CatBoost hyperparameters and optimal feature groups. To prioritize stability over raw metrics, the objective function explicitly penalized cross-fold variance: `score = val_mean - 0.5 * val_std`.
2. **Recursive Feature Elimination (SHAP):** After finding the best grid, I utilized CatBoost's `RecursiveByShapValues`. A feature was kept only if it demonstrated predictive power across multiple independent cross-validation splits (cross-seed stability).

### 2.6. The Dual-Model Classifier
During the optimization phase, a very clear pattern emerged from the logs:
* Maximizing **Accuracy** (which relies on a strict decision boundary) requires statistical and geometric features (`kurtosis`, `drift`, `norms`).
* Maximizing **ROC-AUC** (which evaluates the global ranking of probabilities) relies heavily on the Mean Difference `projections`.

Because of this fundamental difference, a single set of features and hyperparameters cannot be optimal for both metrics. Therefore, my `HallucinationProbe` trains two separate CatBoost models under the hood:
* **The Accuracy Model:** Uses a specific subset of 33 features (`_ACC_SELECTED`) and its own parameters. It is called when `predict()` is executed.
* **The ROC-AUC Model:** Uses a different subset of 41 features (`_AUC_SELECTED`) and its own parameters. It is called when `predict_proba()` is executed.

---

## 3. Results & Metrics

According to the competition rules, Accuracy is the primary ranking metric. My Accuracy model achieves a highly stable ~73.34%, successfully outperforming the 70.10% majority-class baseline.
However, because the dataset is heavily imbalanced (483 hallucinated vs. 206 truthful), Accuracy can be deceptive. To prove that the model genuinely separates the classes and doesn't just exploit the threshold, I also heavily optimized for ROC-AUC (which evaluates the global ranking of probabilities)..

To prove that the model does not rely on a "lucky" data split, I provide three levels of evaluation:
1. **Majority-Class Baseline:** The naive floor performance.
2. **Standard Evaluation (`solution.py`):** The default 18-fold run (6 splits × 3 repeats).
3. **Massive Stress-Test (`run_cached.py`):** Out-of-fold metrics computed across **42 independent K-Fold cross-validations** with different random seeds.

| Evaluation Target | Accuracy | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **1. Majority-Class Baseline** | 70.10% | 82.42% | *N/A* |
| **2. Solution.py (18 folds)** | 73.34% | 81.59% | 72.26% |
| **3. Stress-Test (42 seeds)** | **72.95%** | **81.41%** | **71.81%** |

### 🎯 The Importance of Standard Deviation
The most critical result of the 42-seed stress-test is not just the mean score, but the **Standard Deviation (std)**. 

*   **Accuracy std:** $\pm 0.0060$ (0.6%)
*   **F1 std:** $\pm 0.0044$ (0.4%)
*   **ROC-AUC std:** $\pm 0.0090$ (0.9%)

These variance metrics are **exceptionally low**. They mathematically prove that the feature selection pipeline and the CatBoost parameters are incredibly stable. Regardless of how the 689 samples are shuffled and split, the model reliably extracts the hallucination signal without overfitting to specific data folds.

### Key Contributors to Metric Improvement
1. **Separating the Models:** Using one CatBoost for `predict` (Accuracy) and a completely different CatBoost for `predict_proba` (AUC) provided the biggest overall boost.
2. **Representation Projections:** The mean difference projection scalar was the ultimate key to maximizing the ROC-AUC metric.
3. **Variance Penalty:** Using `score = mean - 0.5 * std` during the Optuna tuning slightly lowered the absolute maximum peak score, but it is exactly what drove the standard deviation down to less than 1%.

---

## 4. Experiments & Failed Attempts

A significant portion of the development time was spent trying to extract the maximum possible signal directly from the raw embedding tensors before pivoting to the CatBoost-based feature engineering approach. Below are the key failed attempts and the insights gained from them.

### 4.1. Tuning the Baseline MLP
**Hypothesis:** The MLP baseline was simply under-tuned. By concatenating the raw embeddings from multiple layers and adding proper regularization (heavy Dropout, Weight Decay, Learning Rate Schedulers), the neural network could learn to detect hallucinations.
**Result (Failed):** The metrics were highly unstable. On some validation splits, the F1 and AUC would spike, giving a false sense of success, but on the next random seed, the model would perform worse than random guessing. Even with heavy regularization, the MLP either immediately memorized the 600 training samples or failed to converge entirely. Adding the engineered geometric features into the MLP only worsened the results, as the noisy high-dimensional space disrupted the gradient descent. 

### 4.2. Unsupervised Dimensionality Reduction (PCA)
**Hypothesis:** If 896 dimensions are too many, using Principal Component Analysis (PCA) to compress the embeddings into a dense 32- or 64-dimensional vector before feeding them to the classifier would solve the overfitting.
**Result (Failed):** PCA catastrophically destroyed the hallucination signal [7]. 
**Why it failed:** PCA works by preserving the axes with the highest variance. In an LLM's hidden state, the highest variance corresponds to general context, grammar, and syntax. The difference between a "truthful" representation and a "hallucinated" one is extremely subtle (low variance). PCA essentially compressed away the hallucination signal, treating it as background noise. This realization led directly to the adoption of *supervised Representation Projections*, which specifically seek the axis of class separation rather than maximum variance.

### 4.3. Mean-Pooling Across All Tokens
**Hypothesis:** Averaging the embeddings of all tokens in the response would provide a "global" understanding of the hallucination, rather than just looking at the last token.
**Result (Failed):** It completely washed out the signal. The LLM "decides" on its factual trajectory at specific critical tokens (often towards the end of the generation). Mean-pooling diluted these critical high-uncertainty moments with dozens of low-uncertainty grammar tokens. I mitigated this by computing statistical metrics (like `max_kurtosis`) across all tokens, rather than averaging the embeddings themselves.

---

## 5. References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. (Reference for the bias-variance tradeoff and the "Small N, Large D" parameter constraint).
2. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* Advances in Neural Information Processing Systems (NeurIPS).
3. Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying*. Findings of the Association for Computational Linguistics (ACL). (Reference for factuality signals being heavily concentrated in middle/late layers).
4. Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv preprint arXiv:2310.01405. (Theoretical foundation for extracting class-separating directions from LLM hidden states).
5. Kuhn, L., et al. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*. International Conference on Learning Representations (ICLR).
6. Zhang, Z., et al. (2025). *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs*. ACL. (Validates that analyzing the dynamic evolution and cosine drift of hidden states across middle and late layers is highly effective for detecting hallucinations).
7. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. (Theoretical proof on why unsupervised variance maximization like PCA often discards class-separating information compared to supervised projections).
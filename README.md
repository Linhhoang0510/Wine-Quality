# Wine Quality Prediction

A machine learning project that predicts wine quality from physicochemical measurements, comparing a Classification & Regression Tree (CART) against a feed-forward Neural Network. Built in **R**, using `rpart` for the decision tree and `torch` for the neural network.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Dataset](#3-dataset)
4. [Tech Stack](#4-tech-stack)
5. [Project Workflow](#5-project-workflow)
6. [Data Cleaning & Preparation](#6-data-cleaning--preparation)
7. [Exploratory Data Analysis](#7-exploratory-data-analysis)
8. [Modelling](#8-modelling)
9. [Evaluation](#9-evaluation)
10. [Results & Discussion](#10-results--discussion)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [How to Run](#12-how-to-run)
13. [Repository Structure](#13-repository-structure)

---

## 1. Project Overview

This project frames wine quality assessment as a **supervised multi-class classification** problem. Raw quality scores (0–10) are grouped into three ordinal classes — **Low, Medium, and High** — and two models are trained and compared to predict these classes from 11 physicochemical input features.

The goal is to deliver an **objective, data-driven alternative** to subjective sensory evaluation, and to benchmark a classic interpretable model (Decision Tree) against a more flexible deep learning model (Neural Network).

---

## 2. Motivation

A wide range of factors affect wine quality — grape variety, brand, climate, and more. Traditionally, quality is evaluated in two parts:

- **Physicochemical analysis** — objective laboratory measurements such as alcohol level, pH, residual sugar, and acidity.
- **Sensory testing** — subjective tasting carried out by wine specialists.

The sensory approach has real limitations: expert taste is subjective, evaluations vary from person to person, and the process is often complex, costly, and time-consuming. Machine learning offers a way to model the relationship between measurable physicochemical properties and perceived quality, providing a **consistent, scalable, and objective** complement to traditional assessment.

---

## 3. Dataset

The data comes from the **Wine Quality** dataset in the UCI Machine Learning Repository — variants of the Portuguese *"Vinho Verde"* wine. It consists of two files:

| Dataset | Samples |
|---|---|
| Red wine | 1,599 |
| White wine | 4,898 |
| **Combined** | **6,497** |

**Features (12 variables per dataset):**

- **Inputs (11):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Target (1):** `quality` — an integer score from 0 to 10 based on sensory tests, where higher is better

A categorical feature, `type` (red/white), is added when the two datasets are combined.

---

## 4. Tech Stack

**Language:** R

| Purpose | Libraries |
|---|---|
| Data wrangling | `dplyr` |
| Visualisation / EDA | `ggplot2`, `GGally`, `corrplot`, `scales` |
| Decision tree + visualisation | `rpart`, `partykit` |
| Model evaluation | `caret` |
| Deep learning | `torch` |

---

## 5. Project Workflow

The project follows a complete end-to-end ML pipeline:

```
Data Loading → Cleaning & Feature Engineering → EDA → Target Binning →
Train/Test Split → Model Training (CART + NN) → Evaluation → Comparison
```

**Model Workflows:**

![Wine Quality drawio (1)](https://github.com/user-attachments/assets/4650a2d4-bd97-4641-9788-162e47d6a275)

---

## 6. Data Cleaning & Preparation

- **Merging datasets** — A `type` label (`red`/`white`) is added to each dataset, then the two are combined into a single dataset of 6,497 samples using `rbind`.
- **Quality control** — Data types and dimensions are inspected with `str()`; a missing-value check (`sum(is.na())`) confirms the data is complete, so no imputation is required.
- **Target construction** — The continuous `quality` score is binned into three ordinal classes using `cut()`:

  | Class | Quality Range |
  |---|---|
  | Low | 3–4 |
  | Medium | 5–6 |
  | High | 7–9 |

- **Feature selection** — The raw `quality` and `type` columns are dropped after binning, leaving 11 physicochemical inputs.
- **Train/Test split** — An 80/20 split is applied with a fixed random seed (`set.seed(0)`) for reproducibility.

---

## 7. Exploratory Data Analysis

- **Quality distribution** — A histogram of the `quality` variable shows the data is heavily concentrated in the mid-range scores (5–6), highlighting a significant **class imbalance** toward the Medium category. In the test set, Medium accounts for ~77% of wines, while Low is only ~4.5%.
- **Feature correlations** — A mixed correlation plot (`corrplot.mixed`) visualises relationships between the physicochemical features, helping identify multicollinearity and features most associated with quality.

---

## 8. Modelling

Two complementary approaches were implemented and compared.

### 8.1 Classification & Regression Tree (CART)

A decision tree is a tree-shaped model that visualises and follows decision paths through the feature space, offering high interpretability.

- Built with `rpart` (`method = "class"`).
- The complexity-parameter (CP) table is inspected, and the **optimal CP is selected programmatically** by minimising cross-validation error (`xerror`).
- The tree is **pruned** at the optimal CP to reduce overfitting.
- The final tree is visualised using `partykit::as.party()`.

### 8.2 Neural Network

A feed-forward neural network explores the ability of deep learning to capture complex, non-linear patterns in the data.

- **Preprocessing** — Features are min-max rescaled (`scales::rescale`) and converted to tensors (`torch_float` inputs, `torch_long` targets).
- **Architecture** — A 4-layer multilayer perceptron:

  ```
  Input (11) → Linear(100) → ReLU
             → Linear(50)  → ReLU
             → Linear(20)  → ReLU
             → Linear(3)   → Output (Low / Medium / High)
  ```

- **Training** — Cross-entropy loss, Adam optimizer (lr = 0.03), 1,000 epochs, with loss and accuracy logged every 100 epochs. A fixed seed (`torch_manual_seed(0)`) ensures reproducibility.

---

## 9. Evaluation

Both models are evaluated on the held-out 20% test set using a **confusion matrix** (`caret::confusionMatrix`), reporting accuracy alongside per-class statistics (sensitivity, specificity, balanced accuracy) and Cohen's Kappa.

- For the Decision Tree, predictions are additionally visualised with a `ggpairs` plot coloured by predicted class.
- For the Neural Network, predicted class indices are mapped back to the Low/Medium/High labels for comparison against ground truth.

---

## 10. Results & Discussion

Both models were evaluated on the held-out 20% test set. Headline accuracy looks reasonable at first glance, but the per-class statistics reveal that both models are heavily biased toward the majority **Medium** class.

| Model | Accuracy | Kappa | Balanced Acc (Low / Medium / High) |
|---|---|---|---|
| Decision Tree (pruned) | 78.4% | 0.24 | 0.50 / 0.59 / 0.62 |
| Neural Network | 74.3% | 0.05 | 0.56 / 0.51 / 0.51 |

**Confusion Matrices**

Decision Tree:

```
          Reference
Prediction  Low  Medium  High
    Low       0       0     0
    Medium   59     951   176
    High      0      46    68
```

Neural Network:

```
          Reference
Prediction  Low  Medium  High
    Low       8      27     3
    Medium   51     945   228
    High      0      25    13
```

**Key observations:**

- **Accuracy is misleading here.** The No Information Rate — the accuracy of simply always predicting "Medium" — is **76.7%**. The Decision Tree (78.4%) barely beats this baseline, and the Neural Network (74.3%) actually performs *below* it. The Decision Tree's edge over the baseline is not statistically significant (p = 0.078).

- **The Decision Tree never predicts "Low" at all.** Its sensitivity (recall) for the Low class is **0.00** — every one of the rare Low-quality wines is misclassified. It also catches only ~28% of High-quality wines. It survives on accuracy purely by predicting "Medium" for almost everything.

- **The Neural Network spreads its predictions slightly more** (it at least identifies a few Low and High wines), giving it marginally better balanced accuracy on the Low class (0.56 vs 0.50). But its overall agreement with the truth is very weak — a Kappa of just **0.05**, close to random.

- **Both models fail the classes that matter most.** Identifying genuinely Low- or High-quality wines is the practically useful task, and both models perform poorly on exactly these minority classes. This is a direct consequence of the untreated class imbalance.

**Takeaway:** Neither model has learned a robust mapping from physicochemical features to quality; both largely default to the majority class. Raw accuracy obscures this — Kappa, per-class recall, and balanced accuracy expose it. This is the central motivation for the imbalance-handling and richer evaluation metrics proposed in the next section.

---

## 11. Limitations & Future Work

- **Class imbalance is the dominant issue.** The dataset is heavily skewed toward Medium-quality wines (~77% of the test set), and this directly drove the results above — the Decision Tree achieved **0.00 recall on the Low class**, and the Neural Network scored only below the majority-class baseline. Future iterations should apply stratified sampling, class weighting, or resampling techniques (e.g. SMOTE), and report balanced accuracy / macro-F1 as the headline metrics rather than raw accuracy.

- **Wine type.** The `type` feature is currently dropped. However, the feature data distribution differs between red and white wines on several measurements — for example, white wines tend to have higher residual sugar and total sulfur dioxide, while reds tend toward higher volatile acidity and chlorides. Retaining `type` could allow the models to capture these red/white differences and improve prediction.

- **Scaling.** Scaling parameters could be fit on the training set only and applied to the test set, rather than rescaling each set independently.

- **Model validation.** Introduce early stopping for the neural network to guard against overfitting across 1,000 epochs.

- **Additional models.** Benchmark against ensemble methods such as Random Forest or Gradient Boosting, which often handle tabular, imbalanced data more robustly.

---

## 12. How to Run

```r
# Install dependencies
install.packages(c("dplyr", "corrplot", "GGally", "ggplot2",
                   "rpart", "caret", "partykit", "scales"))

# torch requires an additional setup step
install.packages("torch")
library(torch)
install_torch()

# Data is loaded directly from the GitHub raw URLs — no manual download required.
```

---

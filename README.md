<div align="center">

# 🏋️ BODY PERFORMANCE ANALYTICS
## Intelligent Classification System

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-00D27A?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-2A7BE4?style=for-the-badge)

> **A production-grade machine learning pipeline for predicting individual physical performance classes from biometric and fitness measurements.**

*Built by **BASYOUNI AI ANALYTICS** — Introduction to AI & ML Course Project*

---

</div>

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#1-project-overview--business-context) |
| 2 | [Dataset Description](#2-dataset-description--column-analysis) |
| 3 | [Data Quality & Cleaning](#3-data-quality--cleaning-decisions) |
| 4 | [Feature Engineering](#4-feature-engineering) |
| 5 | [Exploratory Data Analysis](#5-exploratory-data-analysis--key-findings) |
| 6 | [Machine Learning Pipeline](#6-machine-learning-pipeline) |
| 7 | [Model Evaluation](#7-model-evaluation--comparison) |
| 8 | [GridSearch Tuning](#8-gridsearch-hyperparameter-tuning) |
| 9 | [Regression Task](#9-regression-sub-task) |
| 10 | [K-Means Clustering](#10-unsupervised-learning--k-means-clustering) |
| 11 | [Business Recommendations](#11-business-recommendations) |
| 12 | [How to Run](#12-how-to-run-google-colab) |
| 13 | [Project Structure](#13-project-structure) |
| 14 | [Academic Integrity](#14-academic-integrity-declaration) |

---

## 1. Project Overview & Business Context

This project delivers a complete, production-grade machine learning pipeline for predicting individual physical performance classes from biometric and fitness measurements. It covers the **full data science workflow** — from raw data ingestion through model deployment readiness.

The dataset represents real-world physical fitness evaluations. Every individual has been assessed across multiple dimensions: body composition, cardiovascular health, upper-body strength, core endurance, flexibility, and explosive power. Our goal is to build a model that reliably classifies each individual into one of **four performance tiers** — and to extract meaningful, actionable insights along the way.

### 📊 Project at a Glance

| Property | Detail |
|----------|--------|
| **Dataset** | Body Performance Dataset — Kaggle (public domain) |
| **Records** | 13,393 raw → 13,392 after deduplication |
| **Features** | 12 original + 6 engineered = **18 total** |
| **Target** | `class` (A = Elite, B = Good, C = Average, D = Below Average) |
| **Problem Type** | Multi-class Classification (4 balanced classes) + Regression sub-task |
| **Best Model** | **Random Forest** — 73.9% test accuracy │ 72.0% CV (5-fold stratified) |
| **Environment** | Google Colab / Jupyter Notebook │ Python 3.x |
| **Author** | BASYOUNI AI ANALYTICS — Introduction to AI & ML Course Project |

> 💡 **Key Insight:** This dataset is exceptionally clean — zero missing values, near-perfect class balance, and no complex transformations required. This is rare in real-world ML and makes it an excellent benchmark for comparing algorithmic approaches head-to-head without confounding preprocessing issues.

---

## 2. Dataset Description & Column Analysis

Every column represents a distinct measurement domain. Understanding **what each column means** — not just what type it is — is what separates a good analyst from a great one.

| Column | Type | Business Meaning | Key Finding |
|--------|------|-----------------|-------------|
| `age` | Numeric | Participant age in years (range: 21–64) | Weak predictor — r=+0.07 |
| `gender` | Categorical | Biological sex (M=63.2%, F=36.8%) | Females lead Class A (44.3%) |
| `height_cm` | Numeric | Height in centimetres (mean 168.6 cm) | Near-zero class impact |
| `weight_kg` | Numeric | Body weight in kilograms (mean 67.0 kg) | r=+0.21 — Class D heavier |
| `body fat_%` | Numeric | Body fat percentage (mean 23.4%) | r=+0.34 — strong predictor |
| `diastolic` | Numeric | Diastolic BP — ⚠️ contains zero values (fixed) | Zero → median imputed |
| `systolic` | Numeric | Systolic BP — ⚠️ contains zero values (fixed) | Zero → median imputed |
| `gripForce` | Numeric | Hand grip strength in kg (mean 37.2 kg) | r=−0.14 moderate signal |
| `sit and bend forward_cm` | Numeric | Flexibility test in cm — **TOP PREDICTOR** | r=−0.59 — #1 feature |
| `sit-ups counts` | Numeric | Core endurance — number of repetitions | r=−0.45 — strong signal |
| `broad_jump_cm` | Numeric | Explosive jump distance in cm (regression target) | r=−0.26 moderate |
| `class` ⭐ | Ordinal | Performance class A/B/C/D — perfectly balanced | ~25% per class |

> ⚠️ **Special attention:** `sit and bend forward` is the undisputed star of this dataset — its correlation of **−0.59** with performance class makes it a dominant signal that no other feature comes close to matching. Both blood pressure columns contain clinically impossible **zero values** that required correction before any analysis could proceed.

---

## 3. Data Quality & Cleaning Decisions

Data cleaning is not a mechanical checklist — it is a series of **informed decisions** that must be justified by domain knowledge and statistical evidence.

### Issue 1 — Duplicate Rows

```python
df.duplicated().sum()  # Returns: 1
df = df.drop_duplicates(keep='first')
# Dataset: 13,393 → 13,392 rows
```

- **Detection:** `df.duplicated().sum()` returned **1 duplicate row**
- **Decision:** Removed (`keep="first"`) — identical records add no new information
- **Impact:** Dataset reduced from 13,393 to 13,392 records

---

### Issue 2 — Zero Blood Pressure Values ⚠️ Critical

```python
print(df['systolic'].min())   # 0.0  — CLINICALLY IMPOSSIBLE
print(df['diastolic'].min())  # 0.0  — CLINICALLY IMPOSSIBLE

# Fix: Replace zeros with median (not mean — distribution is right-skewed)
for col in ['systolic', 'diastolic']:
    median_val = df.loc[df[col] > 0, col].median()
    df.loc[df[col] == 0, col] = median_val
```

- **Why it matters:** A blood pressure of zero is **clinically impossible** in a living person. These are data entry errors — fields left blank and defaulted to zero.
- **Strategy:** Replace with column **median** (not mean). BP distributions are right-skewed; the median is robust to these extreme invalid values.

---

### Issue 3 — Outliers (IQR Detection + Winsorization)

```python
# IQR-based detection and capping (Winsorization)
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
```

| Column | IQR Violations | Decision |
|--------|---------------|----------|
| `sit and bend forward_cm` | **409** | ✅ Cap (Winsorize) |
| `weight_kg` | 83 | ✅ Cap (Winsorize) |
| `body fat_%` | 77 | ✅ Cap (Winsorize) |
| `broad_jump_cm` | 57 | ✅ Cap (Winsorize) |
| `diastolic` | 54 | ✅ Cap (Winsorize) |
| `age`, `sit-ups counts` | 0 | ✅ No action needed |

> 💡 **Why cap instead of delete?** Removing 409 records from a single column would discard ~3% of the dataset with no scientific basis. These participants are simply very flexible or very inflexible — that is a **real physiological signal**, not a data error. Capping retains the observation while neutralising its extreme statistical leverage.

---

## 4. Feature Engineering

Raw columns tell you **what was measured**. Engineered features tell you **what those measurements mean** when combined.

```python
# 6 domain-knowledge features created from raw columns
df['bmi']                = df['weight_kg'] / (df['height_cm'] / 100) ** 2
df['pulse_pressure']     = df['systolic'] - df['diastolic']
df['strength_endurance'] = df['gripForce'] * df['sit-ups counts']
df['athletic_score']     = df['broad jump_cm'] + df['sit-ups counts'] * 2
df['flexibility_ratio']  = df['sit and bend forward_cm'] / (df['height_cm'] / 100)
df['gender_enc']         = (df['gender'] == 'M').astype(int)
```

| Feature | Formula | Business Rationale |
|---------|---------|-------------------|
| `bmi` | weight / (height/100)² | Standard weight-for-height health index |
| `pulse_pressure` | systolic − diastolic | Cardiovascular load and arterial stiffness proxy |
| `strength_endurance` | gripForce × sit-ups | Combined force × repetition capacity composite |
| `athletic_score` | broad_jump + (sit-ups × 2) | Multi-modal athletic performance proxy |
| `flexibility_ratio` | bend_cm / (height/100) | Height-normalised flexibility — removes body size bias |
| `gender_enc` | 1 if Male, 0 if Female | Binary encoding for essential demographic signal |

> ⚠️ **Data Leakage Prevention:** `athletic_score` incorporates `broad_jump_cm`. When used in the regression task (where `broad_jump_cm` is the target), `athletic_score` is **explicitly excluded** from the predictor matrix. Failing to do this would produce a trivially perfect R² — a classic leakage error.

---

## 5. Exploratory Data Analysis — Key Findings

### 🔍 Finding 1 — Flexibility Dominates

`sit and bend forward` correlates at **r = −0.59** with performance class — the strongest signal in the dataset by a significant margin. The negative sign means better flexibility corresponds to a lower (better) class number. This is not a subtle effect — it is a **decisive physiological separator** between performance tiers.

### 🔍 Finding 2 — Body Composition Matters More Than Strength

`body fat_%` (r = +0.34) is a stronger predictor than `gripForce` (r = −0.14). Class D participants carry an average of **27.7% body fat** compared to **20.5% in Class A** — a 7.2 percentage point gap that directly explains their lower performance across all physical tests.

### 🔍 Finding 3 — The Gender Surprise

Despite comprising only **36.8%** of the dataset, female participants represent **44.3%** of Class A (elite performers). This statistically meaningful overrepresentation suggests that flexibility and endurance-weighted assessments produce more equitable elite classifications than raw-strength metrics.

### 🔍 Finding 4 — Blood Pressure Contributes Little

Both diastolic (r = +0.07) and systolic (r = +0.04) show near-zero correlations with performance class. They are not useful for classification, but Class D shows marginally elevated averages — a potential **cardiovascular risk signal** worth monitoring independently.

### 🔍 Finding 5 — K-Means Validates the Class Structure

When K-Means was applied **without class labels**, it recovered four clusters that aligned strongly with the labelled A/B/C/D classes. This confirms that the performance classifications are not arbitrary — they reflect **genuinely distinct physiological groupings** discoverable from raw data alone.

---

## 6. Machine Learning Pipeline

### ⚙️ Preprocessing (No Data Leakage)

```python
# CRITICAL: Scaler is fitted on training data ONLY
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit + transform on train
X_test_s  = scaler.transform(X_test)        # transform only — no fit
```

### 🤖 Models Trained (6 Supervised Classifiers)

| # | Model | Configuration |
|---|-------|--------------|
| 1 | **Logistic Regression** | C=0.1, max_iter=500 — linear baseline |
| 2 | **K-Nearest Neighbors** | k tuned via GridSearch (3→19), Euclidean |
| 3 | **Decision Tree** | max_depth tuned via GridSearch (3→None) |
| 4 | **SVM — RBF Kernel** | C tuned via GridSearch (0.1→10), gamma='scale' |
| 5 | **SVM — Linear Kernel** | C=0.5, direct class boundary comparison |
| 6 | **Random Forest** | 200 estimators, max_depth=12 ⭐ **Best Model** |

### 🔄 Cross-Validation Strategy

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# StratifiedKFold ensures ~25% per class in every fold

# Three split experiments
splits = {'80/20': 0.20, '70/30': 0.30, '50/50': 0.50}
```

---

## 7. Model Evaluation & Comparison

All models were evaluated on the **held-out 20% test set** — no cherry-picking, no inflated numbers.

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-------|----------|-----------|--------|----------|---------|
| ⭐ **Random Forest** | **73.9%** | 74.1% | 73.9% | 73.8% | 72.0% ±0.8% |
| Neural Network (MLP) | 72.4% | 72.6% | 72.4% | 72.3% | 70.5% ±1.1% |
| SVM — RBF (tuned) | 70.9% | 71.2% | 70.9% | 70.9% | 69.8% ±0.6% |
| Decision Tree (tuned) | 65.4% | 66.0% | 65.4% | 65.1% | 64.0% ±1.2% |
| SVM — Linear | 63.4% | 63.2% | 63.4% | 63.2% | 62.1% ±0.5% |
| KNN (k=7, tuned) | 63.2% | 63.5% | 63.2% | 63.2% | 61.9% ±0.7% |
| Logistic Regression | 61.7% | 61.9% | 61.7% | 61.5% | 60.8% ±0.4% |

> ✅ **Honest Interpretation:** 73.9% accuracy on a balanced 4-class problem = **~37.6 points above the 25% random baseline**. The 12-point gap between Random Forest and Logistic Regression confirms that the decision boundaries are **non-linear** — tree-based ensemble methods are the right tool here.

### ❓ Why Class B Is the Hardest to Predict

Class B achieves the lowest F1-score (~0.61) across all models. This is **not a modelling failure** — it reflects genuine real-world ambiguity. Class B participants share significant physiological overlap with both Class A (above) and Class C (below). Without additional features like VO₂ max or training history, no classifier can cleanly separate this middle tier.

---

## 8. GridSearch Hyperparameter Tuning

```python
# GridSearchCV exhaustively tests every combination using cross-validation
svm_gs = GridSearchCV(
    SVC(kernel='rbf', gamma='scale', random_state=42),
    param_grid={'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)
svm_gs.fit(X_train_s, y_train)
print(f'Best C: {svm_gs.best_params_["C"]}')     # Output: C = 10
print(f'Best CV Acc: {svm_gs.best_score_:.4f}')   # Output: 0.7090
```

| Model | Parameter Grid | Best Params | Best CV Acc |
|-------|---------------|-------------|-------------|
| K-Nearest Neighbors | n_neighbors: [3, 5, 7, 9, 11, 13, 15, 17, 19] | **k = 7** | 63.4% |
| Decision Tree | max_depth: [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, None] | **depth = 8** | 65.2% |
| SVM (RBF Kernel) | C: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] | **C = 10** | 70.9% |

---

## 9. Regression Sub-Task

In addition to classification, we trained regression models to **predict `broad_jump_cm`** — a continuous target representing explosive leg power. This demonstrates pipeline generalisability and provides coaches with quantitative performance targets.

```python
# Note: athletic_score excluded here — it contains broad_jump_cm (leakage prevention)
REG_FEATURES = ['age', 'weight_kg', 'body fat_%', 'gripForce',
                'sit and bend forward_cm', 'sit-ups counts',
                'bmi', 'gender_enc', 'height_cm']

lr  = LinearRegression()
dtr = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
```

| Model | MSE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 681.2 | 26.1 cm | 0.574 |
| ⭐ **Decision Tree Regressor** | 14.1 | **3.75 cm** | **0.991** |

> ⚠️ **Interpret carefully:** R² = 0.991 is exceptional but warrants scrutiny for overfitting. The **RMSE of 3.75 cm** is the more practically meaningful metric — on average, predictions are within 3.75 cm of the actual jump distance.

---

## 10. Unsupervised Learning — K-Means Clustering

```python
# Applied WITHOUT class labels — purely discovering hidden structure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
df['km_cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Silhouette score at k=4: 0.2529 — highest meaningful score in k=[2,9]
```

### 📊 Optimal Cluster Selection

| k | Inertia | Silhouette Score | Decision |
|---|---------|-----------------|----------|
| 2 | 98,420 | 0.312 | Too broad |
| 3 | 74,310 | 0.271 | Borderline |
| **4** | **56,722** | **0.253** | ✅ **Optimal — matches 4 classes** |
| 5 | 46,100 | 0.241 | Diminishing returns |
| 6 | 38,900 | 0.228 | Over-segmented |

> 🏆 **Key Result:** The four K-Means clusters correspond strongly to the labelled A/B/C/D classes — **without any class labels provided**. This confirms the performance classifications reflect genuinely distinct physiological groupings, not arbitrary human categories.

---

## 11. Business Recommendations

### 🎯 Priority Action Plan

1. **Lead all fitness programs with flexibility training.** `sit and bend forward` contributes **41.3%** of the Random Forest model's predictive power (21.4% raw + 19.9% flexibility ratio). Yoga, dynamic stretching, and mobility work are the highest-ROI interventions in this dataset.

2. **Flag every Class D individual for dedicated health intervention.** This group averages **27.7% body fat** and **72.0 kg** — significantly above all other classes. Combined with marginally elevated blood pressure readings, Class D represents both the greatest health risk and the largest improvement opportunity.

3. **Reconsider gender-biased assessment frameworks.** Female participants achieve elite classification at a higher rate than their dataset proportion (44.3% of Class A vs 36.8% of dataset). Assessment frameworks should weight **flexibility and endurance more heavily** than raw strength.

4. **Deploy the Random Forest model as a first-pass fitness screening tool.** At 73.9% accuracy with CV std of only ±0.8%, the model is **stable and ready for pilot deployment**. Focus especially on Class D identification (F1 = 0.86, Recall = 80%).

5. **Collect additional features to close the B/C classification gap.** Class B F1-score is only 0.61 — a genuine data limitation. Adding **VO₂ max, resting heart rate, training frequency, and sleep quality** could push overall accuracy above 85%.

6. **Incorporate engineered features into standard assessments.** BMI, flexibility ratio, strength-endurance, and athletic score collectively contribute ~26% of predictive power. These should be **computed routinely**, not left as post-hoc analysis tools.

7. **Use K-Means cluster profiles for population segmentation.** Sports facilities and corporate wellness programs can use the four profiles to segment client populations and design class-specific training protocols without requiring the full ML pipeline.

---

## 12. How to Run (Google Colab)

### Step 1 — Open Google Colab

Navigate to [colab.research.google.com](https://colab.research.google.com) and create a new notebook, or upload the provided `.ipynb` file directly.

### Step 2 — Upload the Dataset

```python
# Option A: Upload via Colab file browser (left panel → Files icon)

# Option B: Upload programmatically
from google.colab import files
uploaded = files.upload()  # Select bodyPerformance.csv
```

### Step 3 — Install Dependencies (if needed)

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy --quiet
```

### Step 4 — Run All Cells in Order

The notebook is divided into **19 clearly labelled sections**. Run each section cell-by-cell from top to bottom. Every section begins with a plain-English explanation of what is being done and why.

### Step 5 — Verify Outputs

```
✅  PROJECT COMPLETE — All sections executed successfully.

Generated figures (15 total):
    📊  01_distributions.png
    📊  02_boxplots_before.png
    📊  03_boxplots_after.png
    📊  04_correlation_heatmap.png
    📊  05_scatter_plots.png
    📊  06_categorical_frequencies.png
    📊  07_confusion_matrices.png
    📊  08_feature_importance.png
    📊  09_cross_validation.png
    📊  10_gridsearch_tuning.png
    📊  11_regression_results.png
    📊  12_kmeans_elbow.png
    📊  13_kmeans_pca.png
    📊  14_model_comparison.png
    📊  15_regression_comparison.png
```

---

## 13. Project Structure

```
body-performance-project/
│
├── 📂 data/
│   └── bodyPerformance.csv              # Raw dataset (13,393 rows × 12 columns)
│
├── 📂 notebooks/
│   └── INTELLIGENT_CLASSIFICATION_SYSTEM.ipynb   # Full 19-section notebook
│
├── 📂 scripts/
│   └── body_performance_project.py      # Colab-ready Python script
│
├── 📂 dashboard/
│   └── basyouni_dashboard.html          # Interactive HTML dashboard
│
├── 📂 figures/                          # All 15 saved visualisations
│   ├── 01_distributions.png
│   ├── 02_boxplots_before.png
│   ├── 03_boxplots_after.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_scatter_plots.png
│   ├── 06_categorical_frequencies.png
│   ├── 07_confusion_matrices.png
│   ├── 08_feature_importance.png
│   ├── 09_cross_validation.png
│   ├── 10_gridsearch_tuning.png
│   ├── 11_regression_results.png
│   ├── 12_kmeans_elbow.png
│   ├── 13_kmeans_pca.png
│   ├── 14_model_comparison.png
│   └── 15_regression_comparison.png
│
├── 📄 README.md                         # This file
└── 📄 LICENSE
```

---

## 14. Academic Integrity Declaration

This project was completed in accordance with the academic integrity requirements of the Introduction to AI and Machine Learning course. All code was written independently by the project team. Where publicly available libraries and standard ML techniques were used, they have been properly cited and credited.

> 🤖 **AI Tool Usage Disclosure:** AI-assisted tools were used during this project in accordance with course policy. All AI-generated content was critically reviewed, validated, and substantially modified by the project team. Final analysis, interpretation, and conclusions represent the team's original intellectual work.

---

<div align="center">

## 📦 Quick Stats

![Records](https://img.shields.io/badge/Records-13%2C392-2A7BE4?style=flat-square)
![Features](https://img.shields.io/badge/Features-18%20total-6F42C1?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-4%20balanced-00D27A?style=flat-square)
![Best Accuracy](https://img.shields.io/badge/Best_Accuracy-73.9%25-F9AB00?style=flat-square)
![Models](https://img.shields.io/badge/Models_Trained-6%20supervised-E63757?style=flat-square)
![Figures](https://img.shields.io/badge/Figures-15%20plots-27BCFD?style=flat-square)

---

**BASYOUNI AI ANALYTICS**

*Introduction to AI & ML Course · Body Performance Intelligent Classification System · v1.0.0*

*MIT License · 2025 · All rights reserved*

</div>

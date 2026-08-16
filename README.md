<div align="center">

# 💤 Sleep Disorder Synthetic Data Generation

### SMOTENC vs Conditional CTGAN — Synthetic Tabular Data & ML Utility Study

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SMOTENC-Imbalanced%20Learning-2E8B57?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CTGAN-Synthetic%20Data-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</p>

<p>
  <b>Generate → Validate → Compare → Evaluate → Understand</b>
</p>

</div>

---

## 🌟 Project Highlights

| 🔎 | What this project does |
|---|---|
| 📊 | Starts with a **559-record Sleep Health & Lifestyle dataset** |
| ⚖️ | Handles an imbalanced **Sleep Disorder** target |
| 🧪 | Generates synthetic data using **SMOTENC** |
| 🤖 | Generates synthetic data using **Conditional CTGAN** |
| 🔍 | Validates generated data and compares distributions |
| 📈 | Studies numerical relationships and correlations |
| 🌲 | Evaluates downstream utility using **Random Forest** |
| 🏆 | Compares **Accuracy, Precision, Recall & F1-Score** |

> **Core question:** *Can synthetic tabular data improve or preserve the usefulness of an imbalanced sleep-health dataset for machine-learning tasks?*

---

# 📌 1. About the Project

This project is a practical study of **synthetic tabular data generation** using two different approaches:

- **SMOTENC** — designed for imbalanced datasets containing categorical and numerical variables.
- **Conditional CTGAN** — a generative approach for creating synthetic tabular records while conditioning generation on the target class.

The original dataset contains **559 records and 13 original columns**. Missing `Sleep Disorder` values were filled as `Healthy`, resulting in:

```text
Healthy       375
Insomnia       93
Sleep Apnea    91
-------------------
Total         559
```

The project then explores synthetic-data generation, validation, statistical similarity, and downstream machine-learning performance.

---

# 🎯 2. Objectives

### Primary Objectives

- Understand the class imbalance in `Sleep Disorder`.
- Prepare the dataset for synthetic generation.
- Generate synthetic records using **SMOTENC**.
- Generate synthetic records using **Conditional CTGAN**.
- Validate the generated datasets.
- Compare categorical and numerical distributions.
- Compare correlation structures.
- Train a common Random Forest model.
- Measure downstream predictive utility.
- Identify the strengths and limitations of each synthetic-data approach.

---

# 🧩 3. Dataset

The project uses the **Sleep Health and Lifestyle Dataset**.

### Original Dataset

```text
Rows       : 559
Columns    : 13
Target     : Sleep Disorder
```

### Main Features

| Feature | Type |
|---|---|
| Person ID | Identifier |
| Gender | Categorical |
| Age | Numerical |
| Occupation | Categorical |
| Sleep Duration | Numerical |
| Quality of Sleep | Numerical |
| Physical Activity Level | Numerical |
| Stress Level | Numerical |
| BMI Category | Categorical |
| Blood Pressure | Categorical / Structured |
| Heart Rate | Numerical |
| Daily Steps | Numerical |
| Sleep Disorder | Target |

The project preprocessing also separates Blood Pressure into **systolic (`hp`) and diastolic (`lp`) components** for numerical processing. The source notebook shows the original dataset at 559 rows and the processed representation at 14 columns after this transformation. fileciteturn11file3

---

# ⚠️ 4. Class Imbalance

The original target distribution is:

| Sleep Disorder | Records | Share |
|---|---:|---:|
| 🟢 Healthy | 375 | 67.1% |
| 🟠 Insomnia | 93 | 16.6% |
| 🔴 Sleep Apnea | 91 | 16.3% |
| **Total** | **559** | **100%** |

These class proportions motivated the use of synthetic-data techniques. The original notebook records the same target counts after filling missing `Sleep Disorder` values with `Healthy`. fileciteturn11file2

---

# 🔄 5. Project Architecture

```text
                    ┌──────────────────────┐
                    │   Original Dataset   │
                    │      559 Records     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Data Preprocessing   │
                    │ • Missing values      │
                    │ • Encoding            │
                    │ • Blood Pressure      │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
                    ▼                      ▼
             ┌─────────────┐       ┌─────────────┐
             │   SMOTENC   │       │    CTGAN    │
             │ Oversample  │       │ Conditional │
             └──────┬──────┘       └──────┬──────┘
                    │                     │
                    ▼                     ▼
             Synthetic Data         Synthetic Data
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Synthetic Validation │
                    │ • Missing values     │
                    │ • Duplicates         │
                    │ • Ranges             │
                    │ • Distributions      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Statistical Analysis │
                    │ • Correlations       │
                    │ • Feature patterns   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Random Forest Model  │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Final Comparison     │
                    │ Accuracy / F1 / etc. │
                    └──────────────────────┘
```

---

# 🧪 6. Synthetic Data Generation

## 🔵 SMOTENC

**SMOTENC** is used because the dataset contains both numerical and categorical variables.

The project identifies categorical columns and applies SMOTENC to generate additional minority-class observations.

The notebook demonstrates a final **1,500-record** synthetic dataset after resampling. fileciteturn11file4

### SMOTENC Pipeline

```text
Original Data
     ↓
Identify categorical features
     ↓
SMOTENC
     ↓
Synthetic minority samples
     ↓
1,500-record dataset
```

---

## 🟣 Conditional CTGAN

**CTGAN** is used as a generative-model approach for synthetic tabular data.

Unlike simple oversampling, CTGAN attempts to learn patterns in the underlying tabular dataset and generate new records.

### Conditional Generation

The project uses target-conditioned generation so that the synthetic data can follow the required `Sleep Disorder` class proportions.

```text
CTGAN
  │
  ├── Healthy
  ├── Insomnia
  └── Sleep Apnea
          ↓
    Synthetic Records
```

---

# 🧹 7. Data Preparation

The project includes several preprocessing steps:

### Missing Target Values

```python
dataset['Sleep Disorder'] = \
    dataset['Sleep Disorder'].fillna("Healthy")
```

### Blood Pressure

Original representation:

```text
126/83
125/80
140/90
```

Processed representation:

```text
hp → systolic pressure
lp → diastolic pressure
```

### Categorical Features

The preparation workflow identifies categorical columns including:

```text
Gender
Occupation
BMI Category
Blood Pressure / processed components
Sleep Disorder
```

The source preparation notebook explicitly shows the 559-row dataset and categorical-feature preparation. fileciteturn11file3

---

# 📊 8. Synthetic Dataset Validation

Every synthetic dataset should be checked before being used for machine learning.

### Validation Checklist

```text
              Synthetic Dataset
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Missing Values  Duplicates    Data Types
        │             │             │
        └─────────────┼─────────────┘
                      ▼
               Value Ranges
                      │
                      ▼
             Category Validity
                      │
                      ▼
              Distribution Check
```

### Key Checks

- ✅ Number of generated records
- ✅ Missing values
- ✅ Duplicate records
- ✅ Numerical ranges
- ✅ Valid categorical values
- ✅ Target distribution
- ✅ Feature distributions
- ✅ Correlation structure

---

# 📈 9. Statistical Comparison

The project does not evaluate synthetic data only by row count.

It also compares:

### Numerical Features

```text
Age
Sleep Duration
Quality of Sleep
Physical Activity Level
Stress Level
Heart Rate
Daily Steps
Blood Pressure
```

### Categorical Features

```text
Gender
Occupation
BMI Category
Sleep Disorder
```

### Relationship Analysis

Correlation matrices are used to investigate whether important relationships in the original data are reproduced by the synthetic data.

---

# 🤖 10. Downstream Machine Learning

To test whether the generated datasets remain useful for prediction, a common machine-learning evaluation is performed.

## Model

```text
Random Forest Classifier
```

## Configuration

```text
n_estimators = 200
random_state = 42
class_weight = balanced
```

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| 🎯 Accuracy | Overall correct predictions |
| 🔎 Precision | Correct positive predictions |
| 📥 Recall | Positive cases correctly identified |
| ⭐ F1-Score | Balance between Precision and Recall |

The model-comparison notebook contains the performance-comparison workflow and visualizes Accuracy, Precision, Recall and F1-Score across the datasets.

---

# 🏆 11. Model Performance

### Comparison

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| 🟢 Original | **85.71%** | **87.54%** | **85.71%** | **85.66%** |
| 🔵 SMOTENC | **96.00%** | **96.04%** | **96.00%** | **95.98%** |
| 🟣 Conditional CTGAN | **73.00%** | **67.39%** | **73.00%** | **68.74%** |

## 🥇 Best Performing Approach

### 🔵 SMOTENC

```text
Accuracy   → 96.00%
Precision  → 96.04%
Recall     → 96.00%
F1-Score   → 95.98%
```

### Performance Ranking

```text
🥇 SMOTENC              96.00% Accuracy
🥈 Original             85.71% Accuracy
🥉 Conditional CTGAN    73.00% Accuracy
```

> **Important:** This ranking describes the result of this particular experiment, dataset, preprocessing pipeline, Random Forest configuration, and evaluation setup. It does not mean SMOTENC is universally superior to CTGAN.

---

# 📊 12. Results Visualization

The project produces visual outputs such as:

### Confusion Matrix

```text
Original
   ↓
Random Forest
   ↓
Confusion Matrix
```

### Performance Comparison

```text
             Accuracy
                 │
SMOTENC          ████████████████████ 96%
Original         █████████████████    86%
CTGAN            ██████████████       73%
```

Recommended repository output structure:

```text
results/
└── comparison/
    ├── model_comparison.csv
    ├── confusion_matrix.png
    └── performance_comparison.png
```

---

# 🆚 13. SMOTENC vs Conditional CTGAN

| Capability | SMOTENC | Conditional CTGAN |
|---|:---:|:---:|
| Handles numerical data | ✅ | ✅ |
| Handles categorical data | ✅ | ✅ |
| Synthetic records | ✅ | ✅ |
| Target-aware generation | ⚠️ Oversampling | ✅ Conditional |
| Neural generative model | ❌ | ✅ |
| Distribution analysis | ✅ | ✅ |
| Correlation analysis | ✅ | ✅ |
| Downstream ML evaluation | ✅ | ✅ |
| Best result in this experiment | 🏆 | — |

---

# 🧠 14. Key Findings

### Finding 01 — SMOTENC was strongest for downstream utility

SMOTENC achieved the highest Random Forest performance in this experiment.

### Finding 02 — Synthetic data needs more than valid rows

A synthetic dataset can have:

```text
✔ Correct number of rows
✔ Valid categories
✔ Valid numerical ranges
```

and still fail to reproduce important relationships.

### Finding 03 — CTGAN needs careful evaluation

Conditional CTGAN successfully provides a generative approach, but its downstream performance was lower in this experiment.

### Finding 04 — Synthetic-data quality is multi-dimensional

A strong synthetic-data evaluation should consider:

```text
Data Quality
     +
Statistical Similarity
     +
Relationship Preservation
     +
Machine-Learning Utility
```

---

# ⚠️ 15. Limitations

- The original dataset contains only **559 records**.
- The source dataset may limit the amount of information available to a generative model.
- Some relationships in synthetic data may differ from the original dataset.
- Categorical distributions may not match perfectly.
- CTGAN performance depends strongly on preprocessing, model configuration, training, and conditioning.
- The downstream comparison uses a Random Forest model rather than a large collection of models.
- Synthetic records should not be interpreted as real clinical/patient records.

---

# 🚀 16. Future Improvements

### 🔬 Better Synthetic Data

- Tune CTGAN hyperparameters.
- Experiment with different CTGAN training configurations.
- Compare additional synthetic-data algorithms.
- Increase the size and diversity of the source dataset.

### 🤖 Better ML Evaluation

- Test Logistic Regression.
- Test Random Forest.
- Test XGBoost.
- Test Gradient Boosting.
- Use cross-validation.
- Compare multiple random seeds.

### 📊 Better Statistical Evaluation

- Compare distributions using statistical tests.
- Compare feature-wise similarity.
- Compare covariance/correlation structures.
- Analyze rare categories separately.
- Add privacy and memorization analysis.

---

# 🛠️ 17. Technology Stack

<div align="center">

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming |
| 🐼 Pandas | Data manipulation |
| 🔢 NumPy | Numerical computation |
| ⚖️ imbalanced-learn | SMOTENC |
| 🧬 SDV / CTGAN | Synthetic tabular data |
| 🤖 Scikit-learn | Machine learning |
| 🌳 Random Forest | Downstream evaluation |
| 📊 Matplotlib | Visualization |
| ☁️ Google Colab | Notebook execution |
| 🐙 GitHub | Version control |

</div>

---

# 📁 18. Recommended Repository Structure

```text
Sleep-Disorder-Synthetic-Data-Generation/
│
├── 📄 README.md
│
├── 📂 data/
│   ├── 📂 original/
│   │   └── Sleep_health_and_lifestyle_dataset.csv
│   │
│   └── 📂 synthetic/
│       ├── SMOTENC_1500_Synthetic_Dataset.csv
│       └── CTGAN_1500_Synthetic_Dataset.csv
│
├── 📂 notebooks/
│   ├── Dataset_Preparation.ipynb
│   ├── SMOTENC_Synthetic_Data_Generation.ipynb
│   ├── CTGAN_Synthetic_Data_Generation.ipynb
│   └── model_comparison.ipynb
│
├── 📂 results/
│   └── 📂 comparison/
│       ├── model_comparison.csv
│       ├── confusion_matrix.png
│       └── performance_comparison.png
│
└── 📄 README.md
```

---

# ▶️ 19. How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Sleep-Disorder-Synthetic-Data-Generation.git
cd Sleep-Disorder-Synthetic-Data-Generation
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn sdv
```

### 3. Open the notebooks

Use:

```text
Google Colab
       OR
Jupyter Notebook
```

### 4. Execute in order

```text
Dataset Preparation
        ↓
SMOTENC Generation
        ↓
CTGAN Generation
        ↓
Synthetic Data Validation
        ↓
Model Comparison
        ↓
Results
```

---

# 📚 20. Learning Outcomes

This project demonstrates practical experience with:

```text
✓ Data Cleaning
✓ Missing-Value Handling
✓ Feature Engineering
✓ Categorical Encoding
✓ Class Imbalance
✓ SMOTENC
✓ Synthetic Tabular Data
✓ CTGAN
✓ Conditional Generation
✓ Data Validation
✓ Statistical Analysis
✓ Correlation Analysis
✓ Random Forest
✓ Model Evaluation
✓ Confusion Matrix
✓ Performance Visualization
✓ GitHub Project Organization
```

---

# 🏁 21. Conclusion

This project provides a practical comparison of **SMOTENC and Conditional CTGAN** for synthetic tabular data generation using a Sleep Health & Lifestyle dataset.

The workflow goes beyond simply generating rows. It evaluates the synthetic datasets from multiple perspectives:

```text
Original Dataset
      ↓
Preprocessing
      ↓
Synthetic Generation
      ↓
Quality Validation
      ↓
Distribution Comparison
      ↓
Correlation Analysis
      ↓
Machine Learning
      ↓
Final Evaluation
```

The most important result of this experiment is:

> 🏆 **SMOTENC achieved the strongest downstream Random Forest performance, with 96.00% accuracy and 95.98% F1-score.**

Conditional CTGAN provides a more generative approach, but its downstream performance in this particular experiment was lower.

The project therefore demonstrates an important lesson in synthetic-data generation:

**Generating synthetic records is only the beginning — their quality, statistical behavior, relationship preservation, and downstream usefulness must all be evaluated.**

---

# 👨‍💻 Author

<div align="center">

## **Sandeep Kumar Reddy Chalapala**

**Computer Science & Engineering**

### 💡 Interests

`Data Science` · `Machine Learning` · `Artificial Intelligence` · `Java` · `Synthetic Data`

---

### ⭐ If you find this project useful, consider giving the repository a star!

</div>

---

## 📌 Project Snapshot

```text
┌───────────────────────────────────────────────┐
│          SLEEP DISORDER PROJECT               │
├───────────────────────────────────────────────┤
│ Original Dataset       │ 559 records          │
│ Synthetic Dataset      │ 1,500 records        │
│ Techniques             │ SMOTENC + CTGAN      │
│ ML Model               │ Random Forest        │
│ Best Accuracy          │ 96.00%               │
│ Best F1-Score          │ 95.98%               │
│ Best Approach          │ SMOTENC              │
└───────────────────────────────────────────────┘
```

<div align="center">

### 💤 Synthetic Data • 🤖 Machine Learning • 📊 Statistical Analysis • 🧪 Experimentation

**Built to explore whether synthetic data can improve real-world ML utility.**

</div>

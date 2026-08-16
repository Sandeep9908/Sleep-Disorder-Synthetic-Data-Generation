# Sleep Disorder Synthetic Data Generation

A synthetic data generation project for sleep-health data using **SMOTENC** and **Conditional CTGAN** to address class imbalance and create balanced synthetic datasets for future machine learning experiments.

## Project Overview

This project investigates two approaches for generating synthetic data from a sleep health and lifestyle dataset:

1. **SMOTENC** — an oversampling technique designed for datasets containing numerical and categorical features.
2. **Conditional CTGAN** — a generative approach for tabular data with controlled target-class generation.

The original dataset contains **559 records**. Both approaches were used to generate **1,500-record synthetic datasets** with the same target-class distribution.

## Objectives

- Analyze the original sleep-health dataset.
- Handle missing values in `Sleep Disorder`.
- Prepare categorical and numerical features.
- Handle target-class imbalance.
- Generate synthetic data using SMOTENC.
- Generate synthetic data using Conditional CTGAN.
- Maintain a controlled target distribution.
- Validate generated datasets.
- Compare synthetic data with the original data.
- Prepare datasets for future machine-learning experiments.

## Original Dataset

The original Sleep Health and Lifestyle dataset used in this project was obtained from Kaggle:

**Dataset Source:** [Sleep Health and Lifestyle Dataset – Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

The dataset contains information about sleep, lifestyle, and health-related characteristics.

### Size

```text
559 rows
13 columns
```

### Features

| Feature | Description |
|---|---|
| Person ID | Unique identifier |
| Gender | Gender of the individual |
| Age | Age of the individual |
| Occupation | Occupation category |
| Sleep Duration | Average sleep duration |
| Quality of Sleep | Sleep quality rating |
| Physical Activity Level | Daily physical activity level |
| Stress Level | Stress rating |
| BMI Category | BMI classification |
| Heart Rate | Resting heart rate |
| Daily Steps | Number of daily steps |
| Sleep Disorder | Target variable |
| Blood Pressure | Systolic/diastolic blood pressure |

## Class Imbalance

After treating missing `Sleep Disorder` values as `Healthy`:

| Sleep Disorder | Records |
|---|---:|
| Healthy | 375 |
| Insomnia | 93 |
| Sleep Apnea | 91 |
| **Total** | **559** |

## Data Preparation

### Missing Values

Missing values in `Sleep Disorder` were treated as `Healthy`.

### Blood Pressure

The original combined `Blood Pressure` field was split into:

```text
Systolic Pressure
Diastolic Pressure
```

After generation, these were reconstructed into the original `Blood Pressure` format.

### Person ID

`Person ID` was removed from CTGAN training because it is an identifier rather than a meaningful feature. It was restored after generation with IDs from `1` to `1500`.

## Methodology

```text
Original Dataset
      │
      ↓
Data Preparation
      │
 ┌────┴────┐
 ↓         ↓
SMOTENC   CTGAN
 ↓         ↓
1,500     Conditional
Records   1,500 Records
 └────┬────┘
      ↓
Validation & Comparison
```

## 1. SMOTENC

SMOTENC is designed for oversampling datasets containing both numerical and categorical features.

Categorical features used include:

```text
Gender
Occupation
BMI Category
Sleep Disorder
```

### Target Distribution

| Sleep Disorder | Records |
|---|---:|
| Healthy | 1,006 |
| Sleep Apnea | 244 |
| Insomnia | 250 |
| **Total** | **1,500** |

Output:

```text
SMOTENC_1500_Synthetic_Dataset.csv
```

## 2. Conditional CTGAN

The project uses the SDV implementation:

```python
CTGANSynthesizer
```

An unrestricted CTGAN experiment was initially tested, but its generated target distribution did not meet the required distribution. The final approach therefore uses **conditional generation**.

### Conditions

```text
Healthy      → 1,006
Sleep Apnea  →   244
Insomnia     →   250
```

Total:

```text
1,500 records
```

Output:

```text
CTGAN_1500_Synthetic_Dataset.csv
```

## Conditional CTGAN Result

| Sleep Disorder | Records |
|---|---:|
| Healthy | 1,006 |
| Sleep Apnea | 244 |
| Insomnia | 250 |
| **Total** | **1,500** |

## Synthetic Dataset Validation

Both final datasets were validated.

| Validation | Result |
|---|---:|
| Records | 1,500 |
| Columns | 13 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Person ID | 1–1,500 |

### Numerical Range Validation

| Feature | Original Range | CTGAN Range |
|---|---:|---:|
| Age | 27–59 | 27–59 |
| Sleep Duration | 5.8–8.5 | 5.8–8.5 |
| Quality of Sleep | 4–9 | 4–9 |
| Physical Activity Level | 30–90 | 30–90 |
| Stress Level | 3–8 | 3–8 |
| Heart Rate | 65–86 | 65–86 |
| Daily Steps | 3,000–10,000 | 3,000–10,000 |
| Systolic Pressure | 115–142 | 115–142 |
| Diastolic Pressure | 75–95 | 75–95 |

The Conditional CTGAN values remained within the observed ranges of the original dataset.

## Statistical Comparison

Selected numerical correlations:

| Relationship | Original | Conditional CTGAN |
|---|---:|---:|
| Sleep Duration ↔ Quality of Sleep | 0.84 | 0.15 |
| Quality of Sleep ↔ Stress Level | -0.89 | -0.10 |
| Sleep Duration ↔ Stress Level | -0.78 | -0.13 |
| Physical Activity ↔ Daily Steps | 0.75 | 0.10 |
| Systolic ↔ Diastolic Pressure | 0.97 | 0.31 |

### Interpretation

The Conditional CTGAN successfully reproduced dataset size, target distribution, numerical ranges, and valid categorical values. However, several strong numerical relationships from the original dataset were not reproduced closely.

Therefore, the CTGAN dataset should be considered a **synthetic balanced dataset**, not an exact statistical replica of the original dataset.

## Categorical Distribution Validation

### Gender

Original:

```text
Male      57.2%
Female    42.8%
```

Conditional CTGAN:

```text
Male      66.7%
Female    33.3%
```

### BMI Category

Original:

```text
Normal           60.6%
Overweight       30.4%
Normal Weight     6.8%
Obese             2.1%
```

Conditional CTGAN:

```text
Normal           54.5%
Overweight       29.2%
Normal Weight    10.7%
Obese             5.6%
```

### Sleep Disorder

Original:

```text
Healthy       67.1%
Insomnia      16.6%
Sleep Apnea   16.3%
```

Conditional CTGAN:

```text
Healthy       67.1%
Insomnia      16.7%
Sleep Apnea   16.3%
```

The target distribution was preserved very closely.

## SMOTENC vs Conditional CTGAN

| Aspect | SMOTENC | Conditional CTGAN |
|---|---|---|
| Handles categorical features | Yes | Yes |
| Handles numerical features | Yes | Yes |
| Generates 1,500 records | Yes | Yes |
| Target distribution controlled | Yes | Yes |
| Missing values | 0 | 0 |
| Duplicate rows | 0 | 0 |
| Numerical ranges validated | Yes | Yes |
| Generative model | No | Yes |
| Conditional generation | No | Yes |
| Main purpose | Oversampling | Synthetic tabular generation |

**SMOTENC** provides a practical baseline for class balancing.

**Conditional CTGAN** provides an advanced generative approach where the target distribution can be explicitly controlled during generation.

The final choice should ultimately be based on downstream machine-learning performance rather than synthetic-data statistics alone.

## Project Structure

```text
Sleep-Disorder-Synthetic-Data-Generation/
│
├── README.md
│
├── data/
│   ├── original/
│   │   └── Sleep_health_and_lifestyle_dataset.csv
│   │
│   └── synthetic/
│       ├── SMOTENC_1500_Synthetic_Dataset.csv
│       └── CTGAN_1500_Synthetic_Dataset.csv
│
├── notebooks/
│   ├── SMOTENC_Synthetic_Data_Generation.ipynb
│   └── CTGAN_Synthetic_Data_Generation.ipynb
│
└── results/
    └── comparison/
```

## Technologies Used

- Python
- Pandas
- NumPy
- imbalanced-learn
- SMOTENC
- SDV
- CTGAN
- Google Colab
- GitHub

## Main Libraries

```python
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTENC

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/<your-username>/Sleep-Disorder-Synthetic-Data-Generation.git
cd Sleep-Disorder-Synthetic-Data-Generation
```

Open either notebook:

```text
notebooks/SMOTENC_Synthetic_Data_Generation.ipynb
```

or:

```text
notebooks/CTGAN_Synthetic_Data_Generation.ipynb
```

The notebooks can be executed using Google Colab or a compatible Jupyter environment.

## Experimental Workflow

```text
Original Sleep Health Dataset
              │
              ↓
       Data Preprocessing
              │
      ┌───────┼────────┐
      ↓       ↓        ↓
   Missing  Blood    Person ID
    Values  Pressure  Handling
              │
              ↓
      ┌───────┴────────┐
      ↓                ↓
   SMOTENC          CTGAN
      │                │
      ↓                ↓
 1,500 Records     Conditional
                   1,500 Records
      │                │
      └───────┬────────┘
              ↓
       Dataset Validation
              │
              ↓
      Distribution Analysis
              │
              ↓
      Correlation Analysis
              │
              ↓
       Final Synthetic Data
```

## Limitations

1. **Small Original Dataset** — the original dataset contains only 559 records.
2. **CTGAN Correlation Preservation** — several strong numerical correlations were not reproduced closely.
3. **Categorical Distribution Differences** — some categorical proportions differed from the original data.
4. **Synthetic Data Is Not Real Patient Data** — generated records should not be interpreted as actual medical records or clinical observations.
5. **Downstream Model Evaluation** — the synthetic datasets should ultimately be evaluated using machine-learning models.

## Future Work

- Train classification models on the original dataset.
- Train the same models on SMOTENC data.
- Train the same models on Conditional CTGAN data.
- Compare Accuracy, Precision, Recall, and F1-score.
- Generate confusion matrices.
- Evaluate cross-validation performance.
- Compare feature distributions using visualizations.
- Perform statistical similarity testing.
- Experiment with different CTGAN configurations.
- Investigate other synthetic-data techniques.
- Perform privacy and memorization analysis.
- Evaluate synthetic data using downstream task performance.

## Key Learning Outcomes

- Class imbalance
- Synthetic data generation
- SMOTENC
- CTGAN
- Conditional generation
- Categorical feature handling
- Numerical feature handling
- Data preprocessing
- Distribution analysis
- Correlation analysis
- Synthetic dataset validation
- Tabular generative modeling
- Experimental comparison

## Final Results

### SMOTENC

```text
SMOTENC_1500_Synthetic_Dataset.csv

Records: 1,500
Missing values: 0
Duplicate rows: 0
```

### Conditional CTGAN

```text
CTGAN_1500_Synthetic_Dataset.csv

Records: 1,500
Missing values: 0
Duplicate rows: 0
```

Both datasets use the target distribution:

```text
Healthy       1006
Sleep Apnea    244
Insomnia       250
```

## Conclusion

This project demonstrates two approaches to synthetic data generation for an imbalanced sleep-health dataset.

**SMOTENC** provides a practical baseline for generating additional samples while handling both categorical and numerical features.

**Conditional CTGAN** demonstrates a generative approach capable of producing synthetic tabular records while allowing the target-class distribution to be controlled during generation.

The experiments show that the Conditional CTGAN successfully generated 1,500 records with the required target distribution and maintained valid numerical ranges. However, several relationships and categorical distributions from the original dataset were not reproduced exactly.

The project therefore evaluates synthetic data through multiple validation criteria rather than treating synthetic-data generation as simply producing more rows.

The next stage is to evaluate the synthetic datasets using downstream machine-learning models and determine which approach provides the best predictive performance.

## Author

**Sandeep Kumar Reddy Chalapala**

Computer Science & Engineering

### Interests

- Data Science
- Machine Learning
- Java Development
- Artificial Intelligence
- Synthetic Data Generation

---

### Project Focus

```text
Data → Preprocessing → Synthetic Generation → Validation → Comparison → ML Evaluation
```

**Built as an experimental project to study synthetic tabular data generation using SMOTENC and Conditional CTGAN.**

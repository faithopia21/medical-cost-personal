# Medical Insurance Cost Prediction

## Overview

Healthcare costs can vary substantially between individuals, and understanding the factors associated with those differences is important for both healthcare planning and cost prediction.

This project explores how patient characteristics such as age, BMI, smoking status, number of children, sex, and region relate to individual medical insurance charges using the Medical Cost Personal dataset. The analysis combines exploratory data analysis with predictive modelling to identify key cost drivers and evaluate how well these factors can be used to estimate healthcare costs.

The project explores a central question:

**Which patient characteristics are most strongly associated with medical insurance charges, and how effectively can those relationships be used for prediction?**

---

## Why This Matters

Healthcare spending is influenced by multiple factors, including demographic, behavioural, and health-related characteristics. Understanding these relationships can help explain cost variation and improve prediction approaches.

Rather than treating unusually high medical charges as noise, this project examines whether those observations represent identifiable patterns within the data.

---

## Dataset

- Source: Medical Cost Personal Dataset (Kaggle)
- Total records: 1,338
- Missing values: None
- Population: Individual insurance beneficiaries in the United States

Dataset:

https://www.kaggle.com/datasets/mirichoi0218/insurance

### Variables Used

| Variable | Description |
|---|---|
| age | Age of the patient |
| sex | Female or male |
| bmi | Body Mass Index |
| children | Number of dependants covered |
| smoker | Smoking status |
| region | Residential region |
| charges | Individual medical costs billed (target variable) |

---

## Tools Used

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- SHAP
- Streamlit

---

## Repository Structure
```
medical-insurance-cost-prediction/

│

├── data/  

├── models/  
│   ├── medical_cost_model.joblib  
│   └── scaler.joblib  

├── notebooks/  
│   └── medical_cost_analysis.ipynb  

├── visuals/  
│   ├── Charges by Sex&Region&Children.png  
│   ├── Charges by Smoking Status.png  
│   ├── Charges vs Age by Smoking Status.png  
│   ├── Charges vs BMI by Smoking Status.png  
│   ├── Distribution of Medical Charges by Smoking Status (Violin Plot).png  
│   ├── Distribution of Medical Charges.png  
│   └── Predicted vs Actual Charges.png  

├── app.py  

├── README.md

└── requirements.txt  
```
---

## Key Findings

### Distribution of Medical Charges

Medical charges showed a strong right-skewed distribution.

- Mean charge: $13,270
- Median charge: $9,382

The difference between the mean and median suggested that a smaller group of individuals had substantially higher healthcare costs.

---

### Smoking Status

Smoking status emerged as the strongest predictor of medical charges.

Median charges among smokers were approximately four to five times higher than charges among non-smokers:

- Smokers: approximately $34,000
- Non-smokers: approximately $7,000 to $8,000

A violin plot also showed that smoker charges separated into two distinct groups rather than forming a single continuous distribution.

---

### BMI and Smoking Interaction

Further analysis showed that BMI explained much of this separation.

Smokers with BMI values at or above 30, which corresponds to the standard obesity threshold, occupied a substantially higher charge range:

- BMI < 30: approximately $13,000 to $30,000
- BMI ≥ 30: approximately $33,000 to $63,000

This pattern was not observed among non-smokers.

---

### Age and Charges

Age showed a clear positive relationship with medical charges across both smoker and non-smoker groups.

Medical costs generally increased with age.

---

### Weak Predictors

Sex, region, and number of children showed relatively weak relationships with medical charges.

The groups representing four and five dependants contained very few records, suggesting that any observed differences may reflect small sample effects rather than meaningful patterns.

---

## Data Preprocessing

The following preprocessing steps were applied:

- One-hot encoding for `sex`, `smoker`, and `region`
- `drop_first=True` used to avoid multicollinearity
- 80/20 train-test split performed before scaling to avoid data leakage
- `StandardScaler` applied to `age`, `bmi`, and `children`
- Scaling parameters fitted on training data only

---

## Modelling Approach

### Baseline Model

Multiple Linear Regression

Performance:

- R²: 0.7838
- MAE: $4,176.27
- RMSE: $5,793.66

Residual analysis showed that the model systematically over-predicted lower-BMI smokers and under-predicted higher-BMI smokers.

---

### Improved Model

Smoker × BMI interaction term added

Performance:

- R²: 0.8646
- MAE: $2,759.93
- RMSE: $4,585.54

Adding an interaction term allowed the model to capture the changing effect of BMI across smoking groups.

This improved performance across all evaluation metrics.

---

### Regularisation Check

Ridge and Lasso regression models were evaluated to assess potential overfitting.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Baseline | 0.7838 | $4,176.27 | $5,793.66 |
| + Interaction Term | 0.8646 | $2,759.93 | $4,585.54 |
| Ridge | 0.8647 | $2,780.07 | $4,583.50 |
| Lasso | 0.8646 | $2,760.11 | $4,585.54 |

Regularisation produced minimal improvement, suggesting that the interaction-term model was already stable.

Final model:

**Linear Regression with smoker × BMI interaction term**

---

## Limitations

- A group of non-smoker records with unexpectedly high medical charges remained unexplained by the available variables.
- Sex, region, and number of children did not account for these observations.
- Additional variables such as medical history, chronic conditions, and lifestyle factors may improve interpretation.

---

## Future Directions

Potential extensions include:

- Log transformation of target values
- Threshold-based interaction terms
- Expanded SHAP analysis across the full dataset
- Alternative modelling approaches
- Inclusion of additional health-related variables

---

## Interactive Application

A Streamlit application was developed for interactive prediction.

Features include:

- User input interface for patient attributes
- Live prediction generation
- SHAP-based prediction explanations
- Real-time performance metrics

---
## Author

**Faith Olaniyi**  
Computer Science Graduate | Data Science & AI for Healthcare  
[LinkedIn](https://www.linkedin.com/in/faith-oluwanifemi-olaniyi) · [GitHub](https://github.com/faithopia21)

## About the Author

Faith Olaniyi is a Computer Science graduate with interests in healthcare analytics, public health data, machine learning, and AI applications in healthcare. She is currently building research and technical experience through health-focused data science projects while preparing for graduate studies in Data Science and Artificial Intelligence.
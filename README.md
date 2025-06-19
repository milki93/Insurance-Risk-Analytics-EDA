# Insurance Risk Segmentation & Claim Prediction

This project focuses on analyzing insurance customer and vehicle data to detect high-risk customers, estimate claim severity, and support data-driven pricing strategies. The pipeline includes data cleaning, feature engineering, statistical testing, predictive modeling, and interpretability using modern ML techniques.

---

## Objective

Build a robust and explainable model that:
- Identifies high-risk customers based on historical claims
- Estimates the potential severity of insurance claims
- Supports segmentation strategies for personalized pricing
- Provides business insights through model interpretability

---

## Dataset Summary

- Over 1 million policy records
- Features include customer demographics, vehicle specs, insurance coverage, premiums, and claims
- Target variables:
  - `HasClaim`: binary classification (claim occurred or not)
  - `HighRisk`: derived from loss ratio (claims > premiums)
  - `TotalClaims`: continuous value for regression modeling

---

## Data Preparation

- Missing values handled using:
  - Mode imputation (e.g., `Bank`, `MaritalStatus`)
  - Predictive models (Random Forest) for fields like `Gender`, `CustomValueEstimate`, `WrittenOff`, etc.
- Engineered features:
  - `VehicleAge` from registration year
  - `LossRatio` = `TotalClaims` / `TotalPremium`
  - `Margin` = `TotalPremium` − `TotalClaims`

---

## Statistical Hypothesis Testing

Validated risk segmentation assumptions using statistical tests on key features:

- Compared claim frequency and margin across:
  - Provinces
  - Zip codes
  - Gender groups

Used appropriate statistical methods:
- Chi-squared test for claim frequency
- T-test for claim severity and margin differences

These insights support customized pricing strategies and product targeting.

---

## Predictive Modeling

Implemented multiple models for both classification and regression:

### Classification
- Random Forest Classifier
- XGBoost Classifier

### Regression
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

Evaluated each model using:
- Classification: Accuracy, Precision, Recall, F1-score
- Regression: RMSE, R² Score

---

## Model Interpretability

Used modern interpretability techniques to understand model decisions:

- **Permutation Importance** to rank influential features globally
- **SHAP values** to explain both global and local predictions
- **Partial Dependence Plots (PDPs)** to visualize the marginal effect of key features

These tools helped explain why the models made certain predictions, supporting transparency and trust.

---

## Key Takeaways

- Classification models can effectively identify high-risk profiles with strong recall.
- Regression models offer limited accuracy but may be used for rough estimations.
- Features like `VehicleAge`, `Province`, and `SumInsured` consistently influence predictions.
- Further performance improvement is possible through:
  - Feature engineering
  - Class rebalancing
  - Cost-sensitive learning

---

## Tools & Technologies

- Python (pandas, scikit-learn, xgboost, shap, seaborn, matplotlib)
- Jupyter Notebooks
- Git & GitHub
- DVC for data version control and reproducibility




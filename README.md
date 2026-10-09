# Heart Disease Risk Prediction

A machine-learning project that estimates whether a person is at risk of heart disease from basic health and lifestyle answers (age group, BMI, smoking, diabetes, general health, etc.), with a bilingual (English/Arabic) web app anyone can try.

**🔗 Live demo:** [https://heart-disease-prediction-sn2cjggr7k4vlwdzgkzf77.streamlit.app](https://heart-disease-prediction-sn2cjggr7k4vlwdzgkzf77.streamlit.app) (English / Arabic interface)

> **Disclaimer:** this is a learning project built on survey data. It is **not** a medical tool and must not be used for diagnosis.

## Dataset
[Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) (CDC 2020 survey, ~320K rows, 17 features). Only ~9% of respondents report heart disease, so the classes are heavily imbalanced.

## Approach
1. **EDA & cleaning**: checked nulls and distributions, removed BMI outliers (IQR rule) and duplicate rows.
2. **Feature engineering**: encoded yes/no columns, ordinal encoding for `GenHealth` and `Diabetic`, binned `BMI`, `AgeCategory`, `SleepTime`, `PhysicalHealth`, `MentalHealth`; dropped `Race`.
3. **Split first**: stratified 80/20 train/test split on the original (imbalanced) data.
4. **Modeling**: Random Forest with `class_weight='balanced'` and `min_samples_leaf=50`.
5. **Decision threshold** chosen on the precision-recall curve (best F1 for the heart-disease class).

## What I learned (and fixed)
My first version oversampled the minority class **before** splitting, which put duplicated rows in both train and test (data leakage) and gave a misleading ~84% accuracy on a balanced test set.
After fixing it, I evaluated on an untouched, realistic test set and replaced oversampling with class weighting, which stopped the forest from memorizing duplicated rows.

## Results (test set, 58,266 rows, ~9% positive)

| | Baseline (oversampling) | Final (class weights) |
|---|---|---|
| ROC-AUC | 0.723 | **0.835** |
| PR-AUC | 0.199 | **0.344** |
| Precision (heart disease) | 0.21 | **0.33** |
| Recall (heart disease) | 0.48 | 0.49 |
| F1 (heart disease) | 0.29 | **0.39** |

Accuracy alone is misleading here: predicting "no disease" for everyone would score ~91%.

## Limitations
- Features come from self-reported phone surveys, so signal is limited (ROC-AUC 0.835).
- Precision is 0.33: most positive predictions are false alarms. Recall is 0.49: about half of true cases are caught.
- Data is US-only and from 2020.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
`heart_model.joblib` is produced by the notebook (`heart_disease_clean.ipynb`, last cell).

## Project structure
```
heart_disease_clean.ipynb   # EDA, preprocessing, training, evaluation
heart_model.joblib          # trained model + threshold + column order
app.py                      # Streamlit web app
requirements.txt
```

## Tech
Python, pandas, scikit-learn, imbalanced-learn, seaborn/matplotlib, Streamlit.

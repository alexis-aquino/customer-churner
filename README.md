# Customer Churn Prediction

An end-to-end machine learning project predicting customer churn for a telecom company.
Built over 28 days covering the full data science workflow — from raw data to a deployed
interactive dashboard.

---

## Results

| Metric | Score |
|--------|-------|
| Recall | 96.5% |
| Precision | 36.2% |
| F1 Score | 52.7% |
| ROC-AUC | 0.8037 |
| Net Annual Savings | $130,830 |
| Cost Reduction | 44.8% |

---

## Key Findings

1. **Contract type is the root cause of churn** — month-to-month customers churn at
   near 100% while annual contract customers have close to 0% churn rate.
2. **Price has zero impact** — monthly charges had a 0.001 correlation with churn.
   Competing on price alone will not reduce churn.
3. **The first 12 months are critical** — all churn happens within the first year.
   After 12 months, customers become highly stable.
4. **Threshold tuning was the biggest single improvement** — lowering the decision
   threshold from 50% to 30% increased Recall by +15.2 percentage points.

---

## Project Structure

```
customer-churner/
├── data/
│   ├── churn.csv
│   ├── churn_cleaned.csv
│   ├── churn_encoded.csv
│   └── churn_scaled.csv
├── models/
│   ├── final_xgboost_model.pkl
│   ├── feature_list.pkl
│   └── optimal_threshold.pkl
├── notebooks/
│   ├── 01_load_data.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── reports/
│   └── final_evaluation_report.png
├── src/
│   └── app.py
└── requirements.txt
```
---

## Model Pipeline
Raw Data (7,032 customers)
→ Cleaning (missing values, standardization)
→ Encoding (one-hot encoding for categoricals)
→ Scaling (MinMaxScaler for numerics)
→ Modeling (XGBoost with scale_pos_weight)
→ Tuning (GridSearchCV, 162 combinations)
→ Threshold Tuning (30% optimal cutoff)
→ Evaluation (96.5% Recall on held-out test set)

---

## Tech Stack

- **Python 3.x**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualization
- **scikit-learn** — preprocessing, modeling, evaluation
- **xgboost** — gradient boosting
- **shap** — model explainability
- **streamlit** — interactive dashboard
- **joblib** — model serialization

---

## Running the Dashboard

```bash
# 1. Clone the repo
git clone https://github.com/alexis-aquino/customer-churner.git
cd customer-churner

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate          # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run src/app.py
```

---

## Project Timeline

| Week | Focus | Status |
|------|-------|--------|
| Week 1 (Days 1-7) | Data Exploration & EDA | Complete |
| Week 2 (Days 8-14) | Feature Engineering & Modeling | Complete |
| Week 3 (Days 15-21) | Hyperparameter Tuning & Explanation | Complete |
| Week 4 (Days 22-28) | Evaluation & Deployment | Complete |

---

## Dataset

[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,032 customers, 20 features, 26.6% churn rate.

---

*Built by Alexis Aquino — University of Batangas*

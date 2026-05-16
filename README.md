# Loan Approval Predictor 🏦

Predicts bank loan approval instantly using
Gradient Boosting trained on real loan data.

## Live Demo
[Click here](https://loan-approval-predictor-rvn757h6kzzdjpnjnew3rx.streamlit.app)

## Features
- Instant loan eligibility check
- Approval probability score
- Debt-to-income ratio calculator
- Personalized improvement tips
- Key approval factor analysis
- Model performance dashboard

## Model Details
- Algorithm: Gradient Boosting
- Accuracy: ~82%
- AUC-ROC: ~0.80
- Features: Credit history, income, property area

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

## Tools Used
- Python, Scikit-learn, Streamlit, Pandas

## How to Run Locally
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
python3 train_model.py
streamlit run app.py

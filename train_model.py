import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier)
from sklearn.linear_model import (
    LogisticRegression)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(
    os.path.abspath(__file__)))

# Load data
print("Loading dataset...")
df = pd.read_csv('loan.csv')
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:")
print(df.isnull().sum())

# Drop Loan_ID
df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values
df['Gender'].fillna(
    df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(
    df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(
    df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(
    df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(
    df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(
    df['Loan_Amount_Term'].mode()[0],
    inplace=True)
df['Credit_History'].fillna(
    df['Credit_History'].mode()[0],
    inplace=True)

print(f"\nAfter cleaning: {df.shape}")
print(f"Approval rate: "
      f"{(df['Loan_Status']=='Y').mean():.2%}")

# Encode
le_dict = {}
cat_cols = df.select_dtypes(
    include=['object']).columns.tolist()
cat_cols.remove('Loan_Status')

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

df['Loan_Status'] = (
    df['Loan_Status'] == 'Y').astype(int)

# Feature engineering
df['Total_Income'] = (
    df['ApplicantIncome'] +
    df['CoapplicantIncome'])
df['Income_to_Loan'] = (
    df['Total_Income'] /
    (df['LoanAmount'] + 1))
df['EMI'] = (
    df['LoanAmount'] /
    df['Loan_Amount_Term'])

print(f"\nFeatures: {df.shape[1]-1}")

# Split
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# Train models
models = {
    'Logistic Regression':
        LogisticRegression(max_iter=1000),
    'Random Forest':
        RandomForestClassifier(
            n_estimators=100,
            random_state=42),
    'Gradient Boosting':
        GradientBoostingClassifier(
            n_estimators=100,
            random_state=42)
}

results = {}
print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test,
        model.predict_proba(
            X_test)[:, 1])
    cv    = cross_val_score(
        model, X, y,
        cv=5,
        scoring='accuracy').mean()

    results[name] = {
        'model': model,
        'preds': preds,
        'acc':   round(acc * 100, 2),
        'auc':   round(auc, 4),
        'cv':    round(cv * 100, 2)
    }
    print(f"\n{name}:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  CV Acc   : {cv*100:.2f}%")

# Best model
best_name = max(
    results,
    key=lambda x: results[x]['auc'])
best      = results[best_name]
print(f"\nBest: {best_name} "
      f"(AUC={best['auc']})")

# Plot 1 — Model comparison
fig, axes = plt.subplots(
    1, 3, figsize=(14, 5))
names  = list(results.keys())
accs   = [results[n]['acc'] for n in names]
aucs   = [results[n]['auc'] for n in names]
cvs    = [results[n]['cv']  for n in names]
colors = ['#3498db', '#2ecc71', '#e74c3c']

for ax, vals, title, ylim in zip(
    axes,
    [accs, aucs, cvs],
    ['Accuracy (%)',
     'AUC-ROC',
     'CV Accuracy (%)'],
    [(70, 85), (0.7, 0.9), (70, 85)]
):
    bars = ax.bar(names, vals,
                  color=colors,
                  edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() +
            bar.get_width()/2,
            bar.get_height() + 0.002,
            f'{val}',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
    ax.set_title(title, fontsize=12)
    ax.set_ylim(ylim)
    plt.setp(
        ax.xaxis.get_majorticklabels(),
        rotation=15)

plt.suptitle(
    'Model Comparison — Loan Approval',
    fontsize=14)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nModel comparison saved!")

# Plot 2 — Confusion Matrix
cm = confusion_matrix(
    y_test, best['preds'])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Greens',
            xticklabels=['Rejected',
                         'Approved'],
            yticklabels=['Rejected',
                         'Approved'])
plt.title(
    f'Confusion Matrix — {best_name}',
    fontsize=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved!")

# Plot 3 — Feature importance
if hasattr(best['model'],
           'feature_importances_'):
    feat_imp = pd.Series(
        best['model'].feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    colors_fi = plt.cm.RdYlGn(
        np.linspace(0.3, 1.0,
                    len(feat_imp)))
    plt.barh(feat_imp.index,
             feat_imp.values,
             color=colors_fi,
             edgecolor='black')
    plt.title(
        'Feature Importance — '
        'Loan Approval',
        fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance saved!")

# Plot 4 — Approval by key factors
fig, axes = plt.subplots(
    1, 3, figsize=(14, 4))

# Credit history
cr_approval = df.groupby(
    'Credit_History')[
    'Loan_Status'].mean() * 100
axes[0].bar(
    ['Bad Credit', 'Good Credit'],
    cr_approval.values,
    color=['#e74c3c', '#2ecc71'],
    edgecolor='black'
)
axes[0].set_title(
    'Approval by Credit History',
    fontsize=11)
axes[0].set_ylabel('Approval Rate (%)')

# Property area
prop_map = {0: 'Rural',
            1: 'Semiurban',
            2: 'Urban'}
prop_approval = df.groupby(
    'Property_Area')[
    'Loan_Status'].mean() * 100
axes[1].bar(
    [prop_map.get(i, str(i))
     for i in prop_approval.index],
    prop_approval.values,
    color=['#3498db', '#9b59b6',
           '#f39c12'],
    edgecolor='black'
)
axes[1].set_title(
    'Approval by Property Area',
    fontsize=11)
axes[1].set_ylabel('Approval Rate (%)')

# Education
edu_map = {0: 'Graduate',
           1: 'Not Graduate'}
edu_approval = df.groupby(
    'Education')[
    'Loan_Status'].mean() * 100
axes[2].bar(
    [edu_map.get(i, str(i))
     for i in edu_approval.index],
    edu_approval.values,
    color=['#2ecc71', '#e74c3c'],
    edgecolor='black'
)
axes[2].set_title(
    'Approval by Education',
    fontsize=11)
axes[2].set_ylabel('Approval Rate (%)')

plt.suptitle(
    'Loan Approval Insights',
    fontsize=14)
plt.tight_layout()
plt.savefig('approval_insights.png')
print("Approval insights saved!")

# Save
feature_names = X.columns.tolist()
with open('model.pkl',    'wb') as f:
    pickle.dump(best['model'], f)
with open('le_dict.pkl',  'wb') as f:
    pickle.dump(le_dict, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"\nAll files saved!")
print(f"Best model : {best_name}")
print(f"AUC-ROC    : {best['auc']}")
print("Run app.py next!")
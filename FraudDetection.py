import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter("error", category=RuntimeWarning)

# *** Read data ***
file = 'DataSet.xlsx' # Convert .xlsb to .xlsx
claims = pd.read_excel(file, sheet_name='1. Claims Dataset')
policy = pd.read_excel(file, sheet_name='2. Policy Dataset')
vehicle = pd.read_excel(file, sheet_name='3. Vehicle Dataset')
driver = pd.read_excel(file, sheet_name='4. Driver Dataset')

# Remove duplicate records and unusable driver keys
claims = claims.drop_duplicates().drop_duplicates("Claim Number")
policy = policy.drop_duplicates().drop_duplicates("Policy Number")
vehicle = vehicle.drop_duplicates().drop_duplicates("Vehicle Key")
driver = (driver.drop_duplicates().dropna(subset=["Driver Key"]).drop_duplicates("Driver Key"))

df = claims.merge(policy.drop(columns=["Company", "Province"]),on="Policy Number", how="left", validate="m:1")
df = df.merge(vehicle, on="Vehicle Key", how="left", validate="m:1")
df = df.merge(driver, on="Driver Key", how="left", validate="m:1")

# *** Data preparation & Analysis ***

for c in ["Loss Date", "Report Date", "Policy Effective Date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df["Report Delay Days"] = (df["Report Date"]-df["Loss Date"]).dt.days
df["Policy Age Days"] = (df["Loss Date"]-df["Policy Effective Date"]).dt.days
df["Loss Month"] = df["Loss Date"].dt.month
df["Loss Weekday"] = df["Loss Date"].dt.dayofweek
df["Loss Year"] = df["Loss Date"].dt.year

# Create note indicators
notes = df["Claim Notes"].fillna("").astype(str).str.lower()
for term in ["injury", "moderate", "major", "minor", "rear-ended", "sideswiped", "intersection"]:
    df["Note " + term] = notes.str.contains(term, regex=False).astype(int)

# Normalized match labels
df["Gender"] = df["Gender"].replace({"Girl":"Female", "Woman":"Female", "F":"Female","Boy":"Male", "Man":"Male", "M":"Male"})
df["Marital_Status"] = df["Marital_Status"].replace({"M":"Married", "S":"Single", "Marr":"Married"})
df["INCOME"] = df["INCOME"].replace({"Mid Income":"Middle Income"})

def to_dollar(v):
    if pd.isna(v): 
        return np.nan
    s = str(v).strip().lower().replace(",", "").replace("$", "")
    try: 
        return float(s[:-1])*1000 if s.endswith("k") else float(s)
    except ValueError: 
        return np.nan

df["Coverage Amount"] = df["Coverage Amount"].map(to_dollar)
target = "Fraud Indicator"


# Remove empty columns names
df = df.drop(columns=[col for col in df.columns if str(col).strip() == "" or "Unnamed" in str(col)])

#  Excluding IDs, constants, raw notes/dates, and payment/reserve fields
exclude = {target, "Company", "Province", "Policy Number", "Claim Number", "Driver Key", "Vehicle Key", "Claim Notes", "Loss Date", "Report Date", "Policy Effective Date", "Paid Loss", "Case Loss Amount", "Expense Loss"}

features = [c for c in df.columns if c not in exclude]
X, y = df[features].copy()
df[target].astype(int)
X["Number of Vehicles"] = pd.to_numeric(X["Number of Vehicles"], errors="coerce")

cat_feat = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
num_feat = [c for c in features if c not in cat_feat]
for c in cat_feat: 
    X[c] = X[c].fillna("Missing").astype(str)
for c in num_feat:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X[c] = X[c].fillna(X[c].median())    
train = df["Loss Year"] <= 2023
valid = df["Loss Year"] == 2024
test = df["Loss Year"] == 2025

unique_claims = X.shape[0]
fraud_claims = len(df[df[target] == 1])
non_fraud_claims = len(df[df[target] == 0]) 
fraud_rept_delay = df.loc[df[target] == 1, 'Report Delay Days'].mean()
non_fraud_rept_delay = df.loc[df[target] == 0, 'Report Delay Days'].mean()
incurred_loss_fraud = df.loc[df[target] == 1, 'Incurred Loss'].mean()
incurred_loss_non_fraud = df.loc[df[target] == 0, 'Incurred Loss'].mean()
fraud_dr_age = df.loc[df[target] == 1, 'Age'].mean()
non_fraud_dr_age = df.loc[df[target] == 0, 'Age'].mean()
prior_fault_fraud = df.loc[df[target] == 1, 'AtFaultClaim in the past 12 months'].mean()
prior_fault_non_fraud = df.loc[df[target] == 0, 'AtFaultClaim in the past 12 months'].mean()

print('Unique claims:', unique_claims)
print('Fraud claims:', fraud_claims, ', ', round(fraud_claims/unique_claims *100,1),'%')
print('Non-fraud claims:', non_fraud_claims, ', ', round(non_fraud_claims/unique_claims *100,1),'%')
print('Average incurred loss - Fraud:', '$', round(incurred_loss_fraud))
print('Average incurred loss - Non-fraud:', '$',round(incurred_loss_non_fraud))
print('Average reporting delay-Fraud:',round(fraud_rept_delay,1),'days')
print('Average reporting delay-NonFraud:',round(non_fraud_rept_delay,1),'days')
print('Average driver age - Fraud:', round(fraud_dr_age,1))
print('Average driver age - Non-fraud:', round(non_fraud_dr_age,1))
print('Average prior at-fault claims (12m) - Fraud:',round(prior_fault_fraud,2))
print('Average prior at-fault claims (12m) - Non-fraud:', round(prior_fault_non_fraud,2))

# *** Machine Learning ***
grid = {'max_depth': [6,7,8], 'n_estimators': [700,800,900]}
cat = CatBoostClassifier(learning_rate=0.06, loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced", random_seed=42, l2_leaf_reg=5, 
                         verbose=False, allow_writing_files=False)
gs = GridSearchCV(cat, grid, cv=10, n_jobs=-1, scoring ='roc_auc')
gs.fit(X[train], y[train],cat_features=cat_feat, eval_set=(X[valid], y[valid]))
gs.cv_results_
print(gs.cv_results, 'best_params:', gs.best_params_, 'best_estimator:', gs.best_estimator_.get_params())

# Catboost model

model = CatBoostClassifier(
    iterations=800, depth=7, learning_rate=.06, loss_function="Logloss", 
    eval_metric="AUC", auto_class_weights="Balanced", random_seed=42, 
    l2_leaf_reg=5, verbose=False, allow_writing_files=False)

model.fit(X[train], y[train], cat_features=cat_feat, eval_set=(X[valid], y[valid]), early_stopping_rounds=80)

p_valid = model.predict_proba(X[valid])[:, 1]
precision, recall, thresholds = precision_recall_curve(y[valid], p_valid)
f1 = 2*precision*recall/(precision+recall+1e-12)
cutoff = thresholds[np.nanargmax(f1[:-1])]
p_test = model.predict_proba(X[test])[:, 1]
pred = (p_test >= cutoff).astype(int)

print('Cutoff thresholds:',round(cutoff,3))
print("ROC AUC: ", round(roc_auc_score(y[test], p_test),3))
print("PR AUC: ", round(average_precision_score(y[test], p_test),3))
print('\nConfusion matrix:')
cm = confusion_matrix(y[test], pred)
print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))
print ("\nClassification report:\n", classification_report(y[test], pred))
pd.Series(model.get_feature_importance(), index=features).sort_values(ascending=False).to_csv('FI_table.csv')

preds = np.concat([X[test].to_numpy(), y[test].to_numpy()[:, None], pred[:,None], p_test[:,None]], axis=1)
df_preds = pd.DataFrame(preds, columns=list(X[test].columns) + ['Fraud Indicator'] + ['pred'] + ['proba'])
#df_preds = df_preds.sort_values(by='proba', ascending=False)
df_preds.to_csv("df_preds.csv", index=False)

# For Review Capacity Table
print('2025 test set:', X[test].shape[0], 'claims')

slice5 = max(1, round(len(df_preds) * 0.05))
df_slice5 = df_preds.head(slice5)
fraud5 = df_slice5['Fraud Indicator'].sum()

slice10 = max(1, round(len(df_preds) * 0.1))
df_slice10 = df_preds.head(slice10)
fraud10 = df_slice10['Fraud Indicator'].sum()

slice20 = max(1, round(len(df_preds) * 0.2))
df_slice20 = df_preds.head(slice20)
fraud20 = df_slice20['Fraud Indicator'].sum()

print('Slice 5%', slice5, 'claims', ',', fraud5,'fraud claims') 
print('Slice 10%',slice10,'claims', ',', fraud10,'fraud claims')  
print('Slice 20%',slice20,'claims', ',', fraud20,'fraud claims')   

# Logistic regression model
lr = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_feat),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_feat)])),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])

lr.fit(X[train], y[train])
p_test_lr = lr.predict_proba(X[test])[:, 1]

Appendix C - Python Code continue (4/4)

comparison = pd.DataFrame({
    "Model": ["CatBoost", "Logistic regression"],
    "ROC AUC": [roc_auc_score(y[test], p_test), roc_auc_score(y[test], p_test_lr)],
    "PR AUC": [average_precision_score(y[test], p_test),
               average_precision_score(y[test], p_test_lr)]}).round(3)

print(comparison.to_string(index=False))
comparison.to_csv("models_comparison.csv", index=False)


# *** Plots ***
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CatBoost feature importance chart
fi = pd.Series(model.get_feature_importance(), index=features).sort_values().tail(20)
plt.figure(figsize=(8, 6))
fi.plot(kind="barh", color="darkred")
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.title("CatBoost Feature Importance - top 20")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("importance.png", dpi=150)
plt.close()

# ROC curve (2025 test set), both models
plt.figure(figsize=(6, 5))
for name, p, col in [("CatBoost", p_test, "#2E75B6"), ("Logistic regression", p_test_lr, "#C55A11")]:
    fpr, tpr, _ = roc_curve(y[test], p)
    plt.plot(fpr, tpr, color=col, label="%s (ROC AUC = %.3f)" % (name, roc_auc_score(y[test], p)))

plt.plot([0, 1], [0, 1], "--", color="grey", label="Random guess (0.500)")
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curve - 2025 test set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()


# Precision-recall curve (2025 test set), both models
plt.figure(figsize=(6, 5))
for name, p, col in [("CatBoost", p_test, "#2E75B6"),
                     ("Logistic Regression", p_test_lr, "#C55A11")]:
    prec, rec, _ = precision_recall_curve(y[test], p)
    plt.plot(rec, prec, color=col, label="%s (PR AUC = %.3f)" % (name, average_precision_score(y[test], p)))

plt.axhline(y[test].mean(), ls="--", color="grey", label="Fraud rate baseline (%.3f)" % y[test].mean())
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curve - 2025 test set")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=150)
plt.close()

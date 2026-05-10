import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

df = pd.read_csv("features.csv")

features = ['pts_diff','fg_diff','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','pm_diff','rest_diff',
            'win_pct_diff','def_diff','home_court']

X = df[features]
y = df['home_win']

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Logistic Regression
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])

# Random Forest
rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

# XGBoost
xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(n_estimators=200, learning_rate=0.05,
                          max_depth=4, random_state=42,
                          eval_metric='logloss', verbosity=0))
])
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])

print("=" * 40)
print("MODEL COMPARISON")
print("=" * 40)
print("Model               Accuracy    AUC")
print("-" * 40)
print("Logistic Regression  " + str(round(lr_acc * 100, 1)) + "%      " + str(round(lr_auc, 3)))
print("Random Forest        " + str(round(rf_acc * 100, 1)) + "%      " + str(round(rf_auc, 3)))
print("XGBoost              " + str(round(xgb_acc * 100, 1)) + "%      " + str(round(xgb_auc, 3)))
print("=" * 40)

# Save the best model based on AUC
best_auc = max(lr_auc, rf_auc, xgb_auc)
if best_auc == xgb_auc:
    best_model = xgb
    best_name = "XGBoost"
elif best_auc == rf_auc:
    best_model = rf
    best_name = "Random Forest"
else:
    best_model = lr
    best_name = "Logistic Regression"

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model: " + best_name + " (AUC: " + str(round(best_auc, 3)) + ")")
print("Saved to model.pkl")
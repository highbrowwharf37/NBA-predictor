import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import pickle

df = pd.read_csv("features.csv")

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
            'home_court','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','win_pct_diff']

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
                          max_depth=4, eval_metric='logloss', random_state=42))
])
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])

print("--- Logistic Regression ---")
print("Accuracy: " + str(round(lr_acc * 100, 1)) + "%")
print("AUC:      " + str(round(lr_auc, 3)))
print()
print("--- Random Forest ---")
print("Accuracy: " + str(round(rf_acc * 100, 1)) + "%")
print("AUC:      " + str(round(rf_auc, 3)))
print()
print("--- XGBoost ---")
print("Accuracy: " + str(round(xgb_acc * 100, 1)) + "%")
print("AUC:      " + str(round(xgb_auc, 3)))
print()
print(classification_report(y_test, xgb.predict(X_test)))

# Save best model
best = max([(lr, lr_auc), (rf, rf_auc), (xgb, xgb_auc)], key=lambda x: x[1])[0]
with open("model.pkl", "wb") as f:
    pickle.dump(best, f)
print("Best model saved to model.pkl")
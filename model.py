import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pickle

df = pd.read_csv("features.csv")

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
            'fg3_diff','reb_diff','ast_diff','tov_diff','stl_diff',
            'win_pct_diff','home_court']

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

# Ensemble
ensemble = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', VotingClassifier(estimators=[
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=42))
    ], voting='soft'))
])
ensemble.fit(X_train, y_train)
ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test))
ensemble_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1])

print("--- Ensemble (LR + RF + XGB) ---")
print("Accuracy: " + str(round(ensemble_acc * 100, 1)) + "%")
print("AUC:      " + str(round(ensemble_auc, 3)))

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
# Save best model by AUC
models = [('Logistic Regression', lr, lr_auc), 
          ('Random Forest', rf, rf_auc),
          ('XGBoost', xgb, xgb_auc),
          ('Ensemble', ensemble, ensemble_auc)]

best_name, best_model, best_auc_score = max(models, key=lambda x: x[2])
print("Best model: " + best_name + " (AUC: " + str(round(best_auc_score, 3)) + ")")

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Verify it saved correctly
import pickle as pk
with open("model.pkl", "rb") as f:
    check = pk.load(f)
print("Saved model type: " + str(type(check.named_steps['clf']).__name__))
print("Saved features: " + str(list(check.named_steps['scaler'].feature_names_in_)))
print("Best model saved to model.pkl")
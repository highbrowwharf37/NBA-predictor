import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
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

print("--- Logistic Regression ---")
print("Accuracy: " + str(round(lr_acc * 100, 1)) + "%")
print("AUC:      " + str(round(lr_auc, 3)))
print()
print("--- Random Forest ---")
print("Accuracy: " + str(round(rf_acc * 100, 1)) + "%")
print("AUC:      " + str(round(rf_auc, 3)))
print()
print(classification_report(y_test, rf.predict(X_test)))

# Save the better model
best = rf if rf_auc > lr_auc else lr
with open("model.pkl", "wb") as f:
    pickle.dump(best, f)
print("Best model saved to model.pkl")
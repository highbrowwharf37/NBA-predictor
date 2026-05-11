import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv("features.csv")

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
            'fg3_diff','reb_diff','ast_diff','tov_diff','stl_diff',
            'win_pct_diff','home_court']

X = df[features]
y = df['home_win']

split = int(len(df) * 0.8)
X_test = X.iloc[split:]
y_test = y.iloc[split:]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# 1. Confusion matrix
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Away Win','Home Win'],
            yticklabels=['Away Win','Home Win'])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 2. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 3)))
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# 3. Feature importance
coefs = pd.Series(model.feature_importances_ 
                  if hasattr(model.named_steps['clf'], 'feature_importances_') 
                  else model.named_steps['clf'].coef_[0], 
                  index=features)
plt.figure(figsize=(8,6))
sns.barplot(x=coefs.values, y=coefs.index, palette='coolwarm')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

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

print("All charts saved!")
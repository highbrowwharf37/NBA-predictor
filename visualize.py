import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv("features.csv")

features = ['pts_diff','fg_diff','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','pm_diff','rest_diff',
            'win_pct_diff','def_diff','home_court']

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
coefs = pd.Series(model.named_steps['clf'].coef_[0], index=features)
plt.figure(figsize=(8,5))
sns.barplot(x=coefs.values, y=coefs.index, palette='coolwarm')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("All charts saved!")
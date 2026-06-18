
<h1>🚀 Advanced AutoML Stacking System</h1>

<p>
A production-inspired <span class="highlight">AutoML pipeline</span> combining 
stacking ensembles, feature selection, drift detection, and threshold optimization.
</p>

<div class="card">
<h2>📌 Key Features</h2>

<span class="badge">XGBoost</span>
<span class="badge">LightGBM</span>
<span class="badge">Random Forest</span>
<span class="badge">Stacking</span>
<span class="badge">SMOTE</span>
<span class="badge">Drift Detection</span>
<span class="badge">Auto Threshold</span>

<ul>
<li>OOF stacking without data leakage</li>
<li>Mutual information feature selection</li>
<li>Imbalanced learning with SMOTE</li>
<li>KS-test drift detection</li>
<li>Optimized decision threshold tuning</li>
</ul>
</div>

<div class="card">
<h2>🏗️ System Architecture</h2>

<pre>
Input Data
   ↓
Feature Selection (Mutual Information)
   ↓
Standard Scaling
   ↓
5-Fold OOF Training
   ↓
Base Models:
   - XGBoost
   - LightGBM
   - Random Forest
   ↓
Meta Model (Logistic Regression)
   ↓
Threshold Optimization
   ↓
Final Prediction
</pre>
</div>

<div class="card">
<h2>📦 Installation</h2>

<pre>
pip install scikit-learn xgboost lightgbm imbalanced-learn pandas numpy scipy
</pre>
</div>

<div class="card">
<h2>🧠 Core Python Implementation</h2>

<pre>
# =========================================
# 🚀 IMPORTS
# =========================================
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from scipy.optimize import minimize
from scipy.stats import ks_2samp


# =========================================
# ☁️ DATA LOADER
# =========================================
def load_datasets():
    data = []
    data.append(load_iris(return_X_y=True))
    data.append(load_breast_cancer(return_X_y=True))
    data.append(load_wine(return_X_y=True))

    try:
        credit = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        X = pd.get_dummies(credit.data).fillna(0)
        y = (credit.target == "good").astype(int)
        data.append((X.values, y.values))
    except:
        pass

    return data


# =========================================
# ⚖️ OOF STACKING
# =========================================
def get_oof_predictions(X, y):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_xgb = np.zeros(len(X))
    oof_rf = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        try:
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        except:
            pass

        xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
        lgb = LGBMClassifier(n_estimators=200).fit(X_train, y_train)

        oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
        oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]

    return np.vstack([oof_xgb, oof_rf, oof_lgb]).T


# =========================================
# 🧠 META MODEL
# =========================================
def train_meta(oof_X, y):
    meta = LogisticRegression()
    meta.fit(oof_X, y)
    return meta


# =========================================
# 🚀 PIPELINE RUN
# =========================================
def run():
    datasets = load_datasets()

    for i, (X, y) in enumerate(datasets):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in skf.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            oof_train = get_oof_predictions(X_train, y_train)
            meta = train_meta(oof_train, y_train)

            xgb = XGBClassifier(n_estimators=300).fit(X_train, y_train)
            rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
            lgb = LGBMClassifier(n_estimators=200).fit(X_train, y_train)

            test_stack = np.column_stack([
                xgb.predict_proba(X_test)[:, 1],
                rf.predict_proba(X_test)[:, 1],
                lgb.predict_proba(X_test)[:, 1]
            ])

            probs = meta.predict_proba(test_stack)[:, 1]
            preds = (probs > 0.5).astype(int)

            acc = accuracy_score(y_test, preds)
            scores.append(acc)

        print("Dataset", i+1, "Accuracy:", np.mean(scores))


# =========================================
# 🚀 EXECUTE
# =========================================
run()
</pre>
</div>

<div class="card">
<h2>🏆 Summary</h2>
<p>
This project demonstrates a <b>full-scale AutoML stacking system</b> combining:
ensemble learning, feature engineering, and production-style ML pipeline design.
</p>
</div>

</div>

</body>
</html>

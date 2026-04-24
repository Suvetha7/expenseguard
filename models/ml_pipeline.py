import os, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

LOW_CONFIDENCE_THRESHOLD = 0.65
SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ── SMOTE (manual lightweight version to avoid imblearn dependency issues) ────
def _smote_oversample(X, y, ratio=0.5):
    """Simple SMOTE-like oversampling for minority class."""
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_needed = int(len(majority_idx) * ratio) - len(minority_idx)
    if n_needed <= 0:
        return X, y
    chosen = np.random.choice(minority_idx, size=n_needed, replace=True)
    noise  = np.random.normal(0, 0.05, (n_needed, X.shape[1]))
    X_syn  = X[chosen] + noise
    y_syn  = np.ones(n_needed, dtype=int)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])

def _evaluate(model, X_test, y_test, model_name, is_unsupervised=False):
    if is_unsupervised:
        preds = model.predict(X_test)
        preds = (preds == -1).astype(int)
        proba = np.where(preds == 1, 0.75, 0.30)
    else:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, proba), 4),
    }

def run_full_pipeline(df, feature_cols):
    from utils.features import engineer_features
    df_feat = engineer_features(df)
    X = df_feat[feature_cols].values
    y = df_feat["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE on training set
    X_train_bal, y_train_bal = _smote_oversample(X_train, y_train, ratio=0.6)

    # Scale for unsupervised
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8,
                                                 class_weight="balanced", random_state=42),
        "XGBoost":       xgb.XGBClassifier(n_estimators=150, max_depth=6,
                                             scale_pos_weight=3, eval_metric="logloss",
                                             random_state=42, verbosity=0),
        "LightGBM":      lgb.LGBMClassifier(n_estimators=150, max_depth=6,
                                              class_weight="balanced", random_state=42,
                                              verbose=-1),
        "Isolation Forest": IsolationForest(contamination=0.20, random_state=42),
        "One-Class SVM":    OneClassSVM(kernel="rbf", nu=0.20),
    }

    metrics = {}
    best_name, best_auc = None, -1

    for name, model in models.items():
        unsup = name in ("Isolation Forest", "One-Class SVM")
        if unsup:
            model.fit(X_train_sc[y_train_bal == 0])
            m = _evaluate(model, X_test_sc, y_test, name, is_unsupervised=True)
        else:
            model.fit(X_train_bal, y_train_bal)
            m = _evaluate(model, X_test, y_test, name)

        metrics[name] = m
        joblib.dump(model, os.path.join(SAVED_MODELS_DIR, f"{name.replace(' ','_')}.pkl"))

        if not unsup and m["roc_auc"] > best_auc:
            best_auc  = m["roc_auc"]
            best_name = name

    # Save scaler
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))

    return models, metrics, best_name


def predict_transactions(df, feature_cols):
    from utils.features import engineer_features
    df_feat = engineer_features(df)
    X = df_feat[feature_cols].values

    lgbm_path = os.path.join(SAVED_MODELS_DIR, "LightGBM.pkl")
    if not os.path.exists(lgbm_path):
        raise FileNotFoundError("Models not trained yet. Run full pipeline first.")

    model = joblib.load(lgbm_path)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    df_out = df_feat.copy()
    df_out["prediction"]  = preds
    df_out["confidence"]  = proba
    df_out["model_used"]  = "LightGBM"
    df_out["confidence_level"] = pd.cut(
        proba, bins=[0, LOW_CONFIDENCE_THRESHOLD, 0.80, 1.0],
        labels=["Low", "Medium", "High"])
    return df_out


def self_learning_retrain(reviewed_df, feature_cols):
    from utils.features import engineer_features
    if len(reviewed_df) < 3:
        return None, None

    df_feat = engineer_features(reviewed_df)
    X = df_feat[feature_cols].values
    y = df_feat["label"].values if "label" in df_feat.columns else np.ones(len(df_feat))

    models_to_retrain = ["Random Forest", "XGBoost", "LightGBM"]
    new_metrics = {}
    best_name, best_f1 = None, -1

    for name in models_to_retrain:
        path = os.path.join(SAVED_MODELS_DIR, f"{name.replace(' ','_')}.pkl")
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        try:
            model.fit(X, y)
            preds = model.predict(X)
            proba = model.predict_proba(X)[:, 1]
            m = {
                "accuracy":  round(accuracy_score(y, preds), 4),
                "precision": round(precision_score(y, preds, zero_division=0), 4),
                "recall":    round(recall_score(y, preds, zero_division=0), 4),
                "f1":        round(f1_score(y, preds, zero_division=0), 4),
                "roc_auc":   round(roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.5, 4),
            }
            new_metrics[name] = m
            joblib.dump(model, path)
            if m["f1"] > best_f1:
                best_f1, best_name = m["f1"], name
        except Exception:
            continue

    return new_metrics, best_name


def get_shap_values(df, feature_cols):
    """Returns SHAP values for LightGBM model."""
    try:
        import shap
        from utils.features import engineer_features
        df_feat = engineer_features(df)
        X = df_feat[feature_cols].values

        model_path = os.path.join(SAVED_MODELS_DIR, "LightGBM.pkl")
        if not os.path.exists(model_path):
            return None, None

        model = joblib.load(model_path)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values may be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values, X
    except Exception as e:
        print(f"SHAP error: {e}")
        return None, None

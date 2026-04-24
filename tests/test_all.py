"""
pytest test suite for ExpenseGuard AI
Run: pytest tests/test_all.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

# ── Data Generation Tests ─────────────────────────────────────────────────────
class TestDataGeneration:
    def test_returns_dataframe(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=200)
        assert len(df) == 200

    def test_required_columns_present(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=100)
        required = ["transaction_id","date","employee_id","department",
                    "category","vendor","amount","anomaly_type","label"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_labels_are_binary(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=300)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_anomaly_types_valid(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=500)
        valid = {"Normal","Duplicate","Policy Violation","Ghost Vendor","Redundant Spending"}
        assert set(df["anomaly_type"].unique()).issubset(valid)

    def test_amounts_positive(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=200)
        assert (df["amount"] > 0).all()

    def test_anomaly_rate_approx(self):
        from data.generate_data import generate_full_dataset
        df = generate_full_dataset(n=1000, anomaly_rate=0.20)
        rate = df["label"].mean()
        assert 0.10 <= rate <= 0.35, f"Anomaly rate {rate:.2f} out of expected range"


# ── Feature Engineering Tests ─────────────────────────────────────────────────
class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self):
        from data.generate_data import generate_full_dataset
        return generate_full_dataset(n=200)

    def test_all_feature_cols_present(self, sample_df):
        from utils.features import engineer_features, FEATURE_COLS
        df_feat = engineer_features(sample_df)
        for col in FEATURE_COLS:
            assert col in df_feat.columns, f"Feature missing: {col}"

    def test_no_nan_in_features(self, sample_df):
        from utils.features import engineer_features, FEATURE_COLS
        df_feat = engineer_features(sample_df)
        assert not df_feat[FEATURE_COLS].isnull().any().any()

    def test_amount_log_positive(self, sample_df):
        from utils.features import engineer_features
        df_feat = engineer_features(sample_df)
        assert (df_feat["amount_log"] >= 0).all()

    def test_is_weekend_binary(self, sample_df):
        from utils.features import engineer_features
        df_feat = engineer_features(sample_df)
        assert set(df_feat["is_weekend"].unique()).issubset({0, 1})

    def test_feature_count(self, sample_df):
        from utils.features import FEATURE_COLS
        assert len(FEATURE_COLS) >= 15


# ── ML Pipeline Tests ─────────────────────────────────────────────────────────
class TestMLPipeline:
    @pytest.fixture(scope="class")
    def trained_data(self):
        from data.generate_data import generate_full_dataset
        from utils.features import FEATURE_COLS
        from models.ml_pipeline import run_full_pipeline, predict_transactions
        df = generate_full_dataset(n=400)
        models, metrics, best = run_full_pipeline(df, FEATURE_COLS)
        df_pred = predict_transactions(df, FEATURE_COLS)
        return df, models, metrics, best, df_pred

    def test_pipeline_returns_metrics(self, trained_data):
        _, _, metrics, _, _ = trained_data
        assert isinstance(metrics, dict)
        assert len(metrics) >= 3

    def test_best_model_identified(self, trained_data):
        _, _, _, best, _ = trained_data
        assert best in ("Random Forest", "XGBoost", "LightGBM")

    def test_auc_above_threshold(self, trained_data):
        _, _, metrics, best, _ = trained_data
        auc = metrics[best]["roc_auc"]
        assert auc >= 0.70, f"AUC {auc} below minimum threshold 0.70"

    def test_predictions_binary(self, trained_data):
        _, _, _, _, df_pred = trained_data
        assert set(df_pred["prediction"].unique()).issubset({0, 1})

    def test_confidence_between_0_and_1(self, trained_data):
        _, _, _, _, df_pred = trained_data
        assert df_pred["confidence"].between(0, 1).all()

    def test_confidence_level_categories(self, trained_data):
        _, _, _, _, df_pred = trained_data
        valid = {"Low", "Medium", "High"}
        actual = set(df_pred["confidence_level"].astype(str).unique())
        assert actual.issubset(valid | {"nan"})

    def test_models_saved_to_disk(self):
        import os
        saved_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "saved_models")
        for name in ["LightGBM", "XGBoost", "Random_Forest"]:
            path = os.path.join(saved_dir, f"{name}.pkl")
            assert os.path.exists(path), f"Model not saved: {name}"

    def test_metrics_keys_complete(self, trained_data):
        _, _, metrics, best, _ = trained_data
        required_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert required_keys.issubset(set(metrics[best].keys()))


# ── Ledger Tests ──────────────────────────────────────────────────────────────
class TestImmutableLedger:
    @pytest.fixture(autouse=True)
    def clear(self):
        from ledger.immutable_ledger import clear_ledger
        clear_ledger()
        yield
        clear_ledger()

    def test_add_block(self):
        from ledger.immutable_ledger import add_to_ledger, verify_chain
        txn = {"transaction_id":"TXN_TEST_001","amount":1000,
               "department":"Finance","employee_id":"EMP001",
               "vendor":"TestVendor","category":"Travel","date":"2024-01-01"}
        block = add_to_ledger(txn, "Duplicate", 0.95, "LightGBM")
        assert block["index"] == 0
        assert len(block["hash"]) == 64

    def test_chain_valid_after_add(self):
        from ledger.immutable_ledger import add_to_ledger, verify_chain
        for i in range(5):
            txn = {"transaction_id":f"TXN_{i:03d}","amount":i*100+500,
                   "department":"HR","employee_id":f"EMP{i:03d}",
                   "vendor":"V","category":"Travel","date":"2024-01-01"}
            add_to_ledger(txn, "Ghost Vendor", 0.9, "LightGBM")
        result = verify_chain()
        assert result["valid"] is True
        assert result["length"] == 5

    def test_hash_chaining(self):
        from ledger.immutable_ledger import add_to_ledger, get_ledger_df
        for i in range(3):
            txn = {"transaction_id":f"TXN_{i}","amount":1000,
                   "department":"Finance","employee_id":"EMP001",
                   "vendor":"V","category":"Travel","date":"2024-01-01"}
            add_to_ledger(txn, "Duplicate", 0.9, "LightGBM")
        df = get_ledger_df()
        assert df.iloc[1]["prev_hash"] == df.iloc[0]["hash"]
        assert df.iloc[2]["prev_hash"] == df.iloc[1]["hash"]

    def test_empty_ledger_valid(self):
        from ledger.immutable_ledger import verify_chain
        result = verify_chain()
        assert result["valid"] is True
        assert result["length"] == 0

    def test_ledger_returns_dataframe(self):
        from ledger.immutable_ledger import add_to_ledger, get_ledger_df
        txn = {"transaction_id":"TXN_DF","amount":999,
               "department":"Legal","employee_id":"EMP042",
               "vendor":"V","category":"Consulting","date":"2024-01-01"}
        add_to_ledger(txn, "Policy Violation", 0.88, "XGBoost")
        df = get_ledger_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ── SMOTE Tests ───────────────────────────────────────────────────────────────
class TestSMOTE:
    def test_smote_increases_minority(self):
        from models.ml_pipeline import _smote_oversample
        X = np.random.randn(100, 5)
        y = np.array([0]*90 + [1]*10)
        X_bal, y_bal = _smote_oversample(X, y, ratio=0.5)
        assert y_bal.sum() > 10

    def test_smote_preserves_majority(self):
        from models.ml_pipeline import _smote_oversample
        X = np.random.randn(100, 5)
        y = np.array([0]*80 + [1]*20)
        X_bal, y_bal = _smote_oversample(X, y)
        assert (y_bal == 0).sum() == 80


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

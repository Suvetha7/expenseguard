import hashlib, json, os
from datetime import datetime
import pandas as pd

LEDGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "ledger.json")

def _load():
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE) as f:
            return json.load(f)
    return []

def _save(chain):
    os.makedirs(os.path.dirname(LEDGER_FILE), exist_ok=True)
    with open(LEDGER_FILE, "w") as f:
        json.dump(chain, f, indent=2)

def _hash_block(block):
    b = {k: v for k, v in block.items() if k != "hash"}
    return hashlib.sha256(json.dumps(b, sort_keys=True, default=str).encode()).hexdigest()

def add_to_ledger(txn_data: dict, anomaly_type: str, confidence: float, model: str):
    chain = _load()
    prev_hash = chain[-1]["hash"] if chain else "0" * 64
    block = {
        "index":          len(chain),
        "timestamp":      datetime.utcnow().isoformat(),
        "transaction_id": txn_data.get("transaction_id", "UNKNOWN"),
        "amount":         float(txn_data.get("amount", 0)),
        "department":     txn_data.get("department", ""),
        "employee_id":    txn_data.get("employee_id", ""),
        "vendor":         txn_data.get("vendor", ""),
        "category":       txn_data.get("category", ""),
        "date":           str(txn_data.get("date", "")),
        "anomaly_type":   anomaly_type,
        "confidence":     round(float(confidence), 4),
        "model":          model,
        "prev_hash":      prev_hash,
    }
    block["hash"] = _hash_block(block)
    chain.append(block)
    _save(chain)
    return block

def verify_chain():
    chain = _load()
    if not chain:
        return {"valid": True, "length": 0}
    for i, block in enumerate(chain):
        stored = block["hash"]
        expected = _hash_block(block)
        if stored != expected:
            return {"valid": False, "length": len(chain), "broken_at": i}
        if i > 0 and block["prev_hash"] != chain[i-1]["hash"]:
            return {"valid": False, "length": len(chain), "broken_at": i}
    return {"valid": True, "length": len(chain)}

def get_ledger_df():
    chain = _load()
    if not chain:
        return pd.DataFrame()
    return pd.DataFrame(chain)

def clear_ledger():
    _save([])

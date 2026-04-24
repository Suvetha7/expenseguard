import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, string

random.seed(42)
np.random.seed(42)

DEPARTMENTS = ["Finance", "Marketing", "Sales", "Engineering", "HR", "Legal", "Procurement", "Operations"]
CATEGORIES  = ["Travel", "Meals", "Software", "Hardware", "Training", "Office Supplies", "Utilities", "Consulting"]
VENDORS     = [f"Vendor_{c}" for c in string.ascii_uppercase[:20]]
GHOST_VEND  = ["Ghost_LLC", "Shadow_Corp", "Phantom_Inc", "Mirage_Ltd", "Specter_Co"]

def _txn_id(prefix="TXN"):
    return f"{prefix}_{random.randint(10000,99999)}"

def generate_full_dataset(n=1500, anomaly_rate=0.20):
    rows = []
    start = datetime(2023, 1, 1)

    for i in range(n):
        date  = start + timedelta(days=random.randint(0, 364))
        emp   = f"EMP{random.randint(1,80):04d}"
        dept  = random.choice(DEPARTMENTS)
        cat   = random.choice(CATEGORIES)
        vend  = random.choice(VENDORS)
        amt   = round(np.random.lognormal(7, 1.2), 2)
        label = 0
        atype = "Normal"
        txn   = _txn_id("TXN")

        if random.random() < anomaly_rate:
            kind = random.choice(["duplicate","policy","ghost","redundant"])
            label = 1
            if kind == "duplicate":
                atype = "Duplicate"
                txn   = f"TXN_DUP_{_txn_id()}"
                amt   = round(amt * random.uniform(0.98, 1.02), 2)
            elif kind == "policy":
                atype = "Policy Violation"
                txn   = f"TXN_POL_{random.randint(1000,9999):04d}"
                amt   = round(random.uniform(8000, 50000), 2)
            elif kind == "ghost":
                atype = "Ghost Vendor"
                txn   = f"TXN_GHT_{random.randint(10,99):04d}"
                vend  = random.choice(GHOST_VEND)
                amt   = round(random.uniform(5000, 30000), 2)
            else:
                atype = "Redundant Spending"
                txn   = f"TXN_RED_{random.randint(1000,9999):04d}"
                amt   = round(random.uniform(3000, 15000), 2)

        rows.append({
            "transaction_id": txn,
            "date": date.strftime("%Y-%m-%d"),
            "employee_id": emp,
            "department": dept,
            "category": cat,
            "vendor": vend,
            "amount": amt,
            "anomaly_type": atype,
            "label": label,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

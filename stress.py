import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBClassifier

df = pd.read_parquet(r"C:\ml-data\split.parquet")
skip = ("Label", "src_file", "row", "block", "split_block", "split_rand")
feat = [c for c in df.columns if c not in skip]
y = (df["Label"] != "BENIGN").astype(int).to_numpy()
lab = df["Label"].to_numpy()
x = df[feat].to_numpy(dtype="float32")
kw = dict(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
          n_jobs=-1, random_state=42)

print("block size sensitivity")
for bs in [100, 1000, 10000, 50000]:
    rng = np.random.default_rng(42)
    blk = (df["src_file"] + "_" + (df["row"] // bs).astype(str)).to_numpy()
    ub = pd.unique(blk)
    tb = rng.choice(ub, size=int(0.3 * len(ub)), replace=False)
    te = np.isin(blk, tb)
    tr = ~te
    m = XGBClassifier(**kw).fit(x[tr], y[tr])
    p = m.predict_proba(x[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    print("block", bs, "test rows", int(te.sum()), "attacks", int(y[te].sum()),
          "precision", round(precision_score(y[te], pred), 4),
          "recall", round(recall_score(y[te], pred), 4),
          "f1", round(f1_score(y[te], pred), 4),
          "pr_auc", round(average_precision_score(y[te], p), 4))

print("\nleave one attack type out (train block split, attack type removed from training)")
tr0 = (df["split_block"] == "train").to_numpy()
benign_te = ~tr0 & (y == 0)
counts = pd.Series(lab[y == 1]).value_counts()
for t in counts[counts >= 1000].index:
    tr = tr0 & (lab != t)
    m = XGBClassifier(**kw).fit(x[tr], y[tr])
    held = lab == t
    ph = m.predict_proba(x[held])[:, 1]
    pb = m.predict_proba(x[benign_te])[:, 1]
    print(t, "rows", int(held.sum()),
          "recall when unseen", round(float((ph >= 0.5).mean()), 3),
          "false alarm rate", round(float((pb >= 0.5).mean()), 5))

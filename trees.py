import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, confusion_matrix)
from xgboost import XGBClassifier

df = pd.read_parquet(r"C:\ml-data\split.parquet")
skip = ("Label", "src_file", "row", "block", "split_block", "split_rand")
feat = [c for c in df.columns if c not in skip]
y = (df["Label"] != "BENIGN").astype(int).to_numpy()
lab = df["Label"].to_numpy()
x = df[feat].to_numpy(dtype="float32")

for s, tag in [("split_block", "block"), ("split_rand", "rand")]:
    tr = (df[s] == "train").to_numpy()
    te = ~tr
    yt = y[te]
    models = {
        "rf": RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
                                     max_samples=0.3, n_jobs=-1, random_state=42),
        "xgb": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                             tree_method="hist", n_jobs=-1, random_state=42),
    }
    out = pd.DataFrame({"label": lab[te], "y": yt})
    for name, m in models.items():
        t0 = time.time()
        m.fit(x[tr], y[tr])
        p = m.predict_proba(x[te])[:, 1]
        out["p_" + name] = p
        pred = (p >= 0.5).astype(int)
        print("\n==", tag, name, "fit seconds:", round(time.time() - t0))
        print("accuracy:", round(accuracy_score(yt, pred), 4))
        print("precision:", round(precision_score(yt, pred, zero_division=0), 4))
        print("recall:", round(recall_score(yt, pred), 4))
        print("f1:", round(f1_score(yt, pred), 4))
        print("pr_auc:", round(average_precision_score(yt, p), 4))
        print("confusion matrix [[tn fp] [fn tp]]")
        print(confusion_matrix(yt, pred))
        a = yt == 1
        r = pd.DataFrame({"label": lab[te][a], "hit": pred[a]})
        print("recall by attack type")
        print(r.groupby("label")["hit"].agg(["mean", "count"]).round(3).to_string())
        imp = pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False)
        print("top 10 features")
        print(imp.head(10).round(3).to_string())
    out.to_parquet(rf"C:\ml-data\probs_{tag}.parquet")
print("\nsaved probabilities")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, confusion_matrix)

df = pd.read_parquet(r"C:\ml-data\split.parquet")
skip = ("Label", "src_file", "row", "block", "split_block", "split_rand")
feat = [c for c in df.columns if c not in skip]
y = (df["Label"] != "BENIGN").astype(int).to_numpy()
lab = df["Label"].to_numpy()
x = df[feat].to_numpy(dtype="float64")
print("features:", len(feat), "attack share:", round(y.mean(), 3))

for s in ["split_block", "split_rand"]:
    tr = (df[s] == "train").to_numpy()
    te = ~tr
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(max_iter=300).fit(sc.transform(x[tr]), y[tr])
    p = m.predict_proba(sc.transform(x[te]))[:, 1]
    pred = (p >= 0.5).astype(int)
    yt = y[te]
    print("\n==", s)
    print("test rows:", int(te.sum()), "attacks:", int(yt.sum()))
    print("majority-class accuracy:", round(1 - yt.mean(), 4))
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

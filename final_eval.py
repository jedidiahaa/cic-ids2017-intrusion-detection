import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

thr = 0.89
prev = 1e-4

d = pd.read_parquet(r"C:\ml-data\probs_block.parquet")
s = pd.read_parquet(r"C:\ml-data\split.parquet", columns=["block", "split_block"])
blk = s.loc[s["split_block"] == "test", "block"].to_numpy()
assert len(d) == len(blk)
rng = np.random.default_rng(7)
ub = pd.unique(blk)
val = np.isin(blk, rng.choice(ub, size=len(ub) // 2, replace=False))
fin = ~val

y = d["y"].to_numpy()[fin]
lab = d["label"].to_numpy()[fin]
p = d["p_xgb"].to_numpy()[fin]
pred = (p >= thr).astype(int)

cm = confusion_matrix(y, pred)
tn, fp, fn, tp = cm.ravel()
print("final half, xgboost, threshold", thr)
print("rows:", len(y), "attacks:", int(y.sum()))
print("confusion matrix [[tn fp] [fn tp]]")
print(cm)
print("precision:", round(precision_score(y, pred), 4))
print("recall:", round(recall_score(y, pred), 4))
print("f1:", round(f1_score(y, pred), 4))
print("false alarm rate:", round(fp / (fp + tn), 5))
fa = 1e6 * (1 - prev) * fp / (fp + tn)
hit = 1e6 * prev * tp / (tp + fn)
print("at assumed prevalence", prev, "false alerts per million flows:", round(fa),
      "precision:", round(hit / (hit + fa), 3))

a = y == 1
r = pd.DataFrame({"label": lab[a], "hit": pred[a]})
print("\nrecall by attack type at this threshold")
print(r.groupby("label")["hit"].agg(["mean", "count"]).round(3).to_string())

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["benign", "attack"])
ax.set_yticklabels(["benign", "attack"])
ax.set_xlabel("predicted")
ax.set_ylabel("actual")
ax.set_title("xgboost, threshold " + str(thr))
fig.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\nsaved confusion_matrix.png")

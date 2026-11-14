import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

d = pd.read_parquet(r"C:\ml-data\probs_block.parquet")
s = pd.read_parquet(r"C:\ml-data\split.parquet", columns=["block", "split_block"])
blk = s.loc[s["split_block"] == "test", "block"].to_numpy()
assert len(d) == len(blk)
y = d["y"].to_numpy()
p = d["p_xgb"].to_numpy()

rng = np.random.default_rng(7)
ub = pd.unique(blk)
val = np.isin(blk, rng.choice(ub, size=len(ub) // 2, replace=False))
fin = ~val
print("validation rows:", int(val.sum()), "final rows:", int(fin.sum()))

fig, ax = plt.subplots(figsize=(6, 5))
for c in ["p_rf", "p_xgb"]:
    pc = d[c].to_numpy()
    pr, rc, _ = precision_recall_curve(y[fin], pc[fin])
    ap = round(average_precision_score(y[fin], pc[fin]), 4)
    ax.plot(rc, pr, label=c[2:] + " (ap " + str(ap) + ")")
ax.set_xlim(0.9, 1.0)
ax.set_ylim(0.9, 1.0)
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig("pr_curve.png", dpi=150, bbox_inches="tight")
print("saved pr_curve.png")

ts = np.round(np.r_[np.linspace(0.01, 0.99, 99), 0.995, 0.999], 3)
st = {}
for k, m in [("val", val), ("fin", fin)]:
    yy, pp = y[m], p[m]
    st[k] = ([(pp[yy == 1] >= t).mean() for t in ts], [(pp[yy == 0] >= t).mean() for t in ts])
rv, fv = map(np.array, st["val"])
rf, ff = map(np.array, st["fin"])

prev = 1e-4
print("\nassumed attack prevalence:", prev, "(an assumption, not measured)")
print("thresholds chosen on the validation half, reported on the final half")
rows = []
i5 = int(np.where(ts == 0.5)[0][0])
fa = 1e6 * (1 - prev) * ff[i5]
tp = 1e6 * prev * rf[i5]
rows.append(["default", 0.5, round(rf[i5], 4), round(fa), round(tp / (tp + fa), 3)])
for r in [10, 100, 1000]:
    cost = r * 1e6 * prev * (1 - rv) + 1e6 * (1 - prev) * fv
    i = int(np.argmin(cost))
    fa = 1e6 * (1 - prev) * ff[i]
    tp = 1e6 * prev * rf[i]
    rows.append([r, ts[i], round(rf[i], 4), round(fa), round(tp / (tp + fa), 3)])
cols = ["miss_cost_ratio", "threshold", "recall", "false_alerts_per_million", "precision_at_prev"]
print(pd.DataFrame(rows, columns=cols).to_string(index=False))

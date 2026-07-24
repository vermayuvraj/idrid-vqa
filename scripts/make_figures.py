"""
Generate publication figures (vector PDF) for the IDRiD-VQA paper.

Fig 1  dataset grade distribution (train vs test)  -> class imbalance
Fig 2  confusion matrix of the fine-tuned model    -> error structure
Fig 3  input-resolution ablation (512 vs 768 px)   -> headline finding

Palette: Okabe-Ito blue/vermillion (CVD-validated, grayscale-separable).
Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BLUE, VERM, INK, MUTED, GRID = "#0072B2", "#D55E00", "#1a1a1a", "#5c5c5c", "#d9d9d9"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed TrueType (arXiv-safe)
})
GRADES = ["0\nNo DR", "1\nMild", "2\nModerate", "3\nSevere", "4\nProlif."]


def _bars(ax, labels, s1, s2, n1, n2, ylabel, ymax):
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, s1, w, label=n1, color=BLUE, zorder=3)
    b2 = ax.bar(x + w / 2, s2, w, label=n2, color=VERM, zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}".rstrip("0").rstrip("."),
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")


# ---------------- Fig 1: grade distribution ----------------
man = json.loads((ROOT / "data/processed/idrid_manifest.json").read_text())
tr = collections.Counter(r["grade"] for r in man if r["split"] == "train")
te = collections.Counter(r["grade"] for r in man if r["split"] == "test")
fig, ax = plt.subplots(figsize=(3.4, 2.25))
_bars(ax, GRADES, [tr.get(g, 0) for g in range(5)], [te.get(g, 0) for g in range(5)],
      "Train (413)", "Test (103)", "Number of images", max(tr.values()) * 1.22)
ax.set_xlabel("ICDR severity grade")
fig.savefig(FIGDIR / "grade_distribution.pdf")
plt.close(fig)
print("wrote grade_distribution.pdf")

# ---------------- Fig 3: resolution ablation ----------------
r512 = json.loads((ROOT / "outputs/evaluation_results_512px.json").read_text())["finetuned_qlora"]["metrics"]
r768 = json.loads((ROOT / "outputs/evaluation_results.json").read_text())["finetuned_qlora"]["metrics"]
keys = [("accuracy", "Accuracy"), ("qwk", "QWK"), ("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore")]
fig, ax = plt.subplots(figsize=(3.4, 2.25))
_bars(ax, [lbl for _, lbl in keys],
      [r512[k] * 100 for k, _ in keys], [r768[k] * 100 for k, _ in keys],
      "512 px input", "768 px input", r"Score ($\times$100)", 92)
fig.savefig(FIGDIR / "resolution_ablation.pdf")
plt.close(fig)
print("wrote resolution_ablation.pdf")

# ---------------- Fig 2: confusion matrix ----------------
pf = ROOT / "outputs/predictions_full.json"
if not pf.exists():
    print("[skip] predictions_full.json not found - run evaluate.py first")
else:
    preds = json.loads(pf.read_text())["finetuned_qlora"]
    cm = np.zeros((5, 5), dtype=int)
    for p in preds:
        t = int(p["true_grade"])
        q = p["pred_grade"]
        if q is None:
            continue
        cm[t][int(q)] += 1
    fig, ax = plt.subplots(figsize=(3.4, 2.85))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    thr = cm.max() * 0.55
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thr else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(range(5)); ax.set_yticklabels(range(5))
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0, colors=MUTED)
    cb.set_label("Images", fontsize=7.5)
    per = [(cm[i, i] / cm[i].sum() * 100 if cm[i].sum() else 0) for i in range(5)]
    print("per-grade recall (%):", [f"{v:.0f}" for v in per])
    fig.savefig(FIGDIR / "confusion_matrix.pdf")
    plt.close(fig)
    print("wrote confusion_matrix.pdf")

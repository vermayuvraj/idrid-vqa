"""Generate a clean, publication-style 1200x630 thumbnail (plain matplotlib figure,
not a marketing card) comparing base vs. fine-tuned model performance."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"

W, H = 1200, 630
INK, SUBINK, GRID = "#1a1a2e", "#6b7280", "#e5e7eb"
BASE_C, FT_C = "#9aa5b1", "#2f6fed"

BASE = {"Accuracy\n(%)": 31.1, "QWK\n(x100)": 0.0, "ROUGE-L\n(x100)": 8.1, "BERTScore\n(x100)": 0.0}
FT = {"Accuracy\n(%)": 66.0, "QWK\n(x100)": 76.3, "ROUGE-L\n(x100)": 25.3, "BERTScore\n(x100)": 30.9}

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor("white")

# ---- header (plain text, no boxes) ----
fig.text(0.045, 0.93, "IDRiD-VQA: Explainable Diabetic Retinopathy Grading",
          fontsize=19, fontweight="bold", color=INK)
fig.text(0.045, 0.885, "Qwen2-VL-2B + QLoRA fine-tuning on a single 6GB consumer GPU  ·  evaluated on 103 IDRiD test images",
          fontsize=11.5, color=SUBINK)

# ---- main bar chart (publication style) ----
ax = fig.add_axes([0.06, 0.16, 0.62, 0.62])
metrics = list(BASE.keys())
x = range(len(metrics))
w = 0.34
base_vals = [BASE[m] for m in metrics]
ft_vals = [FT[m] for m in metrics]

b1 = ax.bar([i - w / 2 for i in x], base_vals, width=w, color=BASE_C, label="Zero-shot base", zorder=3)
b2 = ax.bar([i + w / 2 for i in x], ft_vals, width=w, color=FT_C, label="Fine-tuned (ours)", zorder=3)
for bars, vals in [(b1, base_vals), (b2, ft_vals)]:
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2,
                f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=10.5)

ax.set_xticks(list(x))
ax.set_xticklabels(metrics, color=INK, fontsize=10.5)
ax.set_ylim(0, 88)
ax.set_ylabel("score", color=SUBINK, fontsize=10)
ax.tick_params(axis="y", colors=SUBINK, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=INK)
ax.set_title("Grading accuracy & explanation quality", fontsize=12.5, color=INK,
             loc="left", pad=10)

# ---- right-side plain stat list (text only, no cards) ----
sx = 0.72
fig.text(sx, 0.72, "Result summary", fontsize=12.5, fontweight="bold", color=INK)
rows = [
    ("Grading accuracy", "31.1%  →  66.0%"),
    ("QWK (agreement)", "0.00  →  0.76"),
    ("Explanation ROUGE-L", "0.08  →  0.25"),
    ("Training time", "~90 min"),
    ("Peak VRAM", "3.5 GB (RTX 4050)"),
]
y = 0.655
for label, val in rows:
    fig.text(sx, y, label, fontsize=10.5, color=SUBINK)
    fig.text(sx, y - 0.032, val, fontsize=13, color=INK, fontweight="bold")
    y -= 0.10

# ---- footer ----
fig.text(0.045, 0.045, "Yuvraj Verma", fontsize=11.5, fontweight="bold", color=INK)
fig.text(0.045, 0.02,
         "github.com/vermayuvraj/idrid-vqa   ·   huggingface.co/vermayuvraj/idrid-qwen2vl-2b-qlora",
         fontsize=10, color=SUBINK)
fig.text(0.955, 0.02, "research prototype — not for clinical use", fontsize=9,
         color=SUBINK, ha="right", style="italic")

out = Path(__file__).resolve().parents[1] / "assets" / "thumbnail.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor="white", dpi=100)
print("WROTE", out)

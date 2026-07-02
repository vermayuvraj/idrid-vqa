"""Generate a detailed 1200x675 LinkedIn/GitHub thumbnail with real comparison bar charts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

W, H = 1200, 675
BG, CARD = "#0b1220", "#141f36"
TEAL, BLUE, WHITE, GREY, GREY_D = "#33d9b2", "#4aa3ff", "#f5f7fa", "#aab8cc", "#5a6b85"

BASE = {"Accuracy": 31.1, "QWK": 0.0, "ROUGE-L": 8.1, "BERTScore": 0.0}
FT = {"Accuracy": 66.0, "QWK": 76.3, "ROUGE-L": 25.3, "BERTScore": 30.9}
# QWK/ROUGE-L/BERTScore are 0-1 scores; shown x100 so all four bars share one axis.

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 14, H, color=TEAL, zorder=5))

# ---- header ----
ax.text(58, H - 52, "MEDICAL AI  ·  VISION-LANGUAGE MODELS  ·  RESEARCH",
        color=TEAL, fontsize=12.5, fontweight="bold")
ax.text(56, H - 96, "Explainable Diabetic Retinopathy Grading", color=WHITE,
        fontsize=26, fontweight="bold")
ax.text(56, H - 127, "on a 6 GB Laptop GPU", color=WHITE, fontsize=26, fontweight="bold")
ax.text(58, H - 158, "Qwen2-VL-2B + QLoRA, fine-tuned with a Gemini-built 2,064-pair VQA dataset",
        color=GREY, fontsize=13)

# ---- left: headline stat cards ----
cx, cy0, cw, ch, cgap = 58, 285, 300, 96, 14
stats = [("GRADE ACCURACY", "31.1% → 66.0%", BLUE),
         ("QWK AGREEMENT", "0.00 → 0.76", TEAL),
         ("TRAINED IN", "90 min · 3.5 GB VRAM", TEAL)]
for i, (label, val, acc) in enumerate(stats):
    y = cy0 - i * (ch + cgap)
    ax.add_patch(FancyBboxPatch((cx, y), cw, ch, boxstyle="round,pad=3,rounding_size=14",
                                linewidth=0, facecolor=CARD))
    ax.text(cx + 20, y + ch - 30, label, color=GREY, fontsize=11, fontweight="bold")
    ax.text(cx + 20, y + 20, val, color=acc, fontsize=19.5, fontweight="bold")

# ---- right: grouped bar chart panel ----
bx, by, bw, bh = 400, 90, 742, 385
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=3,rounding_size=16",
                            linewidth=0, facecolor=CARD))

# panel title (top-left) + custom legend swatches (top-right) drawn on the main ax,
# fully outside the plotting axes so nothing can overlap the bars
ax.text(bx + 24, by + bh - 30, "Base vs. Fine-tuned", color=WHITE,
        fontsize=15, fontweight="bold")
ax.text(bx + 24, by + bh - 52, "103-image IDRiD test split", color=GREY, fontsize=11.5)
leg_y = by + bh - 34
ax.add_patch(Rectangle((bx + bw - 210, leg_y + 3), 14, 14, color=GREY_D))
ax.text(bx + bw - 190, leg_y, "Zero-shot base", color=GREY, fontsize=11.5)
ax.add_patch(Rectangle((bx + bw - 210, leg_y - 21), 14, 14, color=TEAL))
ax.text(bx + bw - 190, leg_y - 24, "Fine-tuned (ours)", color=GREY, fontsize=11.5)

# axes inset well within the card box: room at top for title/legend, bottom for xtick labels
axL, axB = bx + 34, by + 64
axW, axH = bw - 68, bh - 64 - 78
ax_bar = fig.add_axes([axL / W, axB / H, axW / W, axH / H])
ax_bar.set_facecolor("none")

metrics = list(BASE.keys())
xs = range(len(metrics))
w = 0.32
base_vals = [BASE[m] for m in metrics]
ft_vals = [FT[m] for m in metrics]

b1 = ax_bar.bar([i - w / 2 for i in xs], base_vals, width=w, color=GREY_D, zorder=3)
b2 = ax_bar.bar([i + w / 2 for i in xs], ft_vals, width=w, color=TEAL, zorder=3)

for bars, vals in [(b1, base_vals), (b2, ft_vals)]:
    for rect, v in zip(bars, vals):
        ax_bar.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 3,
                    f"{v:.1f}", ha="center", va="bottom", color=WHITE, fontsize=12.5,
                    fontweight="bold")

ax_bar.set_xticks(list(xs))
ax_bar.set_xticklabels(["Accuracy\n(%)", "QWK\n(x100)", "ROUGE-L\n(x100)", "BERTScore\n(x100)"],
                       color=GREY, fontsize=12)
ax_bar.set_ylim(0, 92)
ax_bar.set_yticks([0, 20, 40, 60, 80])
ax_bar.tick_params(axis="y", colors=GREY_D, labelsize=10, length=0)
ax_bar.tick_params(axis="x", length=0, pad=8)
for spine in ax_bar.spines.values():
    spine.set_visible(False)
ax_bar.grid(axis="y", color="#22304a", linewidth=0.8, zorder=0)

# ---- footer ----
ax.text(58, 32, "Yuvraj Verma", color=WHITE, fontsize=15, fontweight="bold")
ax.text(58, 9, "github.com/vermayuvraj/idrid-vqa   |   huggingface.co/vermayuvraj/idrid-qwen2vl-2b-qlora",
        color=GREY, fontsize=12)

out = Path(__file__).resolve().parents[1] / "assets" / "thumbnail.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG, dpi=100)
print("WROTE", out)

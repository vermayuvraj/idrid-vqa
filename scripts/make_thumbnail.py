"""Generate a 1200x630 LinkedIn / social thumbnail for the IDRiD-VQA project."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

W, H = 1200, 630
BG, CARD = "#0b1220", "#152238"
TEAL, BLUE, WHITE, GREY = "#33d9b2", "#4aa3ff", "#f5f7fa", "#9fb0c3"

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

ax.add_patch(Rectangle((0, 0), 14, H, color=TEAL))

ax.text(60, 575, "MEDICAL AI  ·  VISION-LANGUAGE MODELS  ·  RESEARCH",
        color=TEAL, fontsize=13, fontweight="bold")
ax.text(58, 520, "Explainable Diabetic Retinopathy", color=WHITE, fontsize=31, fontweight="bold")
ax.text(58, 476, "Grading on a 6 GB Laptop GPU", color=WHITE, fontsize=31, fontweight="bold")
ax.text(60, 434, "Fine-tuning Qwen2-VL-2B with QLoRA + a Gemini-built VQA dataset",
        color=GREY, fontsize=14.5)

cards = [
    ("QWK · AGREEMENT", "0.62", "from 0.00", TEAL),
    ("GRADE ACCURACY", "54%", "from 31%", BLUE),
    ("TRAINED IN", "65 min", "3.5 GB · RTX 4050", TEAL),
]
x0, y0, w, h, gap = 60, 150, 350, 205, 15
for i, (label, big, small, acc) in enumerate(cards):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=4,rounding_size=16",
                                linewidth=0, facecolor=CARD))
    ax.text(x + w / 2, y0 + h - 42, label, color=GREY, fontsize=13.5, fontweight="bold", ha="center")
    ax.text(x + w / 2, y0 + 78, big, color=acc, fontsize=50, fontweight="bold", ha="center")
    ax.text(x + w / 2, y0 + 34, small, color=GREY, fontsize=14, ha="center")

ax.text(60, 62, "Yuvraj Verma", color=WHITE, fontsize=16, fontweight="bold")
ax.text(60, 34, "github.com/vermayuvraj/idrid-vqa    |    HuggingFace: vermayuvraj/idrid-vqa",
        color=GREY, fontsize=13.5)

out = Path(__file__).resolve().parents[1] / "assets" / "thumbnail.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG, dpi=100)
print("WROTE", out)

# IDRiD-VQA: Explainable Diabetic Retinopathy Grading with a 2B VLM

Fine-tune **Qwen2-VL-2B-Instruct** with **4-bit QLoRA (Unsloth)** on the IDRiD
dataset to grade diabetic retinopathy (0–4) **and** generate structured clinical
explanations — trained end-to-end on a single **6 GB consumer GPU** (RTX 4050).

> ⚠️ **Research purposes only. Not a medical device. Not for clinical use.**

## Highlights
- **IDRiD-VQA dataset** — 2,064 clinically grounded QA pairs (grade, lesions,
  reasoning, recommendation), **100% Gemini-generated** from IDRiD labels.
- **Fits 6 GB VRAM** — 4-bit weights, Unsloth gradient checkpointing, batch size 1
  with gradient accumulation 8, `max_seq_length=1024`, bf16 (never fp32).
  Full 3-epoch fine-tune: **~65 min, peak 3.52 GB VRAM** on an RTX 4050.
- **Explainable** — outputs a structured report, not just a label.
- **Cheap & offline** — the demo runs fully locally with no API calls.

## Results (103-image IDRiD test split)
| Model | Accuracy | QWK | ROUGE-L | BERTScore | s/img |
|---|---|---|---|---|---|
| Qwen2-VL-2B (zero-shot) | 31.1% | 0.00 | 0.082 | −0.029 | 9.73 |
| **+ QLoRA (ours)** | **54.4%** | **0.62** | **0.235** | **0.287** | **4.75** |

Fine-tuning lifts the standard DR metric (quadratic weighted kappa) from *no
agreement* to *substantial agreement*, ~3× better explanations, and 2× faster
inference — all on a single consumer GPU.

## Model & data releases
- 🤗 Model: https://huggingface.co/vermayuvraj/idrid-qwen2vl-2b-qlora
- 🤗 Dataset: https://huggingface.co/datasets/vermayuvraj/idrid-vqa

## Requirements
- Windows 11 + **WSL2 Ubuntu**, NVIDIA GPU (tested: RTX 4050, 6 GB), recent driver.
- Python 3.12, managed with [`uv`](https://github.com/astral-sh/uv).

## Setup
All Python/CUDA work runs **inside WSL2**.

```bash
# 1. install uv (no sudo needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the venv
uv venv ~/idrid-vlm-venv --python 3.12 --seed
source ~/idrid-vlm-venv/bin/activate

# 3. install torch (CUDA 12.1) FIRST, then the rest
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install unsloth transformers peft trl datasets bitsandbytes accelerate \
  google-generativeai openai gradio rouge-score bert-score scikit-learn scipy \
  pandas matplotlib seaborn wandb tqdm \
  --extra-index-url https://download.pytorch.org/whl/cu121

# 4. verify CUDA / GPU / VRAM
python src/verify_env.py
```

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` (and optionally
`OPENAI_API_KEY`, `WANDB_API_KEY`). Load it with `set -a; source .env; set +a`.

> **Tip (training):** WSL is capped at ~50% of host RAM by default (~7.6 GB here).
> Before Phase 4, raise it in `C:\Users\<you>\.wslconfig` (`[wsl2]` `memory=12GB`),
> then `wsl --shutdown`.

## Get the data
Download IDRiD from
[IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
(free account required) and arrange it as:

```
data/raw/images/                 IDRiD_001.jpg … IDRiD_516.jpg
data/raw/labels/                 grading CSV(s): Image name, Retinopathy grade, Risk of macular edema
data/raw/segmentation/{MA,HE,EX,SE}/   lesion masks (PNG) — segmentation subset only
```

The disease-grading set (516 images) has no per-pixel masks; only the 81-image
segmentation subset does. The pipeline reads lesion flags from masks where present
and falls back to grade-based clinical priors otherwise (recorded in the manifest).

## Usage
```bash
# Phase 2 — build the manifest (grade distribution + lesion co-occurrence)
python src/dataset.py manifest

# Phase 3 — generate IDRiD-VQA (Gemini 1.5 Flash; --offline for no-API templates)
python src/dataset.py qa            # or: python src/dataset.py all

# Phase 4 — QLoRA fine-tuning (3 epochs on the 416-image train split)
python src/train.py                 # --limit 32 --epochs 1 for a smoke test

# Phase 5 — evaluation (base vs fine-tuned vs GPT-4o-mini)
python src/evaluate.py

# Phase 6 — offline Gradio demo
python src/demo.py                  # http://localhost:7860
```

## Paper
```bash
python paper/make_table.py          # regenerate the results table from eval JSON
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Requires the ACL style files (`acl.sty`, `acl_natbib.bst`) from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files) placed in `paper/`.

## Project layout
```
idrid-vlm/
├── CLAUDE.md              project context / constraints
├── data/{raw,processed}/  raw IDRiD + generated manifest & QA
├── src/
│   ├── dataset.py         manifest + QA generation (Phases 2–3)
│   ├── train.py           Unsloth QLoRA training (Phase 4)
│   ├── evaluate.py        metrics + comparison (Phase 5)
│   ├── demo.py            Gradio app (Phase 6)
│   └── verify_env.py      CUDA/VRAM sanity check
├── paper/main.tex         ACL-format preprint (Phase 7)
├── outputs/checkpoints/   LoRA adapters + evaluation_results.json
└── requirements.txt
```

## Metrics
Grading accuracy (exact match), **Quadratic Weighted Kappa (QWK)**, **ROUGE-L**
and **BERTScore** for explanations, plus latency and per-image cost vs GPT-4o-mini.

## Acknowledgements & license
Built on IDRiD (Porwal et al., 2018), Qwen2-VL (Alibaba), Unsloth, and Hugging Face
TRL/PEFT. Code released for research use; IDRiD is subject to its own license.

## Citation
```bibtex
@misc{verma2026idridvqa,
  title  = {IDRiD-VQA: Fine-Tuning a 2B Vision-Language Model for Explainable
            Diabetic Retinopathy Grading on Consumer Hardware},
  author = {Verma, Yuvraj},
  year   = {2026},
  note   = {Preprint}
}
```

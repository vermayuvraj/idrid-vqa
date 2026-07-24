"""
Phase 5 — Evaluation.

Runs the 100-image test split through:
    (1) base Qwen2-VL-2B  (zero-shot),
    (2) the fine-tuned QLoRA adapter,
    (3) GPT-4o-mini zero-shot  (only if OPENAI_API_KEY is set),
and reports:
    grading accuracy (exact match 0-4), Quadratic Weighted Kappa (QWK),
    ROUGE-L and BERTScore for the explanation, and an inference-cost comparison.

Results -> outputs/evaluation_results.json  (+ printed comparison table).

Run (inside WSL, venv active):
    python src/evaluate.py                 # base + fine-tuned (+ gpt-4o-mini if key)
    python src/evaluate.py --limit 10      # quick check
    python src/evaluate.py --models finetuned
"""
from __future__ import annotations

import os
# torch.compile/inductor needs a system C compiler (absent in this WSL); disable it.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from unsloth import FastVisionModel  # noqa: E402 (before transformers)

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQA_PATH = PROJECT_ROOT / "data" / "processed" / "idrid_vqa.json"
ADAPTER = PROJECT_ROOT / "outputs" / "checkpoints" / "final"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.json"

BASE_MODEL = os.environ.get("IDRID_MODEL", "unsloth/Qwen2-VL-2B-Instruct")
IMAGE_MAX_SIDE = 768  # must match training resolution
MAX_NEW_TOKENS = 256

GRADE_Q = ("What is the diabetic retinopathy severity grade (0-4) shown in this "
           "fundus image? Answer with the number and its name.")
REASON_Q = "Explain the clinical reasoning that supports this diabetic retinopathy grade."

# GPT-4o-mini pricing (USD per 1M tokens), for the cost comparison.
GPT4O_MINI_IN = 0.15 / 1e6
GPT4O_MINI_OUT = 0.60 / 1e6


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_test_images() -> list[dict]:
    """One entry per test image: gold grade + gold reasoning explanation."""
    data = json.loads(VQA_PATH.read_text())
    by_img: dict[str, dict] = {}
    for r in data:
        if r.get("split") != "test":
            continue
        img = r["image_id"]
        e = by_img.setdefault(img, {"image_id": img, "image": r["image"],
                                    "grade": r["grade"], "gold_reason": ""})
        if r["qa_type"] == "clinical_reasoning":
            e["gold_reason"] = r["conversations"][1]["content"]
    return list(by_img.values())


def load_image(rel_path: str) -> Image.Image:
    img = Image.open(PROJECT_ROOT / rel_path).convert("RGB")
    w, h = img.size
    if max(w, h) > IMAGE_MAX_SIDE:
        s = IMAGE_MAX_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
    return img


_WORD2GRADE = {
    "no dr": 0, "no diabetic": 0, "mild": 1, "moderate": 2,
    "severe": 3, "proliferative": 4,
}


def parse_grade(text: str) -> int | None:
    """Extract a 0-4 grade from free-text; return None if not found."""
    t = text.lower()
    m = re.search(r"grade\s*[:=]?\s*([0-4])", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([0-4])\b", t)
    if m:
        return int(m.group(1))
    for kw, g in _WORD2GRADE.items():
        if kw in t:
            return g
    return None


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class UnslothBackend:
    def __init__(self, model_ref: str, name: str):
        self.name = name
        self.model, self.tok = FastVisionModel.from_pretrained(
            model_ref, load_in_4bit=True)
        FastVisionModel.for_inference(self.model)

    def generate(self, image: Image.Image, question: str) -> tuple[str, float, float]:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        input_text = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tok(image, input_text, add_special_tokens=False,
                          return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      use_cache=True, do_sample=False)
        dt = time.time() - t0
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        return gen.strip(), dt, 0.0  # local marginal cost ~ 0

    def close(self):
        del self.model
        torch.cuda.empty_cache()


class OpenAIBackend:
    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.name = model

    def generate(self, image: Image.Image, question: str) -> tuple[str, float, float]:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            max_tokens=MAX_NEW_TOKENS,
        )
        dt = time.time() - t0
        u = resp.usage
        cost = u.prompt_tokens * GPT4O_MINI_IN + u.completion_tokens * GPT4O_MINI_OUT
        return resp.choices[0].message.content.strip(), dt, cost


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def compute_metrics(preds: list[dict]) -> dict:
    from sklearn.metrics import cohen_kappa_score
    from rouge_score import rouge_scorer

    y_true = [p["true_grade"] for p in preds]
    y_pred = [p["pred_grade"] if p["pred_grade"] is not None else 0 for p in preds]
    n = len(preds)
    acc = sum(int(a == b) for a, b in zip(y_true,
              [p["pred_grade"] for p in preds])) / n
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                labels=[0, 1, 2, 3, 4])
    except Exception:  # noqa: BLE001
        qwk = float("nan")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = sum(scorer.score(p["gold_reason"], p["pred_reason"])["rougeL"].fmeasure
                for p in preds) / n

    bscore = float("nan")
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score([p["pred_reason"] for p in preds],
                              [p["gold_reason"] for p in preds],
                              lang="en", rescale_with_baseline=True, verbose=False)
        bscore = float(f1.mean())
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] BERTScore skipped: {exc}")

    parse_fail = sum(1 for p in preds if p["pred_grade"] is None)
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "qwk": round(float(qwk), 4),
        "rougeL": round(rouge, 4),
        "bertscore_f1": round(bscore, 4) if bscore == bscore else None,
        "grade_parse_failures": parse_fail,
    }


def evaluate_backend(backend, images: list[dict]) -> dict:
    preds, total_time, total_cost = [], 0.0, 0.0
    for i, item in enumerate(images, 1):
        img = load_image(item["image"])
        g_txt, dt1, c1 = backend.generate(img, GRADE_Q)
        r_txt, dt2, c2 = backend.generate(img, REASON_Q)
        total_time += dt1 + dt2
        total_cost += c1 + c2
        preds.append({
            "image_id": item["image_id"],
            "true_grade": item["grade"],
            "pred_grade": parse_grade(g_txt),
            "gold_reason": item["gold_reason"],
            "pred_reason": r_txt,
        })
        if i % 10 == 0 or i == len(images):
            print(f"    [{backend.name}] {i}/{len(images)}")
    metrics = compute_metrics(preds)
    metrics["total_inference_s"] = round(total_time, 2)
    metrics["sec_per_image"] = round(total_time / len(images), 3)
    metrics["total_cost_usd"] = round(total_cost, 6)
    metrics["cost_per_image_usd"] = round(total_cost / len(images), 6)
    return {"metrics": metrics, "predictions": preds}


# --------------------------------------------------------------------------- #
def print_table(results: dict) -> None:
    cols = ["accuracy", "qwk", "rougeL", "bertscore_f1",
            "sec_per_image", "cost_per_image_usd"]
    hdr = f"{'model':<26}" + "".join(f"{c:>16}" for c in cols)
    print("\n" + "=" * len(hdr))
    print("EVALUATION SUMMARY  (test split)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, res in results.items():
        m = res["metrics"]
        row = f"{name:<26}"
        for c in cols:
            v = m.get(c)
            row += f"{'--' if v is None else v:>16}"
        print(row)
    print("=" * len(hdr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="+",
                    default=["base", "finetuned", "gpt-4o-mini"],
                    choices=["base", "finetuned", "gpt-4o-mini"])
    args = ap.parse_args()

    images = load_test_images()
    if args.limit:
        images = images[:args.limit]
    print(f"[eval] {len(images)} test images")

    results: dict[str, dict] = {}

    if "base" in args.models:
        print("\n[eval] === base Qwen2-VL-2B (zero-shot) ===")
        b = UnslothBackend(BASE_MODEL, "base-qwen2vl-2b")
        results["base_qwen2vl_2b"] = evaluate_backend(b, images)
        b.close()

    if "finetuned" in args.models:
        if not ADAPTER.exists():
            print(f"[skip] fine-tuned adapter not found at {ADAPTER} — run train.py")
        else:
            print("\n[eval] === fine-tuned QLoRA ===")
            f = UnslothBackend(str(ADAPTER), "finetuned-qlora")
            results["finetuned_qlora"] = evaluate_backend(f, images)
            f.close()

    if "gpt-4o-mini" in args.models:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[skip] OPENAI_API_KEY not set — skipping GPT-4o-mini baseline")
        else:
            print("\n[eval] === GPT-4o-mini (zero-shot API) ===")
            g = OpenAIBackend()
            results["gpt_4o_mini"] = evaluate_backend(g, images)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # store metrics + a few sample predictions (keep file small)
    out = {name: {"metrics": r["metrics"], "sample_predictions": r["predictions"][:5]}
           for name, r in results.items()}
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    # full per-image predictions (for confusion matrices / error analysis)
    full = {name: r["predictions"] for name, r in results.items()}
    (RESULTS_PATH.parent / "predictions_full.json").write_text(json.dumps(full, indent=2))
    print(f"\n[ok] wrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

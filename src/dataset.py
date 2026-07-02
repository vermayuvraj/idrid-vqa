"""
IDRiD-VQA data pipeline (Phases 2 & 3).

Reads the official IDRiD "Disease Grading" subset in its canonical split
(413 training + 103 testing images; train and test reuse the same filenames,
so the split is tracked by folder, never by index).

Expected normalised layout (produced by scripts/organize_data.py):
    data/raw/images/train/IDRiD_001.jpg … IDRiD_413.jpg
    data/raw/images/test/ IDRiD_001.jpg … IDRiD_103.jpg
    data/raw/labels/train.csv   (Image name, Retinopathy grade, Risk of macular edema)
    data/raw/labels/test.csv
    data/raw/segmentation/{MA,HE,EX,SE}/*.tif   (optional; separate 81-image subset)

Lesion masks belong to a separate 81-image segmentation subset with different
numbering, so they do not map onto the 516 grading images. Lesion-presence flags
for the grading set therefore come from grade-conditioned clinical priors,
recorded in the manifest as lesion_source="grade_prior".

Phase 2 — build data/processed/idrid_manifest.json (+ grade distribution and
          lesion co-occurrence matrix).
Phase 3 — generate_qa_pairs(): 4 clinical QA pairs per image with Gemini 1.5
          Flash (derived from labels, not pixels) -> data/processed/idrid_vqa.json.

Usage (WSL, venv active):
    python src/dataset.py manifest
    python src/dataset.py qa                 # uses GEMINI_API_KEY
    python src/dataset.py qa --offline       # deterministic templates, no API
    python src/dataset.py all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW / "images"          # contains train/ and test/ subfolders
LABELS_DIR = RAW / "labels"          # train.csv, test.csv
SEG_DIR = RAW / "segmentation"       # optional lesion masks (separate subset)
PROCESSED = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED / "idrid_manifest.json"
VQA_PATH = PROCESSED / "idrid_vqa.json"

SPLITS = ("train", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")

LESIONS = ["MA", "HE", "EX", "SE"]
LESION_NAMES = {
    "MA": "microaneurysms",
    "HE": "hemorrhages",
    "EX": "hard exudates",
    "SE": "soft exudates (cotton-wool spots)",
}
GRADE_NAMES = {
    0: "No DR",
    1: "Mild non-proliferative DR",
    2: "Moderate non-proliferative DR",
    3: "Severe non-proliferative DR",
    4: "Proliferative DR",
}

# Clinical priors: standard lesion progression by DR severity. Used for the
# grading set (which ships no per-image masks); flagged lesion_source="grade_prior".
GRADE_LESION_PRIOR = {
    0: {"MA": False, "HE": False, "EX": False, "SE": False},
    1: {"MA": True,  "HE": False, "EX": False, "SE": False},
    2: {"MA": True,  "HE": True,  "EX": True,  "SE": False},
    3: {"MA": True,  "HE": True,  "EX": True,  "SE": True},
    4: {"MA": True,  "HE": True,  "EX": True,  "SE": True},
}


# --------------------------------------------------------------------------- #
# Phase 2: manifest
# --------------------------------------------------------------------------- #
def _norm_image_id(name: str) -> str:
    """'IDRiD_001.jpg' / ' IDRiD_1 ' -> 'IDRiD_001'."""
    stem = Path(str(name).strip()).stem
    if stem.upper().startswith("IDRID_"):
        num = "".join(ch for ch in stem.split("_")[-1] if ch.isdigit())
        if num:
            return f"IDRiD_{int(num):03d}"
    return stem


def _read_split_csv(split: str) -> pd.DataFrame:
    """Read labels/<split>.csv (or any csv matching the split) -> normalised df."""
    candidates = sorted(LABELS_DIR.glob(f"*{split}*.csv")) + \
        ([LABELS_DIR / f"{split}.csv"] if (LABELS_DIR / f"{split}.csv").exists() else [])
    seen, csvs = set(), []
    for c in candidates:
        if c.exists() and c not in seen:
            seen.add(c)
            csvs.append(c)
    if not csvs:
        raise FileNotFoundError(
            f"No {split} label CSV in {LABELS_DIR}. Run scripts/organize_data.py.")
    df = pd.read_csv(csvs[0])
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    img_col = lower.get("image name") or lower.get("image_name") or df.columns[0]
    grade_col = lower.get("retinopathy grade") or lower.get("retinopathy_grade") or df.columns[1]
    edema_col = lower.get("risk of macular edema") or (df.columns[2] if len(df.columns) > 2 else None)
    keep = df[[img_col, grade_col] + ([edema_col] if edema_col else [])].copy()
    keep.columns = ["image_id", "grade"] + (["edema_risk"] if edema_col else [])
    if "edema_risk" not in keep.columns:
        keep["edema_risk"] = np.nan
    keep = keep.dropna(subset=["image_id", "grade"])
    keep["image_id"] = keep["image_id"].map(_norm_image_id)
    keep["grade"] = keep["grade"].astype(float).astype(int)
    keep["split"] = split
    keep["_n"] = keep["image_id"].str.extract(r"(\d+)").astype(int)
    return keep.sort_values("_n").drop(columns="_n").reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    return pd.concat([_read_split_csv(s) for s in SPLITS], ignore_index=True)


def _find_image_path(image_id: str, split: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = IMAGES_DIR / split / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _find_mask_path(image_id: str, lesion: str) -> Optional[Path]:
    d = SEG_DIR / lesion
    if not d.exists():
        return None
    num = image_id.split("_")[-1].lstrip("0") or "0"
    for p in d.iterdir():
        if p.is_file() and "_" in p.stem:
            digits = "".join(ch for ch in p.stem.split("_")[1] if ch.isdigit())
            if digits and str(int(digits)) == num:
                return p
    return None


def _mask_has_lesion(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            return bool((np.asarray(im.convert("L")) > 0).any())
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] unreadable mask {path.name}: {exc}", file=sys.stderr)
        return False


def detect_lesions(image_id: str, grade: int) -> tuple[dict[str, bool], str]:
    """(flags, source). source='mask' if a matching seg mask exists, else 'grade_prior'."""
    found, flags = False, {}
    for les in LESIONS:
        mp = _find_mask_path(image_id, les)
        if mp is not None:
            found = True
            flags[les] = _mask_has_lesion(mp)
        else:
            flags[les] = False
    if found:
        return flags, "mask"
    return dict(GRADE_LESION_PRIOR.get(grade, GRADE_LESION_PRIOR[0])), "grade_prior"


def build_manifest(verbose: bool = True) -> list[dict]:
    labels = load_labels()
    records, missing = [], 0
    for _, row in labels.iterrows():
        image_id, grade, split = row["image_id"], int(row["grade"]), row["split"]
        img_path = _find_image_path(image_id, split)
        if img_path is None:
            missing += 1
            continue
        lesions, source = detect_lesions(image_id, grade)
        edema = row.get("edema_risk", np.nan)
        records.append({
            "image_id": image_id,
            "image": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "grade": grade,
            "grade_name": GRADE_NAMES[grade],
            "edema_risk": None if pd.isna(edema) else int(edema),
            "lesions": lesions,
            "lesion_source": source,
            "split": split,
        })
    if not records:
        raise RuntimeError("No images matched labels. Run scripts/organize_data.py first.")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(records, indent=2))
    if verbose:
        _report_manifest(records, missing)
        print(f"\n[ok] wrote {len(records)} records -> {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return records


def _report_manifest(records: list[dict], missing: int) -> None:
    n = len(records)
    print("=" * 64)
    print(f"IDRiD MANIFEST  ({n} images; {missing} label rows had no image file)")
    print("=" * 64)

    sc = Counter(r["split"] for r in records)
    print(f"Splits: train={sc.get('train', 0)}  test={sc.get('test', 0)}")

    print("\nGrade distribution (overall):")
    gc = Counter(r["grade"] for r in records)
    mx = max(gc.values()) if gc else 1
    for g in range(5):
        c = gc.get(g, 0)
        print(f"  {g} {GRADE_NAMES[g]:<32} {c:4d}  {'#' * int(40 * c / mx)}")

    print("\nGrade distribution per split:")
    for s in SPLITS:
        gs = Counter(r["grade"] for r in records if r["split"] == s)
        print(f"  {s:<6} " + "  ".join(f"g{g}={gs.get(g, 0)}" for g in range(5)))

    src = Counter(r["lesion_source"] for r in records)
    print(f"\nLesion source: masks={src.get('mask', 0)}  grade-prior={src.get('grade_prior', 0)}")

    print("\nLesion prevalence:")
    for les in LESIONS:
        c = sum(1 for r in records if r["lesions"][les])
        print(f"  {les} ({LESION_NAMES[les]:<30}) {c:4d} / {n}")

    print("\nLesion co-occurrence matrix (# images with both):")
    idx = {les: k for k, les in enumerate(LESIONS)}
    mat = np.zeros((4, 4), dtype=int)
    for r in records:
        present = [les for les in LESIONS if r["lesions"][les]]
        for a in present:
            for b in present:
                mat[idx[a]][idx[b]] += 1
    print("       " + "".join(f"{l:>6}" for l in LESIONS))
    for a in LESIONS:
        print(f"  {a:>4} " + "".join(f"{mat[idx[a]][idx[b]]:>6}" for b in LESIONS))


# --------------------------------------------------------------------------- #
# Phase 3: QA generation
# --------------------------------------------------------------------------- #
QA_TYPES = ["grade_identification", "lesion_identification",
            "clinical_reasoning", "patient_recommendation"]

QA_QUESTIONS = {
    "grade_identification":
        "What is the diabetic retinopathy severity grade shown in this fundus image, and what does it mean?",
    "lesion_identification":
        "Which diabetic retinopathy lesions are visible in this fundus image?",
    "clinical_reasoning":
        "Explain the clinical reasoning that supports this diabetic retinopathy grade.",
    "patient_recommendation":
        "Based on this fundus image, what follow-up and management would you recommend?",
}


def _facts_block(rec: dict) -> str:
    present = [LESION_NAMES[l] for l in LESIONS if rec["lesions"][l]]
    edema = rec.get("edema_risk")
    return (
        f"- DR severity grade: {rec['grade']} ({rec['grade_name']})\n"
        f"- Lesions present: {', '.join(present) if present else 'none'}\n"
        f"- Risk of macular edema: {edema if edema is not None else 'unknown'}\n"
        f"- Lesion evidence source: {rec['lesion_source']}"
    )


def _gemini_prompt(rec: dict) -> str:
    return f"""You are a board-certified ophthalmologist writing training data for a
medical vision-language model. Below are the GROUND-TRUTH findings for one
retinal fundus image (derived from expert labels, NOT from you looking at the
image). Write 4 concise, clinically accurate question/answer pairs.

GROUND-TRUTH FINDINGS:
{_facts_block(rec)}

Produce EXACTLY these 4 QA types, in this order:
1. grade_identification  - state the grade (0-4) and its name.
2. lesion_identification - list the lesions consistent with the findings.
3. clinical_reasoning    - explain why the findings yield this grade.
4. patient_recommendation- give follow-up interval / management for this grade.

Rules:
- Answers must be consistent with the ground-truth findings above.
- 2-4 sentences per answer. Professional but readable.
- Do NOT invent findings not implied by the grade/lesions.
Return ONLY a JSON array of 4 objects: {{"type": <one of the 4>, "answer": <text>}}.
No markdown, no commentary."""


def _strip_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    return t.strip().strip("`").strip()


# Gemini free tier is ~15-20 requests/min. Throttle to stay under it.
_MIN_INTERVAL = float(os.environ.get("GEMINI_MIN_INTERVAL", "4.0"))  # seconds between calls
_last_call_ts = [0.0]


def _throttle() -> None:
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call_ts[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[0] = time.monotonic()


def _gemini_call(model, prompt: str, max_retries: int = 3) -> str:
    delay = 8.0
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            return model.generate_content(prompt).text
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            sleep = delay * attempt + random.uniform(0, 2)
            print(f"  [retry {attempt}/{max_retries}] {type(exc).__name__}: "
                  f"{str(exc)[:100]} -> sleep {sleep:.1f}s", file=sys.stderr)
            time.sleep(sleep)
    raise RuntimeError("unreachable")


def _gemini_batch_prompt(batch: list[dict]) -> str:
    cases = "\n".join(
        f"[{i}] grade {r['grade']} ({r['grade_name']}); lesions: "
        f"{', '.join([LESION_NAMES[l] for l in LESIONS if r['lesions'][l]]) or 'none'}; "
        f"macular-edema risk: "
        f"{r.get('edema_risk') if r.get('edema_risk') is not None else 'unknown'}"
        for i, r in enumerate(batch, 1)
    )
    return f"""You are a board-certified ophthalmologist writing training data for a
medical vision-language model. For EACH numbered retinal fundus case below
(findings from expert labels, NOT from viewing an image), write 4 short,
clinically accurate answers:
- grade_identification: state the grade (0-4) and its name.
- lesion_identification: the lesions consistent with the findings.
- clinical_reasoning: why the findings yield this grade.
- patient_recommendation: follow-up interval / management for this grade.
Each answer 2-4 sentences, consistent with the given findings; do not invent
findings not implied by the grade/lesions.

CASES:
{cases}

Return ONLY a JSON array with one object per case, in order:
{{"case": <1-based index>, "grade_identification": "...", "lesion_identification": "...",
  "clinical_reasoning": "...", "patient_recommendation": "..."}}
No markdown, no commentary."""


def _parse_batch(text: str, batch: list[dict]) -> list[tuple[dict[str, str], str]]:
    """Per-case (answers, source); template fallback for any missing/invalid case."""
    parsed: list[Optional[dict]] = [None] * len(batch)
    try:
        arr = json.loads(_strip_json(text))
        for o in arr:
            idx = int(o.get("case", 0)) - 1
            if 0 <= idx < len(batch) and all(o.get(t) for t in QA_TYPES):
                parsed[idx] = {t: str(o[t]).strip() for t in QA_TYPES}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] batch parse failed: {str(exc)[:120]} -> templates", file=sys.stderr)
    return [(parsed[i], "gemini") if parsed[i] else (_template_answers(batch[i]), "template")
            for i in range(len(batch))]


def _template_answers(rec: dict) -> dict[str, str]:
    """Deterministic offline fallback (no API)."""
    g, name = rec["grade"], rec["grade_name"]
    present = [LESION_NAMES[l] for l in LESIONS if rec["lesions"][l]]
    les_txt = ", ".join(present) if present else "no discrete lesions"
    followup = {
        0: "annual screening", 1: "re-examination in 6-12 months",
        2: "review in 3-6 months", 3: "prompt referral to an ophthalmologist (2-4 weeks)",
        4: "urgent ophthalmology referral for consideration of pan-retinal photocoagulation or anti-VEGF therapy",
    }[g]
    return {
        "grade_identification":
            f"This image corresponds to grade {g} diabetic retinopathy ({name}).",
        "lesion_identification":
            f"The findings are consistent with {les_txt}.",
        "clinical_reasoning":
            f"Grade {g} ({name}) is assigned because the retinal findings "
            f"({les_txt}) match the severity criteria for this category.",
        "patient_recommendation":
            f"For grade {g} ({name}), the recommended management is {followup}. "
            "This is decision support only and not a substitute for clinical judgement.",
    }


def _record_to_qa(rec: dict, answers: dict[str, str], source: str) -> list[dict]:
    return [{
        "image": rec["image"],
        "image_id": rec["image_id"],
        "grade": rec["grade"],
        "split": rec["split"],
        "qa_type": t,
        "qa_source": source,
        "conversations": [
            {"role": "user", "content": f"<image>\n{QA_QUESTIONS[t]}"},
            {"role": "assistant", "content": answers[t]},
        ],
    } for t in QA_TYPES]


# Free-tier quota is per-model; cycle through these as each bucket exhausts.
DEFAULT_MODELS = ("gemini-flash-latest,gemini-3.1-flash-lite,gemini-3-flash-preview,"
                  "gemini-2.5-flash-lite,gemini-2.0-flash-lite,gemini-2.0-flash,gemini-2.5-flash")


def generate_qa_pairs(offline: bool = False, limit: Optional[int] = None,
                      resume: bool = True) -> list[dict]:
    if not MANIFEST_PATH.exists():
        print("[info] manifest missing -> building it first")
        build_manifest(verbose=False)
    records = json.loads(MANIFEST_PATH.read_text())
    if limit:
        records = records[:limit]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # key by (split, image_id): train and test reuse the same image_id (IDRiD_001),
    # so image_id alone is NOT unique and collides across splits.
    def _key(r: dict) -> tuple:
        return (r.get("split"), r["image_id"])

    # resume: reuse QA pairs already sourced from Gemini in a prior run (dedup on load)
    existing: dict[tuple, list[dict]] = {}
    if resume and VQA_PATH.exists():
        try:
            seen = set()
            for d in json.loads(VQA_PATH.read_text()):
                dk = (d.get("split"), d["image_id"], d.get("qa_type"))
                if dk in seen:
                    continue
                seen.add(dk)
                existing.setdefault(_key(d), []).append(d)
        except Exception:  # noqa: BLE001
            existing = {}

    def _have_gemini(r: dict) -> bool:
        rs = existing.get(_key(r), [])
        return len(rs) >= len(QA_TYPES) and all(x.get("qa_source") == "gemini" for x in rs)

    todo = [r for r in records if not _have_gemini(r)]

    # model-cycling pool (each free-tier model has a separate quota bucket)
    genai = None
    model_names: list[str] = []
    if not offline:
        if not os.environ.get("GEMINI_API_KEY"):
            print("[warn] GEMINI_API_KEY not set -> offline templates.", file=sys.stderr)
        else:
            import google.generativeai as genai  # noqa: PLC0415
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model_names = [m.strip() for m in
                           os.environ.get("GEMINI_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
    mi = 0
    model = genai.GenerativeModel(model_names[0]) if (genai and model_names) else None
    print(f"[resume] {len(records) - len(todo)} images already Gemini; regenerating "
          f"{len(todo)}. model pool: {model_names or ['(offline templates)']}")

    results_by_id: dict[tuple, list[dict]] = {}

    def _assemble() -> list[dict]:
        out: list[dict] = []
        for r in records:
            k = _key(r)
            if k in results_by_id:
                out.extend(results_by_id[k])
            elif k in existing:
                out.extend(existing[k][:len(QA_TYPES)])  # cap at 4 (dedup safety)
            else:
                out.extend(_record_to_qa(r, _template_answers(r), "template"))
        return out

    batch_size = int(os.environ.get("GEMINI_BATCH", "8"))
    i = 0
    while i < len(todo):
        batch = todo[i:i + batch_size]
        results = None
        while model is not None and results is None:
            try:
                results = _parse_batch(_gemini_call(model, _gemini_batch_prompt(batch)), batch)
            except Exception as exc:  # noqa: BLE001  current model's quota exhausted
                print(f"  [model '{model_names[mi]}' exhausted: {type(exc).__name__}] "
                      f"-> next model", file=sys.stderr)
                mi += 1
                model = genai.GenerativeModel(model_names[mi]) if mi < len(model_names) else None
        if results is None:  # offline or every model exhausted -> templates
            results = [(_template_answers(r), "template") for r in batch]
        for r, (answers, source) in zip(batch, results):
            results_by_id[_key(r)] = _record_to_qa(r, answers, source)
        i += len(batch)
        dataset = _assemble()
        VQA_PATH.write_text(json.dumps(dataset, indent=2))  # incremental, always complete
        srcc = Counter(d["qa_source"] for d in dataset)
        active = model_names[mi] if model is not None else "templates"
        print(f"  [{i}/{len(todo)} regen | model={active}] total {len(dataset)} pairs; "
              f"gemini={srcc.get('gemini', 0)} template={srcc.get('template', 0)}")

    dataset = _assemble()
    VQA_PATH.write_text(json.dumps(dataset, indent=2))
    srcc = Counter(d["qa_source"] for d in dataset)
    print(f"\n[ok] wrote {len(dataset)} QA pairs -> {VQA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[source] gemini={srcc.get('gemini', 0)}  template={srcc.get('template', 0)}")
    _show_examples(dataset, 3)
    return dataset


def _show_examples(dataset: list[dict], k: int) -> None:
    print("\n" + "=" * 64)
    print(f"{k} EXAMPLE QA PAIRS")
    print("=" * 64)
    for ex in dataset[:k]:
        u = ex["conversations"][0]["content"].replace("<image>\n", "[IMG] ")
        print(f"\n[{ex['image_id']} | {ex['split']} | grade {ex['grade']} | {ex['qa_type']}]")
        print(f"  Q: {u}")
        print(f"  A: {ex['conversations'][1]['content']}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="IDRiD-VQA data pipeline")
    ap.add_argument("stage", choices=["manifest", "qa", "all"])
    ap.add_argument("--offline", action="store_true", help="QA without the Gemini API")
    ap.add_argument("--limit", type=int, default=None, help="first N images (debug)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing idrid_vqa.json (do not resume Gemini pairs)")
    args = ap.parse_args()
    if args.stage in ("manifest", "all"):
        build_manifest()
    if args.stage in ("qa", "all"):
        generate_qa_pairs(offline=args.offline, limit=args.limit, resume=not args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

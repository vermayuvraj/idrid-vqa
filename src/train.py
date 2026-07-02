"""
Phase 4 — QLoRA fine-tuning of Qwen2-VL-2B-Instruct with Unsloth.

Hard constraints (RTX 4050, <=5.5 GB VRAM):
    load_in_4bit=True, use_gradient_checkpointing="unsloth",
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    max_seq_length=1024, bf16 (never fp32).

Run (inside WSL, venv active):
    python src/train.py                       # 3 epochs on the 416 train split
    python src/train.py --limit 32 --epochs 1 # quick smoke test
"""
from __future__ import annotations

import os
# torch.compile/inductor needs a system C compiler (absent in this WSL); Unsloth's
# Triton kernels do not. Disable compile so training runs without gcc. Must be set
# before importing unsloth/torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# Unsloth must be imported before transformers/trl for its patches to apply.
from unsloth import FastVisionModel  # noqa: E402  (import order matters)
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402

import argparse
import functools
import json
import os
from pathlib import Path

import torch
from PIL import Image
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQA_PATH = PROJECT_ROOT / "data" / "processed" / "idrid_vqa.json"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

MODEL_NAME = os.environ.get("IDRID_MODEL", "unsloth/Qwen2-VL-2B-Instruct")
MAX_SEQ_LEN = 1280
IMAGE_MAX_SIDE = 768  # higher res so small lesions (microaneurysms) are visible; still fits 6 GB


# --------------------------------------------------------------------------- #
def _vram(tag: str) -> None:
    if not torch.cuda.is_available():
        print(f"[VRAM/{tag}] CUDA not available")
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    gib = 1024 ** 3
    print(f"[VRAM/{tag}] allocated={alloc/gib:.2f}  reserved={reserved/gib:.2f}  "
          f"free={free/gib:.2f}  total={total/gib:.2f} GiB")


@functools.lru_cache(maxsize=None)
def _load_image(rel_path: str) -> Image.Image:
    # cached by path: the 4 QA records per image share one decoded PIL object
    p = PROJECT_ROOT / rel_path
    img = Image.open(p).convert("RGB")
    w, h = img.size
    if max(w, h) > IMAGE_MAX_SIDE:
        scale = IMAGE_MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def _to_messages(rec: dict) -> dict:
    """Convert a saved idrid_vqa record to Unsloth's vision message format."""
    user = rec["conversations"][0]["content"].replace("<image>\n", "").strip()
    assistant = rec["conversations"][1]["content"].strip()
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": _load_image(rec["image"])},
                {"type": "text", "text": user},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": assistant},
            ]},
        ]
    }


def load_train_dataset(limit: int | None = None) -> list[dict]:
    """Return a plain list of message-dicts with PIL images. NOT a HF Dataset:
    Arrow would encode nested PIL images into {"bytes","path"} dicts, which breaks
    Unsloth's vision collator. Unsloth's examples pass a plain list here."""
    if not VQA_PATH.exists():
        raise FileNotFoundError(
            f"{VQA_PATH} not found — run `python src/dataset.py all` first."
        )
    data = json.loads(VQA_PATH.read_text())
    train = [r for r in data if r.get("split") == "train"]
    if limit:
        train = train[:limit]
    print(f"[data] {len(train)} training QA pairs "
          f"(from {len({r['image_id'] for r in train})} images)")
    return [_to_messages(r) for r in train]


def _patch_num_items_in_batch() -> None:
    """Unsloth's gradient-accumulation fix injects `num_items_in_batch` into the
    model forward, but Qwen2-VL's forward (transformers 4.51.3) does not accept it,
    raising TypeError mid-training. Wrap the class forward to drop that one kwarg;
    any Unsloth-optimised forward underneath is preserved (we call it via `orig`)."""
    import functools
    try:
        from transformers.models.qwen2_vl import modeling_qwen2_vl as mq
    except Exception:  # noqa: BLE001
        return
    cls = getattr(mq, "Qwen2VLForConditionalGeneration", None)
    if cls is None or getattr(cls.forward, "_idrid_patched", False):
        return
    orig = cls.forward

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return orig(self, *args, **kwargs)

    forward._idrid_patched = True
    cls.forward = forward
    print("[patch] Qwen2-VL forward now drops num_items_in_batch")


class VramCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        _vram("train-begin")

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        _vram(f"epoch-{int(state.epoch)}-end")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="cap #train pairs (smoke test)")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    # wandb: use project 'idrid-vlm'; run offline if no API key so it never blocks.
    os.environ.setdefault("WANDB_PROJECT", "idrid-vlm")
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
        print("[wandb] no WANDB_API_KEY -> logging offline to ./wandb")

    print(f"[model] loading {MODEL_NAME} (4-bit)")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=MAX_SEQ_LEN,
    )
    _patch_num_items_in_batch()
    _vram("model-loaded")

    # QLoRA: r=16, alpha=16, all attention + MLP (+ vision) layers.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )
    _vram("peft-attached")

    train_ds = load_train_dataset(limit=args.limit)

    FastVisionModel.for_training(model)
    bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=bf16,
            fp16=not bf16,                 # never fp32
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_steps=1,
            output_dir=str(CKPT_DIR),
            save_strategy="epoch",         # adapter checkpoint after each epoch
            save_total_limit=3,
            report_to="wandb",
            run_name="qwen2vl-2b-idrid-qlora",
            # vision-model specifics for TRL:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=2,
            max_seq_length=MAX_SEQ_LEN,
        ),
        callbacks=[VramCallback()],
    )

    # transformers>=4.46 passes `num_items_in_batch` into the model forward when it
    # believes the model accepts loss kwargs; Qwen2-VL's forward (transformers
    # 4.51.3) does not, raising "unexpected keyword argument 'num_items_in_batch'".
    # Disabling this flag only changes grad-accum loss normalisation, not correctness.
    trainer.model_accepts_loss_kwargs = False

    _vram("pre-train")
    print(f"[train] {args.epochs} epoch(s), bf16={bf16}, "
          f"effective batch = 1 x 8 grad-accum")
    stats = trainer.train()
    print(f"[train] done: {stats.metrics}")

    final = CKPT_DIR / "final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[ok] saved LoRA adapter -> {final.relative_to(PROJECT_ROOT)}")
    if torch.cuda.is_available():
        print(f"[VRAM] peak reserved = "
              f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

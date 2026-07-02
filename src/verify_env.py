"""
Phase 1 sanity check: confirm CUDA is visible and the GPU/VRAM meets the
project's hard constraints (RTX 4050, <= 5.5 GB usable training budget).

Run (inside WSL, venv activated):
    python src/verify_env.py
"""
from __future__ import annotations

import importlib
import sys

TRAIN_VRAM_BUDGET_GB = 5.5  # hard ceiling for training/inference


def _mib_to_gib(x: int) -> float:
    return x / (1024 ** 3)


def check_torch_cuda() -> bool:
    import torch

    print("=" * 60)
    print("TORCH / CUDA")
    print("=" * 60)
    print(f"torch version      : {torch.__version__}")
    print(f"compiled CUDA      : {torch.version.cuda}")
    print(f"cuDNN              : {torch.backends.cudnn.version()}")
    ok = torch.cuda.is_available()
    print(f"cuda.is_available  : {ok}")
    if not ok:
        print("\n[FAIL] CUDA is not available to torch inside WSL.")
        print("       Check: nvidia-smi works in WSL, and torch is a cu121 build.")
        return False

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_gib = _mib_to_gib(props.total_memory)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    free_gib = _mib_to_gib(free_b)

    print(f"\nGPU name           : {props.name}")
    print(f"compute capability : sm_{props.major}{props.minor}")
    print(f"total VRAM         : {total_gib:.2f} GiB")
    print(f"free VRAM (now)    : {free_gib:.2f} GiB")
    print(f"bfloat16 supported : {torch.cuda.is_bf16_supported()}")
    print(f"training budget    : <= {TRAIN_VRAM_BUDGET_GB} GiB (project constraint)")

    if not torch.cuda.is_bf16_supported():
        print("\n[WARN] bf16 not reported as supported — Ada GPUs support it; "
              "verify the driver/torch build.")
    if total_gib < TRAIN_VRAM_BUDGET_GB:
        print(f"\n[WARN] total VRAM ({total_gib:.2f} GiB) is below the "
              f"{TRAIN_VRAM_BUDGET_GB} GiB budget — expect OOM without 4-bit + "
              "gradient checkpointing.")
    return True


def check_imports() -> dict[str, str]:
    print("\n" + "=" * 60)
    print("PACKAGE VERSIONS")
    print("=" * 60)
    pkgs = [
        "transformers", "peft", "trl", "datasets", "bitsandbytes",
        "accelerate", "unsloth", "gradio", "google.generativeai",
        "openai", "rouge_score", "bert_score", "sklearn", "scipy",
        "numpy", "pandas", "PIL", "matplotlib", "seaborn", "wandb", "tqdm",
    ]
    found: dict[str, str] = {}
    for name in pkgs:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            found[name] = ver
            print(f"  [ok]  {name:<22} {ver}")
        except Exception as exc:  # noqa: BLE001
            found[name] = f"MISSING ({type(exc).__name__})"
            print(f"  [--]  {name:<22} not importable: {exc}")
    return found


def main() -> int:
    cuda_ok = check_torch_cuda()
    found = check_imports()
    missing = [k for k, v in found.items() if v.startswith("MISSING")]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CUDA ready         : {cuda_ok}")
    print(f"packages missing   : {len(missing)}"
          + (f" -> {', '.join(missing)}" if missing else ""))
    # CUDA is the phase-1 gate; missing optional pkgs are a warning, not a hard fail.
    return 0 if cuda_ok else 1


if __name__ == "__main__":
    sys.exit(main())

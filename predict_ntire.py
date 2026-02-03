# predict_ntire.py
# ============================================================
# Place in the REPO ROOT of LAA-Net.
# Run:
#   python predict_ntire.py \
#       --checkpoint pretrained/LAANET_SBI.pth \
#       --dataset Deep_Dataset \
#       --output submission.txt \
#       --split test
# ============================================================

import os
import sys
import argparse
import importlib.util
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Import our dataset (lives in repo root, NOT inside datasets/) ─────
# This is intentional: datasets/__init__.py triggers the broken
# register/ → misc relative-import chain. We skip it entirely.
from ntire_dataset import NTIREDataset


# ================================================================
# HOW THE MODEL IS LOADED — step by step
# ================================================================
# 1.  The repo's models/ folder contains the actual model source files.
#     We need to find the right .py file and import the class from it
#     WITHOUT triggering models/__init__.py (which may also pull in
#     the registry).
# 2.  We use importlib to load a single .py file by path — no
#     __init__.py involved.
# 3.  We then inspect the checkpoint to figure out which keys exist,
#     and load weights accordingly.
# ================================================================


def _import_module_from_file(module_name, file_path):
    """Load a single .py file as a module without touching __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    # Put it in sys.modules so internal relative imports inside that file
    # can resolve (if any reference other files in models/)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_model_file(repo_root):
    """
    Scan models/ for the file that actually defines the LAA-Net class.
    The file is likely named efn_adv.py, laa_net.py, or similar.
    We look for any file that contains 'class ' + something + '(nn.Module)'.
    """
    models_dir = os.path.join(repo_root, "models")
    candidates = []
    for f in os.listdir(models_dir):
        if f.startswith("__") or not f.endswith(".py"):
            continue
        path = os.path.join(models_dir, f)
        with open(path, "r") as fh:
            src = fh.read()
        # Collect any file that defines a class inheriting nn.Module
        # and contains "forward" (i.e. it's an actual model, not a utility)
        if "class " in src and "def forward" in src:
            candidates.append((f, path, src))
    return candidates


def _discover_model_class(repo_root):
    """
    Auto-discover the model class inside models/.
    Returns (class_object, module).
    Prints what it found so you can verify.
    """
    candidates = _find_model_file(repo_root)
    if not candidates:
        raise RuntimeError(
            "Could not find any nn.Module class in models/. "
            "Please check your repo structure."
        )

    print(f"\n[🔍] Scanning models/ — found {len(candidates)} candidate file(s):")
    for fname, _, _ in candidates:
        print(f"      • {fname}")

    # If there's exactly one, use it. Otherwise prefer files with
    # known naming patterns.
    priority_names = ["efn_adv", "laa_net", "laanet", "model"]
    chosen = None
    for pname in priority_names:
        for fname, fpath, src in candidates:
            if pname in fname.lower():
                chosen = (fname, fpath, src)
                break
        if chosen:
            break
    if chosen is None:
        chosen = candidates[0]  # fallback: first candidate

    fname, fpath, src = chosen
    print(f"[✓] Using model file: models/{fname}")

    # ── Import it via importlib (bypasses __init__.py) ──────────
    # First, make sure models/ itself is importable for any internal
    # imports that the file might do (e.g. "from .efpn import EFPN")
    models_dir = os.path.join(repo_root, "models")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    # Also add repo root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    mod = _import_module_from_file(f"models.{fname[:-3]}", fpath)

    # ── Find the class ───────────────────────────────────────────
    import inspect
    import torch.nn as nn
    classes = [
        (name, obj) for name, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, nn.Module) and obj is not nn.Module
        and obj.__module__ == mod.__name__   # defined in THIS file
    ]
    if not classes:
        raise RuntimeError(f"No nn.Module subclass found in {fname}")

    # Prefer the class whose name is longest / most specific
    # (avoids picking a small helper class over the main model)
    classes.sort(key=lambda x: len(x[0]), reverse=True)
    class_name, model_class = classes[0]
    print(f"[✓] Model class: {class_name}")
    return model_class


def load_model(checkpoint_path, repo_root, device):
    """
    Full model loading pipeline:
      1. Inspect the .pth to understand its structure
      2. Auto-discover the model class in models/
      3. Try to instantiate the model (handling constructor args)
      4. Load weights
    """
    # ── Step 1: inspect checkpoint ────────────────────────────────
    print(f"\n[→] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        print(f"[🔍] Checkpoint keys: {list(checkpoint.keys())}")
        # Find the state_dict inside
        state_dict = None
        for key in ["model", "model_state_dict", "state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                print(f"[✓] State dict found under key: '{key}'")
                break
        if state_dict is None:
            # The checkpoint itself might be the state_dict
            # (all keys would be param-name strings with tensor values)
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
                print("[✓] Checkpoint IS the state_dict directly")
            else:
                raise RuntimeError(
                    f"Cannot find state_dict in checkpoint. "
                    f"Top-level keys: {list(checkpoint.keys())}"
                )
    else:
        state_dict = checkpoint
        print("[✓] Checkpoint is a raw state_dict (OrderedDict)")

    # Strip 'module.' prefix (DataParallel artifacts)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Print a few sample keys so we know what we're dealing with
    sample_keys = list(state_dict.keys())[:6]
    print(f"[🔍] Sample state_dict keys: {sample_keys}")
    print(f"[🔍] Total parameters in checkpoint: {len(state_dict)}")

    # ── Step 2: discover + instantiate the model ─────────────────
    ModelClass = _discover_model_class(repo_root)

    # ── Step 3: try to instantiate ────────────────────────────────
    # The model constructor might require arguments.  We try common
    # patterns used by LAA-Net style repos.
    model = None
    instantiation_attempts = [
        # Most likely: no-arg constructor (all defaults in __init__)
        lambda: ModelClass(),
        # EfficientNet-B4 based, 2 classes
        lambda: ModelClass(backbone="efficientnet-b4", num_classes=2),
        lambda: ModelClass(num_classes=2),
        lambda: ModelClass(backbone_name="efficientnet-b4"),
    ]

    for attempt_fn in instantiation_attempts:
        try:
            model = attempt_fn()
            print(f"[✓] Model instantiated successfully")
            break
        except TypeError as e:
            print(f"[⚠] Instantiation attempt failed: {e}")
            continue

    if model is None:
        raise RuntimeError(
            "Could not instantiate the model class. "
            "Please check the __init__ signature of the model class "
            "and adjust the instantiation_attempts list in this script."
        )

    # ── Step 4: load weights ─────────────────────────────────────
    # Use strict=False first to see what's missing/unexpected
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[⚠] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[⚠] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("[✓] All weights loaded perfectly (strict match)")

    model.to(device)
    model.eval()
    print(f"[✓] Model on {device}, eval mode\n")
    return model


def predict(model, dataloader, device):
    """
    Batch inference with flip-test augmentation.
    LAA-Net outputs a tuple: (cls_logit, heatmap, consistency).
    We only use cls_logit (index 0).
    """
    results = []
    total = len(dataloader.dataset)

    with torch.no_grad():
        for batch_idx, (images, _, filenames) in enumerate(dataloader):
            images = images.to(device)

            # ── Original forward ────────────────────────────────
            outputs = model(images)

            # LAA-Net returns tuple: (classification, heatmap, consistency)
            # During eval some repos return only classification — handle both
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            # ── Interpret logits shape ──────────────────────────
            # Squeeze any extra dims (e.g. [B,1] → [B])
            logits = logits.squeeze(-1) if logits.dim() > 1 and logits.shape[-1] == 1 else logits

            if logits.dim() == 1:
                # Single value per sample → sigmoid
                probs = torch.sigmoid(logits)
            elif logits.shape[-1] == 2:
                # [P(real), P(fake)] → softmax, take fake column
                probs = F.softmax(logits, dim=1)[:, 1]
            else:
                raise ValueError(f"Unexpected logit shape: {logits.shape}")

            # ── Flip-test (horizontal flip augmentation) ────────
            flipped = torch.flip(images, dims=[3])
            flipped_out = model(flipped)
            if isinstance(flipped_out, (tuple, list)):
                flipped_out = flipped_out[0]
            flipped_out = flipped_out.squeeze(-1) if flipped_out.dim() > 1 and flipped_out.shape[-1] == 1 else flipped_out

            if flipped_out.dim() == 1:
                flipped_probs = torch.sigmoid(flipped_out)
            elif flipped_out.shape[-1] == 2:
                flipped_probs = F.softmax(flipped_out, dim=1)[:, 1]
            else:
                flipped_probs = torch.sigmoid(flipped_out.squeeze(-1))

            # Average
            final_probs = (probs + flipped_probs) / 2.0

            for fname, prob in zip(filenames, final_probs.cpu().tolist()):
                results.append((fname, prob))

            done = min((batch_idx + 1) * len(images), total)
            print(f"  [{done}/{total}] images processed")

    return results


def write_submission(results, output_path):
    """
    Write submission.txt: one float per line, alphabetically sorted
    by filename. Exactly what the NTIRE challenge expects.
    """
    results.sort(key=lambda x: x[0])  # alphabetical by filename

    with open(output_path, "w") as f:
        for fname, prob in results:
            f.write(f"{prob:.6f}\n")

    print(f"\n[✓] submission written → {output_path}")
    print(f"    Lines: {len(results)}")
    print(f"    Preview (first 5):")
    for fname, prob in results[:5]:
        print(f"      {fname}  →  {prob:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="NTIRE 2026 Robust Deepfake Detection — LAA-Net inference"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LAANET_SBI.pth")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to Deep_Dataset/ root")
    parser.add_argument("--output", type=str, default="submission.txt")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Lower if you get OOM (EfficientNet-B4 @ 384 is heavy)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    args = parser.parse_args()

    # Repo root = directory where this script lives
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[→] Device: {device}")

    # ── Load model ──────────────────────────────────────────────
    model = load_model(args.checkpoint, REPO_ROOT, device)

    # ── Dataset + DataLoader ────────────────────────────────────
    dataset = NTIREDataset(root_dir=args.dataset, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)  # 0 workers = safer on all OS
    print(f"[→] {len(dataset)} images loaded from {args.split}/")

    # ── Predict ─────────────────────────────────────────────────
    results = predict(model, loader, device)

    # ── Write ───────────────────────────────────────────────────
    write_submission(results, args.output)


if __name__ == "__main__":
    main()
"""
prepare_datasets.py
====================
Splits the raw ASL datasets into train / val / test folders (80 / 10 / 10)
with stratified sampling.  Run once before uploading to Google Drive.

Outputs:
    datasets/asl_commands/{train,val,test}/{del,nothing,space}/
    datasets/asl_digits/{train,val,test}/{0..9}/
    datasets/asl_alphabets/{train,val,test}/{A..Z}/
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
ALPHABET_SUBSAMPLE = 1500        # images per class for alphabets dataset
BASE_DIR = Path(__file__).resolve().parent

# Source paths
ASL_ALPHA_TRAIN = BASE_DIR / "datasets" / "ASL-Alphabet" / "asl_alphabet_train" / "asl_alphabet_train"
DIGITS_SRC      = BASE_DIR / "datasets" / "Sign-Language-Digits-Dataset" / "Dataset"

# Output paths
OUT_COMMANDS  = BASE_DIR / "datasets" / "asl_commands"
OUT_DIGITS    = BASE_DIR / "datasets" / "asl_digits"
OUT_ALPHABETS = BASE_DIR / "datasets" / "asl_alphabets"

# Class definitions
COMMAND_CLASSES  = ["del", "nothing", "space"]
ALPHABET_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGIT_CLASSES    = [str(d) for d in range(10)]


# ─── Helpers ────────────────────────────────────────────────────────────────
def split_and_copy(file_list: list[str], src_dir: Path, out_base: Path, class_name: str):
    """Split a list of filenames 80/10/10 and copy into out_base/{train,val,test}/class_name/."""
    # First split: 80% train, 20% temp
    train_files, temp_files = train_test_split(
        file_list, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    # Second split: 50/50 of the 20% → 10% val, 10% test
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=RANDOM_STATE, shuffle=True
    )

    counts = {}
    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        dest = out_base / split_name / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for fname in split_files:
            shutil.copy2(src_dir / fname, dest / fname)
        counts[split_name] = len(split_files)

    return counts


def process_dataset(name: str, src_base: Path, out_base: Path, classes: list[str],
                    subsample: int | None = None):
    """Process an entire dataset: subsample (optional) → split → copy."""
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    # Clear output directory if it already exists
    if out_base.exists():
        shutil.rmtree(out_base)

    summary_rows = []
    for cls in classes:
        class_dir = src_base / cls
        if not class_dir.exists():
            print(f"  [WARNING] Class directory not found: {class_dir}")
            continue

        files = sorted([f for f in os.listdir(class_dir)
                        if os.path.isfile(class_dir / f)])

        # Subsample if requested
        if subsample and len(files) > subsample:
            random.seed(RANDOM_STATE)
            files = sorted(random.sample(files, subsample))

        counts = split_and_copy(files, class_dir, out_base, cls)
        summary_rows.append((cls, len(files), counts["train"], counts["val"], counts["test"]))

    # Print summary table
    print(f"\n  {'Class':<12} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*5} {'-'*5}")
    total_all = [0, 0, 0, 0]
    for cls, total, tr, va, te in summary_rows:
        print(f"  {cls:<12} {total:>6} {tr:>6} {va:>5} {te:>5}")
        total_all[0] += total
        total_all[1] += tr
        total_all[2] += va
        total_all[3] += te
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*5} {'-'*5}")
    print(f"  {'TOTAL':<12} {total_all[0]:>6} {total_all[1]:>6} {total_all[2]:>5} {total_all[3]:>5}")


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ASL Dataset Preparation Script")
    print("=" * 60)
    print(f"Source (Alphabet): {ASL_ALPHA_TRAIN}")
    print(f"Source (Digits):   {DIGITS_SRC}")
    print(f"Random state:      {RANDOM_STATE}")

    # 1. ASL Commands (del, nothing, space) — from ASL-Alphabet train
    process_dataset(
        name="ASL Commands",
        src_base=ASL_ALPHA_TRAIN,
        out_base=OUT_COMMANDS,
        classes=COMMAND_CLASSES,
    )

    # 2. ASL Digits (0–9) — from Sign-Language-Digits-Dataset
    process_dataset(
        name="ASL Digits",
        src_base=DIGITS_SRC,
        out_base=OUT_DIGITS,
        classes=DIGIT_CLASSES,
    )

    # 3. ASL Alphabets (A–Z, subsampled to 1500/class)
    process_dataset(
        name="ASL Alphabets (subsampled to 1500/class)",
        src_base=ASL_ALPHA_TRAIN,
        out_base=OUT_ALPHABETS,
        classes=ALPHABET_CLASSES,
        subsample=ALPHABET_SUBSAMPLE,
    )

    print("\n✓ All datasets prepared successfully!")
    print(f"  Commands  → {OUT_COMMANDS}")
    print(f"  Digits    → {OUT_DIGITS}")
    print(f"  Alphabets → {OUT_ALPHABETS}")

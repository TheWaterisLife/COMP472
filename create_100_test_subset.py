import os
import shutil
import random
from pathlib import Path

random.seed(42)

def create_even_subset(src_dir, dest_dir, classes, total_images=100):
    dest_dir = Path(dest_dir)
    src_dir = Path(src_dir)
    
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(classes)
    base_per_class = total_images // num_classes
    remainder = total_images % num_classes
    
    # Assign number of images per class
    counts = {}
    for i, cls in enumerate(classes):
        counts[cls] = base_per_class + (1 if i < remainder else 0)
        
    for cls in classes:
        cls_src = src_dir / cls
        cls_dest = dest_dir / cls
        cls_dest.mkdir(parents=True, exist_ok=True)
        
        if not cls_src.exists():
            print(f"Warning: {cls_src} does not exist.")
            continue
            
        files = sorted(os.listdir(cls_src))
        files = [f for f in files if os.path.isfile(cls_src / f)]
        
        needed = counts[cls]
        if len(files) < needed:
            print(f"Warning: Not enough files in {cls_src}. Need {needed}, have {len(files)}")
            selected = files
        else:
            selected = random.sample(files, needed)
            
        for f in selected:
            shutil.copy2(cls_src / f, cls_dest / f)
            
    print(f"Created subset in {dest_dir} with {total_images} total images.")

def main():
    base_dir = Path(r"c:\Users\samyb\Downloads\ai-models\datasets")
    out_dir = base_dir / "test_100_per_dataset"
    
    # Commands
    commands_src = base_dir / "asl_commands" / "test"
    commands_out = out_dir / "asl_commands"
    commands_classes = ["del", "nothing", "space"]
    create_even_subset(commands_src, commands_out, commands_classes, 100)
    
    # Digits
    digits_src = base_dir / "asl_digits" / "test"
    digits_out = out_dir / "asl_digits"
    digits_classes = [str(i) for i in range(10)]
    create_even_subset(digits_src, digits_out, digits_classes, 100)
    
    # Alphabets
    alpha_src = base_dir / "asl_alphabets" / "test"
    alpha_out = out_dir / "asl_alphabets"
    alpha_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    create_even_subset(alpha_src, alpha_out, alpha_classes, 100)

if __name__ == "__main__":
    main()

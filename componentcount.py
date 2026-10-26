
from pathlib import Path
from collections import Counter

# Directory containing this Python script
script_dir = Path(__file__).resolve().parent

# Dataset directory relative to the script
dataset_dir = (
    script_dir
    / "photos"
    / "train_photos"
    / "DATASET_BLUE_21091417"
)

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# Count photos per component
counts = Counter()

for file in dataset_dir.iterdir():
    if file.is_file() and file.suffix.lower() in image_extensions:
        # Component name is everything before the first underscore
        component = file.stem.split("_")[0]
        counts[component] += 1

# Print results
if not counts:
    print("No photos found. Check that the dataset path is correct.")
else:
    print("Photos per component:")
    print("-" * 35)

    for component, count in sorted(counts.items()):
        print(f"{component:<25} {count:>5}")

    print("-" * 35)
    print(f"{'TOTAL':<25} {sum(counts.values()):>5}")

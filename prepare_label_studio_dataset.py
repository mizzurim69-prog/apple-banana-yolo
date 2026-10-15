
from pathlib import Path
import random
import shutil

SOURCE = Path(
    "label_studio_export/project-1-at-2026-10-15-05-48-38287306"
)
OUTPUT = Path("label_studio_yolo_dataset")

IMAGE_DIR = SOURCE / "images"
LABEL_DIR = SOURCE / "labels"
CLASSES_FILE = SOURCE / "classes.txt"

random.seed(42)
extensions = {".jpg", ".jpeg", ".png", ".webp"}

if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
    raise SystemExit("ERROR: Label Studio images or labels folder not found.")

if not CLASSES_FILE.exists():
    raise SystemExit("ERROR: classes.txt not found.")

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

pairs = []
missing_labels = []

for image in images:
    label = LABEL_DIR / f"{image.stem}.txt"

    if label.is_file() and label.stat().st_size > 0:
        pairs.append((image, label))
    else:
        missing_labels.append(image.name)

if missing_labels:
    print("Images with missing or empty labels:")
    for name in missing_labels:
        print(" -", name)
    raise SystemExit("Please check these annotations before splitting.")

if len(pairs) < 2:
    raise SystemExit("ERROR: Not enough image-label pairs.")

random.shuffle(pairs)

split_index = int(len(pairs) * 0.8)
splits = {
    "train": pairs[:split_index],
    "val": pairs[split_index:],
}

for split, split_pairs in splits.items():
    images_out = OUTPUT / split / "images"
    labels_out = OUTPUT / split / "labels"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image, label in split_pairs:
        shutil.copy2(image, images_out / image.name)
        shutil.copy2(label, labels_out / label.name)

shutil.copy2(CLASSES_FILE, OUTPUT / "classes.txt")

print("Dataset preparation completed!")
print("Total image-label pairs:", len(pairs))
print("Training pairs:", len(splits["train"]))
print("Validation pairs:", len(splits["val"]))
print("Output folder:", OUTPUT)

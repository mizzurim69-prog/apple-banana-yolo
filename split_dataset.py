
from pathlib import Path
import random
import shutil

root = Path("dataset_export")

train_images = root / "train" / "images"
train_labels = root / "train" / "labels"

valid_images = root / "valid" / "images"
valid_labels = root / "valid" / "labels"

# Stop if validation files already exist
if valid_images.exists() and any(valid_images.iterdir()):
    raise SystemExit(
        "Validation images already exist. Stop and check before splitting again."
    )

# Find images that have matching YOLO label files
extensions = {".jpg", ".jpeg", ".png", ".webp"}
pairs = []

for image in train_images.iterdir():
    if image.is_file() and image.suffix.lower() in extensions:
        label = train_labels / f"{image.stem}.txt"

        if label.exists():
            pairs.append((image, label))
        else:
            print(f"Skipping image without label: {image.name}")

if len(pairs) < 2:
    raise SystemExit("Not enough image-label pairs to split.")

# Randomly select 20% for validation
random.Random(42).shuffle(pairs)
valid_count = max(1, round(len(pairs) * 0.2))
valid_pairs = pairs[:valid_count]

valid_images.mkdir(parents=True, exist_ok=True)
valid_labels.mkdir(parents=True, exist_ok=True)

# MOVE validation files out of training folders
for image, label in valid_pairs:
    shutil.move(str(image), str(valid_images / image.name))
    shutil.move(str(label), str(valid_labels / label.name))

print("Dataset split completed!")
print(f"Training image-label pairs: {len(pairs) - valid_count}")
print(f"Validation image-label pairs: {valid_count}")

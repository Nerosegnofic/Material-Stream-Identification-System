import os
import shutil
import random

# =========================
# CONFIG
# =========================
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/raw"))
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/trainset"))
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/testset"))

TEST_RATIO = 0.2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================
# CREATE OUTPUT DIRECTORIES
# =========================
for split_dir in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(split_dir, exist_ok=True)

# =========================
# SPLIT PER CLASS
# =========================
for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * TEST_RATIO)
    test_images = images[:split_idx]
    train_images = images[split_idx:]

    # Create class folders
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, class_name, img)
        shutil.copy(src, dst)

    # Copy test images
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TEST_DIR, class_name, img)
        shutil.copy(src, dst)

    print(f"{class_name}: {len(train_images)} train / {len(test_images)} test")

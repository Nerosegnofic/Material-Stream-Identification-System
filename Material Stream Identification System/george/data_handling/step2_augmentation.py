import cv2
import os
import albumentations as A

augmenter = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
    ]
)

# ===============================
# 2. Paths and parameters
# ===============================
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/trainset"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/augmented"))
AUG_PER_IMAGE = 1  # 1 → +100%

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 3. Augmentation loop
# ===============================
for class_name in os.listdir(INPUT_DIR):
    in_class_dir = os.path.join(INPUT_DIR, class_name)
    out_class_dir = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.isdir(in_class_dir):
        continue

    os.makedirs(out_class_dir, exist_ok=True)

    for img_name in os.listdir(in_class_dir):
        img_path = os.path.join(in_class_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # Save original image
        cv2.imwrite(os.path.join(out_class_dir, img_name), image)

        # Generate augmented images
        for i in range(AUG_PER_IMAGE):
            augmented = augmenter(image=image)
            aug_img = augmented["image"]

            # # Ensure size consistency (important!)
            # aug_img = cv2.resize(aug_img, (512, 384))

            aug_name = img_name.replace(".jpg", f"_aug{i}.jpg")
            cv2.imwrite(os.path.join(out_class_dir, aug_name), aug_img)

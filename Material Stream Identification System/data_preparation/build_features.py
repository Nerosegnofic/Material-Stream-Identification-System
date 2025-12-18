import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
SEED = 42
AUGMENT_FACTOR = 1.3   # at least 30% augmentation

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# IMAGE TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# CNN FEATURE EXTRACTOR
# =========================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

# =========================
# DATA LOADING
# =========================
def load_images(dataset_dir):
    classes = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    images, labels = [], []

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_path):
            path = os.path.join(cls_path, fname)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                labels.append(idx)
            except:
                continue

    return images, labels, classes

# =========================
# AUGMENTATION
# =========================
def augment_dataset(images, labels, factor):
    target_len = int(len(images) * factor)
    new_imgs, new_lbls = images[:], labels[:]

    while len(new_imgs) < target_len:
        idx = random.randint(0, len(images) - 1)
        img = train_transform(images[idx])
        img = transforms.ToPILImage()(img)
        new_imgs.append(img)
        new_lbls.append(labels[idx])

    return new_imgs, new_lbls

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(images, labels, model, device):
    model.eval()
    X, y = [], []

    with torch.no_grad():
        for img, lbl in tqdm(zip(images, labels), total=len(images)):
            t = base_transform(img).unsqueeze(0).to(device)
            feat = model(t).cpu().numpy().squeeze()
            X.append(feat)
            y.append(lbl)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =========================
# MAIN PIPELINE
# =========================
def build_feature_dataset(dataset_dir, train_file, test_file, test_size=0.2):
    # Load images
    images, labels, classes = load_images(dataset_dir)
    print(f"Original dataset size: {len(images)}")

    # Train/Test split
    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=SEED
    )

    # Augment training set
    train_imgs, train_lbls = augment_dataset(train_imgs, train_lbls, AUGMENT_FACTOR)
    print(f"Training set after augmentation: {len(train_imgs)}")
    print(f"Test set size (no augmentation): {len(test_imgs)}")

    # Feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFeatureExtractor().to(device)

    X_train, y_train = extract_features(train_imgs, train_lbls, model, device)
    X_test, y_test = extract_features(test_imgs, test_lbls, model, device)

    # Save features
    np.savez_compressed(train_file, X=X_train, y=y_train)
    np.savez_compressed(test_file, X=X_test, y=y_test)

    print("Saved training features:", train_file)
    print("Saved test features:", test_file)
    print("Training feature shape:", X_train.shape)
    print("Test feature shape:", X_test.shape)

    return X_train, y_train, X_test, y_test, classes

# =========================
# RUN
# =========================
if __name__ == "__main__":
    DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    TRAIN_FILE = "train_features.npz"
    TEST_FILE = "test_features.npz"

    build_feature_dataset(DATASET_DIR, TRAIN_FILE, TEST_FILE)

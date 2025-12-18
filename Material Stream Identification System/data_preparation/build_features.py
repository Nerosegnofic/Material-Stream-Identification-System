import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

IMG_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                            scale=(0.95, 1.05), shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
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

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

def load_images(dataset_dir):
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

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

def augment_dataset_balanced(images, labels):
    class_to_imgs = defaultdict(list)

    for img, lbl in zip(images, labels):
        class_to_imgs[lbl].append(img)

    max_count = max(len(v) for v in class_to_imgs.values())

    new_imgs, new_lbls = [], []

    for lbl, imgs in class_to_imgs.items():
        cur_imgs = imgs[:]
        while len(cur_imgs) < max_count:
            idx = random.randint(0, len(imgs) - 1)
            aug_img = train_transform(imgs[idx])
            cur_imgs.append(aug_img)

        new_imgs.extend(cur_imgs)
        new_lbls.extend([lbl] * len(cur_imgs))

    return new_imgs, new_lbls

def extract_features(images, labels, model, device):
    model.eval()
    X, y = [], []

    with torch.no_grad():
        for img, lbl in tqdm(zip(images, labels), total=len(images)):
            if isinstance(img, Image.Image):
                t = base_transform(img).unsqueeze(0).to(device)
            else:
                t = img.unsqueeze(0).to(device)

            feat = model(t).cpu().numpy().squeeze()
            X.append(feat)
            y.append(lbl)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X = normalize(X, norm="l2", axis=1)

    return X, y

def build_feature_dataset(dataset_dir, train_file, test_file, test_size=0.2):
    images, labels, classes = load_images(dataset_dir)

    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        images,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=SEED
    )

    train_imgs, train_lbls = augment_dataset_balanced(
        train_imgs, train_lbls
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNNFeatureExtractor().to(device)

    X_train, y_train = extract_features(
        train_imgs, train_lbls, cnn, device
    )
    X_test, y_test = extract_features(
        test_imgs, test_lbls, cnn, device
    )

    np.savez_compressed(train_file, X=X_train, y=y_train)
    np.savez_compressed(test_file, X=X_test, y=y_test)

    return X_train, y_train, X_test, y_test, classes

if __name__ == "__main__":
    DATASET_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset")
    )
    TRAIN_FILE = "train_features.npz"
    TEST_FILE = "test_features.npz"

    build_feature_dataset(DATASET_DIR, TRAIN_FILE, TEST_FILE)

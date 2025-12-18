import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
DATASET_DIR = os.path.abspath(DATASET_DIR)
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
MODEL_PATH = "cnn_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# SAFE IMAGE CHECK
# =========================
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

# =========================
# AUGMENTATIONS
# =========================
def rotate_image(image):
    angle = random.uniform(-45, 45)
    return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))

def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def random_scale(image):
    scale_factor = random.uniform(0.7, 1.3)
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled = image.resize((new_width, new_height), Image.LANCZOS)
    if scale_factor > 1:
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        scaled = scaled.crop((left, top, left + width, top + height))
    else:
        fill_color = tuple(random.randint(50, 200) for _ in range(3))
        result = Image.new(image.mode, (width, height), fill_color)
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        result.paste(scaled, (paste_x, paste_y))
        scaled = result
    return scaled

def color_jitter(image):
    brightness_factor = random.uniform(0.5, 1.5)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    contrast_factor = random.uniform(0.5, 1.5)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    saturation_factor = random.uniform(0.5, 1.5)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    return image

def gaussian_blur(image):
    radius = random.uniform(0.5, 3.0)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

# Compose random augmentations
AUGMENTATIONS = [rotate_image, horizontal_flip, vertical_flip, random_scale, color_jitter, gaussian_blur]

def apply_random_augmentations(image, num_aug=2):
    for _ in range(num_aug):
        aug = random.choice(AUGMENTATIONS)
        image = aug(image)
    return image

# =========================
# DATASET
# =========================
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, augment=True, augment_factor=0.4):
        self.samples = []
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.augment = augment
        self.augment_factor = augment_factor

        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue
            for f in os.listdir(cls_path):
                path = os.path.join(cls_path, f)
                if is_valid_image(path):
                    self.samples.append((path, self.class_to_idx[cls]))

        # Add augmented samples
        if augment:
            aug_count = int(len(self.samples) * self.augment_factor)
            augmented_samples = random.choices(self.samples, k=aug_count)
            self.samples += [(path, label, True) for path, label in augmented_samples]  # mark as augmented
        print(f"Loaded {len(self.samples)} images ({len(self.classes)} classes, augmented included)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if len(sample) == 2:
            path, label = sample
            do_aug = False
        else:
            path, label, do_aug = sample

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.augment and do_aug:
            img = apply_random_augmentations(img)

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# MODEL
# =========================
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# =========================
# TRAIN LOOP
# =========================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# =========================
# MAIN
# =========================
def main():
    dataset = ImageDataset(DATASET_DIR, transform, augment=True, augment_factor=0.4)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = build_model(len(dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {DEVICE}")

    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optimizer, criterion)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

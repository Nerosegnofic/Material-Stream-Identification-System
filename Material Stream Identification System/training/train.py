import os
import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from models.svm_model import create_svm
from data_preparation.build_features import ImageDataset, transform, build_model  # reuse your dataset & model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def main():
    # =========================
    # LOAD CNN MODEL
    # =========================
    model_path = os.path.join("..", "data_preparation", "cnn_model.pth")
    dataset_dir = os.path.join("..", "dataset")

    dataset = ImageDataset(dataset_dir, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(len(dataset.classes)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Loaded CNN model for feature extraction")

    # =========================
    # EXTRACT FEATURES
    # =========================
    features_list = []
    labels_list = []

    # Remove final classification layer for feature extraction
    feat_model = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            feats = feat_model(imgs)  # [B, 512, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [B, 512]
            features_list.append(feats.cpu())
            labels_list.append(labels)

    X = torch.cat(features_list).numpy()
    y = torch.cat(labels_list).numpy()
    print(f"Extracted features for {len(y)} samples, feature dim: {X.shape[1]}")

    # =========================
    # SHUFFLE + SPLIT
    # =========================
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # TRAIN SVM
    # =========================
    model_svm = create_svm()
    model_svm.fit(X_train, y_train)

    # =========================
    # SAVE MODEL + TEST SET
    # =========================
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model_svm, "saved_models/svm.pkl")
    np.savez("saved_models/test_set.npz", X=X_test, y=y_test)

    print("SVM trained successfully using CNN features.")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples:  {len(y_test)}")


if __name__ == "__main__":
    main()

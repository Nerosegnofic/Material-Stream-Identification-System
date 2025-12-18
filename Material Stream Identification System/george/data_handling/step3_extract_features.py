import cv2
import numpy as np
from skimage.feature import hog
import os
import joblib
from skimage.feature import local_binary_pattern

def extract_features(image_bgr):
    """
    Input:
        image_bgr : numpy array (BGR image from OpenCV)

    Output:
        feature_vector : 1D numpy array (fixed length)
    """

    # =========================
    # 1. Resize
    # =========================
    image_bgr = cv2.resize(image_bgr, (256, 192))

    # =========================
    # 2. Grayscale (for HOG)
    # =========================
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # optional but safe

    # =========================
    # 3. HOG Features (Shape)
    # =========================
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    # =========================
    # 4. Color Histogram (HSV)
    # =========================
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    color_features = color_features / np.sum(color_features)  # normalize

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    lbp_hist = lbp_hist / np.sum(lbp_hist)

    # =========================
    # 5. Combine Features
    # =========================
    feature_vector = np.concatenate([hog_features, color_features, lbp_hist])

    return feature_vector


if __name__ == "__main__":
    # =========================
    # CONFIG
    # =========================
    DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/testset"))  # after it finishs change "augmented" to "testset"
    OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "dataset/features/test_hog_hsv_features.pkl"))  # after it finishs change "train" to "test"

    IMAGE_SIZE = (256, 192)  # (width, height)

    # Class name → label mapping
    CLASS_MAP = {
        "glass": 0,
        "paper": 1,
        "cardboard": 2,
        "plastic": 3,
        "metal": 4,
        "trash": 5,
    }


    # =========================
    # MAIN EXTRACTION LOOP
    # =========================
    X = []
    y = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(DATASET_DIR, class_name)

        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class: {class_name}")

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            features = extract_features(image)
            X.append(features)
            y.append(label)

    # =========================
    # CONVERT TO NUMPY ARRAYS
    # =========================
    X = np.array(X)
    y = np.array(y)

    print("Final feature matrix shape:", X.shape)
    print("Final labels shape:", y.shape)

    # =========================
    # SANITY CHECKS
    # =========================
    assert X.ndim == 2
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

    # =========================
    # SAVE FEATURES
    # =========================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump((X, y), OUTPUT_PATH)

    print(f"Features saved to: {OUTPUT_PATH}")
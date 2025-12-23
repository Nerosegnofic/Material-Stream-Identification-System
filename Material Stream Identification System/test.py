import os
import sys
import torch
import numpy as np
import joblib
import csv
from PIL import Image
from sklearn.preprocessing import normalize
from models.rejection import reject_unknown

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "Material Stream Identification System"))

try:
    from data_preparation.build_features import CNNFeatureExtractor, base_transform
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(
        "Ensure 'Material Stream Identification System' is a valid python package "
        "(has __init__.py if needed, or just importable)."
    )
    sys.exit(1)

MODEL_TO_PROJECT_ID = {
    0: 2,  # cardboard -> 2
    1: 0,  # glass -> 0
    2: 4,  # metal -> 4
    3: 1,  # paper -> 1
    4: 3,  # plastic -> 3
    5: 5,  # trash -> 5
    6: 6,  # unknown -> 6
}

PROJECT_ID_TO_CLASS_NAME = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash",
    6: "unknown"
}

THRESHOLD = 0.5

def predict(dataFilePath, bestModelPath, output_csv="predictions.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = CNNFeatureExtractor().to(device)
    feature_extractor.eval()

    features_list = []
    image_names = []

    try:
        files = sorted(
            f for f in os.listdir(dataFilePath)
            if os.path.isfile(os.path.join(dataFilePath, f))
        )
    except FileNotFoundError:
        print(f"Error: Directory not found: {dataFilePath}")
        return []

    if not files:
        print(f"Warning: No files found in directory: {dataFilePath}")
        return []

    with torch.no_grad():
        for fname in files:
            path = os.path.join(dataFilePath, fname)
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = base_transform(img).unsqueeze(0).to(device)
                feat = feature_extractor(img_tensor).cpu().numpy().squeeze()

                features_list.append(feat)
                image_names.append(fname)

            except Exception:
                continue

    if not features_list:
        print("No valid images processed.")
        return []

    X = np.array(features_list, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    X = normalize(X, norm="l2", axis=1)

    try:
        clf = joblib.load(bestModelPath)
    except Exception as e:
        print(f"Error loading model from {bestModelPath}: {e}")
        return []

    probs = clf.predict_proba(X)

    predictions_internal = [
        reject_unknown(p, threshold=THRESHOLD) for p in probs
    ]

    predictions_mapped = [
        MODEL_TO_PROJECT_ID[p] for p in predictions_internal
    ]

    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "prediction_id", "prediction_class"])

            for img, pid in zip(image_names, predictions_mapped):
                writer.writerow([
                    img,
                    pid,
                    PROJECT_ID_TO_CLASS_NAME.get(pid, "unknown")
                ])

        print(f"Predictions saved to: {output_csv}")

    except Exception as e:
        print(f"Failed to write CSV: {e}")

    return predictions_mapped

if __name__ == "__main__":
    default_data_path = "sample_test_images"
    default_model_path = "training/saved_models/svm.pkl"
    default_output_csv = "predictions.csv"

    if len(sys.argv) >= 3:
        data_path = sys.argv[1]
        model_path = sys.argv[2]
        output_csv = sys.argv[3] if len(sys.argv) == 4 else default_output_csv
    else:
        data_path = default_data_path
        model_path = default_model_path
        output_csv = default_output_csv

    print("Testing predict function...")
    print(f"Data Path: {data_path}")
    print(f"Model Path: {model_path}")
    print(f"Output CSV: {output_csv}")

    try:
        preds = predict(data_path, model_path, output_csv)
        print(f"Predictions: {preds}")
        print(f"Number of predictions: {len(preds)}")
    except Exception as e:
        print(f"An error occurred: {e}")

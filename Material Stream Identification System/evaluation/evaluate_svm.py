import os
import numpy as np
import joblib
from evaluation.evaluate import evaluate

model_path = os.path.join("../training/saved_models", "svm.pkl")
model = joblib.load(model_path)
print("Loaded SVM model for evaluation")

test_path = os.path.join("../training/saved_models", "test_set_svm.npz")
test_data = np.load(test_path)
X_test, y_test = test_data["X"], test_data["y"]
print(f"Loaded test set: {len(y_test)} samples, feature dim: {X_test.shape[1]}")

THRESHOLD = 0.5
acc, report = evaluate(model, X_test, y_test, threshold=THRESHOLD)

print(f"SVM Test Accuracy (threshold={THRESHOLD}): {acc * 100:.2f}%")

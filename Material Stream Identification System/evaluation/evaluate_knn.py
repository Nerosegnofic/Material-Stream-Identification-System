import os
import numpy as np
import joblib
from evaluation.evaluate import evaluate

model_path = os.path.join("../training/saved_models", "knn.pkl")
model = joblib.load(model_path)
print("Loaded KNN model for evaluation")

test_path = os.path.join("../training/saved_models", "test_set.npz")
test_data = np.load(test_path)
X_test, y_test = test_data["X"], test_data["y"]
print(f"Loaded test set: {len(y_test)} samples, feature dim: {X_test.shape[1]}")

THRESHOLD = 0.6
acc, report = evaluate(model, X_test, y_test, threshold=THRESHOLD)

print(f"KNN Test Accuracy (threshold={THRESHOLD}): {acc * 100:.2f}%")
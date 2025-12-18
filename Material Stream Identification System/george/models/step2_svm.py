import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


X_train, y_train = joblib.load("../../dataset/features/train_hog_hsv_features.pkl")

X_val, y_val = joblib.load("../../dataset/features/test_hog_hsv_features.pkl")

scaler = joblib.load("../storage/scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm = SVC(
    kernel="rbf",
    C=2,
    probability=False,  # we’ll use decision_function
    gamma="scale",
    random_state=42,
)

svm.fit(X_train_scaled, y_train)
scores = svm.decision_function(X_val_scaled)
y_val_pred = np.argmax(scores, axis=1)
print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

joblib.dump(svm, "../storage/svm.pkl")
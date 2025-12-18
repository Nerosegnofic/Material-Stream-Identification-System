import cv2
import numpy as np
import joblib
from data_handling.step3_extract_features import extract_features

# =========================
# CONFIG
# =========================
THRESHOLD = 5

CLASS_NAMES = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
}

# =========================
# LOAD MODEL & SCALER
# =========================
svm = joblib.load("storage/svm.pkl")
scaler = joblib.load("storage/scaler.pkl")

# =========================
# CAMERA LOOP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

WINDOW_NAME = "Waste Classification (SVM)"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Extract features
    features = extract_features(frame)
    features_scaled = scaler.transform([features])

    # SVM decision
    scores = svm.decision_function(features_scaled)[0]
    confidence = np.max(scores)
    pred_class = np.argmax(scores)

    if confidence < THRESHOLD:
        label = "Unknown"
        color = (0, 0, 255)  # red
    else:
        label = CLASS_NAMES[pred_class]
        color = (0, 255, 0)  # green

    text = f"{label} | conf: {confidence:.2f}"

    # Draw label
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exit requested via keyboard")
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed by user")
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released, program terminated")
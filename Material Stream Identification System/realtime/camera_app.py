import cv2
import joblib
from features.feature_extractor import extract_features
from models.rejection import reject_unknown
from utils.constants import CLASS_NAMES

model = joblib.load("saved_models/svm.pkl")
THRESHOLD = 0.6
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_features(frame).reshape(1, -1)

    probs = model.predict_proba(feat)[0]
    label_id = reject_unknown(probs, threshold=THRESHOLD)

    label = "Unknown" if label_id == 6 else CLASS_NAMES[label_id]

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Material Classification", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

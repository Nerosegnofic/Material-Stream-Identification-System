import cv2
import joblib
import torch
import numpy as np
from PIL import Image
from collections import deque
from sklearn.preprocessing import normalize

from data_preparation.build_features import CNNFeatureExtractor, base_transform
from models.rejection import reject_unknown
from utils.constants import CLASS_NAMES

MODEL_PATH = "saved_models/svm.pkl"
THRESHOLD = 0.5
SMOOTHING_FRAMES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = joblib.load(MODEL_PATH)

cnn = CNNFeatureExtractor().to(device)
cnn.eval()

history = deque(maxlen=SMOOTHING_FRAMES)

cap = cv2.VideoCapture(0)


debug_printed = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    if not debug_printed:
        print(f"Camera Resolution: {w}x{h}")
        debug_printed = True

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = base_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = cnn(t).cpu().numpy().reshape(1, -1)
        feat = normalize(feat, norm='l2', axis=1)

    probs = classifier.predict_proba(feat)[0]
    
    # Debug: Print top prediction details
    top3_idx = np.argsort(probs)[-3:][::-1]
    print(f"Top: {CLASS_NAMES[top3_idx[0]]} ({probs[top3_idx[0]]:.2f}), "
          f"2nd: {CLASS_NAMES[top3_idx[1]]} ({probs[top3_idx[1]]:.2f})")

    label_id = reject_unknown(probs, threshold=THRESHOLD)

    history.append(label_id)

    final_label = max(set(history), key=history.count)
    label = "Unknown" if final_label == 6 else CLASS_NAMES[final_label]

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Material Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



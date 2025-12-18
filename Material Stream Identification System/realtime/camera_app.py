import cv2
import joblib
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from data_preparation.build_features import CNNFeatureExtractor, base_transform
from models.rejection import reject_unknown
from utils.constants import CLASS_NAMES

MODEL_PATH = "../training/saved_models/svm.pkl"
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = joblib.load(MODEL_PATH)
cnn_model = CNNFeatureExtractor().to(device)
cnn_model.eval()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = base_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = cnn_model(t).cpu().numpy().reshape(1, -1)
        feat = normalize(feat, norm='l2')

    probs = model.predict_proba(feat)[0]
    label_id = reject_unknown(probs, threshold=THRESHOLD)
    label = "Unknown" if label_id == 6 else CLASS_NAMES[label_id]

    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Material Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

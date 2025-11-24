import io
import base64
import torch
import torchvision.transforms as transforms
from torch import nn
import timm
from flask import Flask, render_template, jsonify, request
import atexit
import cv2
import numpy as np
import os
import threading
from waitress import serve  # أخف من Flask built-in server

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "best_vit_lstm.pt"
DEVICE = torch.device("cpu")  # استخدام CPU فقط
SEQ_LEN = 4  # تقليل طول التسلسل
IMG_SIZE = 160  # تقليل حجم الصورة

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Model
# -----------------------------
class ViT_LSTM_Classifier(nn.Module):
    def __init__(self, vit_name="vit_tiny_patch16_224", lstm_hidden=256, lstm_layers=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        self.feat_dim = self.vit.num_features if hasattr(self.vit, "num_features") else 192
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# Load model
model = ViT_LSTM_Classifier().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# -----------------------------
# Prediction buffer
# -----------------------------
_frames_buffer = []
buffer_lock = threading.Lock()
current_label = "Non-Violent"

def process_frame(frame_bgr):
    global _frames_buffer, current_label

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor_frame = transform(img_rgb)

    with buffer_lock:
        _frames_buffer.append(tensor_frame)
        if len(_frames_buffer) > SEQ_LEN:
            _frames_buffer.pop(0)

        if len(_frames_buffer) == SEQ_LEN:
            clip = torch.stack(_frames_buffer).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(clip)
                pred = torch.argmax(out, dim=1).item()
        else:
            pred = 0

    current_label = "Violent" if pred == 1 else "Non-Violent"

# -----------------------------
# Flask endpoints
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    data = request.json.get('frame')
    if data is None:
        return jsonify({"status": "no frame"}), 400
    header, encoded = data.split(",", 1)
    frame_bytes = base64.b64decode(encoded)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    process_frame(frame)
    return jsonify({"status": "ok"})

@app.route('/status')
def status():
    return jsonify({"label": current_label})

# -----------------------------
# Cleanup
# -----------------------------
def _cleanup():
    pass
atexit.register(_cleanup)

# -----------------------------
# Run via Waitress (أخف)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)

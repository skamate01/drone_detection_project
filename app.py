from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
CLIP_FOLDER = "clip_embeddings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -----------------------------
# Load YOLO model
# -----------------------------
yolo_model = YOLO("yolo/best.pt")

# -----------------------------
# Load CLIP model + embeddings
# -----------------------------
import clip
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

clip_embeddings = np.load(f"{CLIP_FOLDER}/Clip_image_embeddings_embeddings.npy")
clip_labels = np.load(f"{CLIP_FOLDER}/Clip_image_embeddings_labels.npy")

clip_embeddings_tensor = torch.tensor(clip_embeddings, dtype=torch.float32)


def classify_with_clip(image):
    """Returns: 'drone' or 'bird'"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(Image.fromarray(image)).unsqueeze(0)

    with torch.no_grad():
        image_features = clip_model.encode_image(image).float()

    similarities = (image_features @ clip_embeddings_tensor.T).squeeze(0)
    best_match = torch.argmax(similarities).item()
    return clip_labels[best_match]


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No file uploaded"

    file = request.files["video"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return redirect(url_for("process_video", filename=file.filename))


@app.route("/process/<filename>")
def process_video(filename):
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(RESULT_FOLDER, f"{filename}_processed.mp4")

    cap = cv2.VideoCapture(input_path)
    fps = 1  # process 1 frame per second
    classified_label = None

    frame_id = 0
    success, frame = cap.read()

    while success:
        if frame_id % int(cap.get(cv2.CAP_PROP_FPS)) == 0:

            results = yolo_model.predict(frame, conf=0.5)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = frame[y1:y2, x1:x2]

                    label = classify_with_clip(cropped)
                    classified_label = label

                    if label == "drone":
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if classified_label:
                break

        success, frame = cap.read()
        frame_id += 1

    cap.release()

    return render_template("result.html", label=classified_label)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

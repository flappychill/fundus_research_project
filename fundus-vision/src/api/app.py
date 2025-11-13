import os, json
from flask import Flask, request, render_template, jsonify
from .infer import Predictor

app = Flask(__name__, static_folder="../web/static", template_folder="../web/templates")
predictor = None

@app.before_first_request
def load_predictor():
    global predictor
    ckpt = os.environ.get("MODEL_PATH", "models/checkpoints/best.pt")
    classes = os.environ.get("CLASSES_PATH", "models/checkpoints/classes.json")
    backbone = os.environ.get("BACKBONE", "convnext_base")
    head = os.environ.get("HEAD", "rpattn")
    predictor = Predictor(ckpt, classes, backbone, head, img_size=int(os.environ.get("IMG_SIZE","224")))

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error":"missing file"}), 400
    f = request.files["file"]
    b = f.read()
    out = predictor.predict_bytes(b)
    return jsonify(out)

@app.get("/ping")
def ping():
    return jsonify({"status":"ok"})

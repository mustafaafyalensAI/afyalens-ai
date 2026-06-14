from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1uiYBXrqtbm1X5F_bJEpOMrRs44_h0yYd"
MODEL_PATH = "/tmp/malaria_model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)

download_model()

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return "AfyaLens AI Running"

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "Uninfected"
        confidence = float(prediction * 100)
    else:
        result = "Parasitized"
        confidence = float((1 - prediction) * 100)

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import os
import requests
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

FILE_ID = "1uiYBXrqtbm1X5F_bJEpOMrRs44_h0yYd"
MODEL_PATH = "/tmp/malaria_model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")

        url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"

        session = requests.Session()
        response = session.get(url, stream=True)

        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

        print("Model downloaded successfully!")

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

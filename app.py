from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

app = Flask(__name__)

MODEL_PATH = "malaria_model.keras"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1uiYBXrqtbm1X5F_bJEpOMrRs44_h0yYd"

# Robust download function
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")

        session = requests.Session()
        response = session.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

        print("Model downloaded!")

download_model()

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return "AfyaLens AI Running"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("malaria_model.keras")

@app.route("/")
def home():
    return "AfyaLens AI Running"

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = img.resize((128,128))
    img = np.array(img) / 255.0
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

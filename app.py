from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load TFLite model using TensorFlow (NO tflite-runtime)
interpreter = tf.lite.Interpreter(model_path="malaria_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/")
def home():
    return "AfyaLens AI Running (Render Fixed)"

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = img.resize((128,128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

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
    app.run(host="0.0.0.0", port=10000)

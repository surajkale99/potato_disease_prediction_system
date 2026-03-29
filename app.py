from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("vgg19_potato.keras")

# Class labels (same order as training)
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy"
]

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(224,224))
            img = image.img_to_array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)

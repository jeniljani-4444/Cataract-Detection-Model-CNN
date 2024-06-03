from flask import Flask, render_template, request  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image as keras_image  # type: ignore
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

class Eye:
    def __init__(self):
        try:
            # Use absolute path for the model file
            model_path = os.path.abspath("cataract_detection_model_2.h5")
            print(f"Loading model from: {model_path}")
            self.model = load_model(model_path)
            self.class_names = ['normal', 'cataract']
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def img_preprocessor(self, img):
        if self.model is None:
            return "Error", None
        
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale the image

        prediction = self.model.predict(img_array)

        if prediction[0] > 0.5:
            return self.class_names[1], prediction[0]
        else:
            return self.class_names[0], 1 - prediction[0]

eye = Eye()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            image = Image.open(file.stream)
            label, confidence = eye.img_preprocessor(image)
            if label == "Error":
                return render_template('index.html', error="Error loading model or processing image. Please check the model file and try again.")
            return render_template('index.html', label=label, confidence=confidence[0] * 100)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

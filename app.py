import tensorflow as tf
from tensorflow import keras
import streamlit as st
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image #type: ignore
from PIL import Image
from tensorflow.keras.utils import custom_object_scope #type: ignore
import tensorflowhub as hub #type: ignore


class Eye:
    def __init__(self):
        self.model = load_model("cataract_detection_model_2.h5")
        self.class_names = ['normal', 'cataract']

    
    def img_preprocessor(self, img):
            img = img.resize((224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Rescale the image

            print("Predicting...")
            prediction = self.model.predict(img_array)
            print(f"Prediction: {prediction}")

            if prediction[0] > 0.5:
                return self.class_names[1], prediction[0]
            else:
                return self.class_names[0], 1 - prediction[0]
    
    def streamlit_app(self):
        st.set_page_config(
            page_title="Cataract Classification App",
            page_icon=":eye:"
        )

        st.title("Cataract Classifier :eye:")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                st.write("Classifying...")
                label, confidence = self.img_preprocessor(image)
                if label == "Error":
                    st.write("Error loading model or processing image. Please check the model file and try again.")
                else:
                    st.write(f"Prediction: {label} ({confidence[0] * 100 :.2f}% confidence)")

if __name__ == "__main__":
    app = Eye()
    app.streamlit_app()

import os

model_path = os.path.abspath("cataract_detection_model_2.h5")
print(f"Model path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

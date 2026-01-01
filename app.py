import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("fruit_disease_model.h5")

# Class names (same as your train folders)
class_names = [
    'apple_healthy',
    'apple_black_rot',
    'apple_blotch',
    'apple_scab',
    'mango_alternaria',
    'mango_anthracnose',
    'mango_black_mold',
    'mango_healthy',
    'mango_stem_rot'
]

st.title("🍎 Fruit Disease Detection 🍌")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # same as training
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")
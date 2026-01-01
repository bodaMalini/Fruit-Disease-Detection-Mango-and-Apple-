import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(page_title="Fruit Disease Detection", page_icon="🍎", layout="centered")

# Load the trained model
model = load_model("fruit_disease_model.h5")

# Class names (same as your training folders)
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

# App title
st.title("🍎 Fruit Disease Detection 🍌")
st.write("Upload a fruit image and the model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((224, 224))        # Resize to model input
        img = np.array(img) / 255.0           # Normalize
        img = np.expand_dims(img, axis=0)     # Add batch dimension

        # Predict
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display results
        st.success(f"Prediction: {class_names[predicted_class]}")
        st.info(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error("Error processing the image. Make sure it is a valid image file.")

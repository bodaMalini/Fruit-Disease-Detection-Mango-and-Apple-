import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("apple_disease_model.keras")

class_names = [
    "apple_black_rot",
    "apple_blotch",
    "apple_healthy",
    "apple_scab"
]

CONFIDENCE_THRESHOLD = 0.60  # 60%

st.set_page_config(page_title="Apple Disease Detection")

st.title("Apple Fruit Disease Detection System")
st.write("Upload a clear image of an **apple fruit only**")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption="Uploaded Image", width=250)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    max_confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)

    if max_confidence < CONFIDENCE_THRESHOLD:
        st.warning("Invalid image ❌ Please upload a clear apple fruit image.")
    else:
        disease = class_names[predicted_class]
        st.success(f"Detected Disease: **{disease.replace('_', ' ').title()}**")

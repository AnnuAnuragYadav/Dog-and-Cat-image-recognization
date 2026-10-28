from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱", layout="centered")

MODEL_PATH = Path(__file__).parent / "cat_dog_model.keras"
IMAGE_SIZE = (128, 128)


@st.cache_resource
def load_model():
    """Load the trained model once and reuse it between Streamlit reruns."""
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload a cat or dog image and the trained CNN will make a prediction.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`")
    st.info(
        "Download `cat_dog_model.keras` from Google Colab and place it in the same "
        "folder as `app.py`. Then refresh this page."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error("The model could not be loaded. Check that it is the `.keras` file saved from Colab.")
    st.exception(exc)
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG and PNG.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This file could not be opened as an image. Please try another JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    # The model expects 128x128 RGB pixels. Its Rescaling(1./255) layer is
    # already inside the model, so do not divide the pixels by 255 here.
    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    try:
        score = float(np.asarray(model.predict(image_batch, verbose=0)).reshape(-1)[0])
    except Exception as exc:
        st.error("Prediction failed. Please confirm the model expects 128x128 RGB images.")
        st.exception(exc)
        st.stop()

    if score >= 0.5:
        predicted_class = "Dog 🐶"
        confidence = score
    else:
        predicted_class = "Cat 🐱"
        confidence = 1.0 - score

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
    st.caption("This is the model's confidence for its predicted class, not a guarantee.")

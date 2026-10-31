from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


st.set_page_config(
    page_title="PawSight · Cat & Dog AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "cat_dog_model.keras"
IMAGE_SIZE = (128, 128)
MOBILENET_SIZE = (224, 224)

# ImageNet class numbers for dog breeds and cat breeds.
DOG_CLASS_IDS = set(range(151, 269))
CAT_CLASS_IDS = set(range(281, 286))
CAT_DOG_CLASS_IDS = DOG_CLASS_IDS | CAT_CLASS_IDS


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #1d1b27;
        --muted: #777584;
        --violet: #7458f5;
        --violet-dark: #6245e5;
        --violet-soft: #f0edff;
        --line: #e9e7ee;
        --surface: #ffffff;
        --page: #f7f6f9;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1180px; padding: 1.6rem 2.2rem 3rem; }
    p { color: var(--muted); }

    .topbar {
        display:flex; justify-content:space-between; align-items:center;
        padding: 0.35rem 0 1.15rem; border-bottom:1px solid var(--line);
        margin-bottom: 2.5rem;
    }
    .brand { display:flex; gap:0.7rem; align-items:center; color:var(--ink); }
    .brand-mark {
        width:38px; height:38px; display:grid; place-items:center; border-radius:13px;
        background:var(--violet); color:white; font-size:19px;
    }
    .brand-name { font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-0.04em; font-size:1.08rem; }
    .top-tag {
        border:1px solid var(--line); color:#625f70; background:white; border-radius:99px;
        padding:0.42rem 0.75rem; font-size:0.76rem; font-weight:700; letter-spacing:0.02em;
    }
    .hero { margin-bottom:1.8rem; }
    .eyebrow {
        color:var(--violet); font-size:0.73rem; font-weight:800; letter-spacing:0.13em;
        text-transform:uppercase; margin-bottom:0.6rem;
    }
    .hero h1 {
        font-family:'Manrope',sans-serif; font-size:clamp(2.15rem,4vw,3.35rem);
        line-height:1.08; letter-spacing:-0.065em; color:var(--ink); margin:0;
    }
    .hero p { margin:0.8rem 0 0; max-width:590px; font-size:1.02rem; line-height:1.65; }
    .panel-title {
        font-family:'Manrope',sans-serif; font-size:1.02rem; font-weight:800;
        letter-spacing:-0.025em; color:var(--ink); margin:0 0 0.25rem;
    }
    .panel-note { color:var(--muted); font-size:0.86rem; margin:0 0 1rem; }
    .upload-hint {
        background:var(--violet-soft); border-radius:12px; padding:0.8rem 0.9rem;
        color:#5f5981; font-size:0.82rem; margin-top:0.8rem;
    }
    .empty-preview {
        min-height:310px; display:flex; flex-direction:column; align-items:center;
        justify-content:center; text-align:center; border:1px dashed #d9d5e3;
        border-radius:18px; background:#fbfaff; padding:1.5rem;
    }
    .empty-icon {
        width:64px; height:64px; display:grid; place-items:center; border-radius:20px;
        background:var(--violet-soft); font-size:28px; margin-bottom:1rem;
    }
    .empty-title { font-family:'Manrope',sans-serif; font-weight:800; color:var(--ink); font-size:1.02rem; }
    .empty-copy { color:var(--muted); max-width:260px; line-height:1.5; font-size:0.86rem; margin-top:0.35rem; }
    .result-label {
        color:var(--muted); font-size:0.75rem; font-weight:800; letter-spacing:0.12em;
        text-transform:uppercase; margin-bottom:0.7rem;
    }
    .result-name {
        font-family:'Manrope',sans-serif; color:var(--ink); font-size:2rem; font-weight:800;
        line-height:1.1; letter-spacing:-0.06em; margin-bottom:0.3rem;
    }
    .result-sub { color:var(--muted); font-size:0.88rem; margin-bottom:1.35rem; }
    .confidence-row { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:0.4rem; }
    .confidence-title { color:#555261; font-size:0.84rem; font-weight:700; }
    .confidence-number { color:var(--violet); font-family:'Manrope',sans-serif; font-size:1.28rem; font-weight:800; }
    .prob-title { font-size:0.78rem; font-weight:800; color:#5c5968; margin:1.15rem 0 0.25rem; }
    .footer-note { text-align:center; color:#918e9c; font-size:0.75rem; padding-top:1.3rem; }

    [data-testid="stFileUploader"] section {
        border:1.5px dashed #d9d5e3; border-radius:16px; background:#fbfaff;
        padding:1.2rem 1rem; transition:all .18s ease;
    }
    [data-testid="stFileUploader"] section:hover { border-color:var(--violet); background:#f8f6ff; }
    [data-testid="stFileUploader"] button {
        border-radius:10px; border:1px solid #ded9fb; color:var(--violet-dark);
        background:white; font-weight:700;
    }
    [data-testid="stImage"] img { border-radius:16px; }

    div.stButton > button[kind="primary"] {
        width:100%; border:0; border-radius:12px; min-height:3rem;
        background:var(--violet); color:white !important; font-weight:800; transition:all .18s ease;
        box-shadow:0 7px 18px rgba(116,88,245,.2);
    }
    div[data-testid="stButton"] button[kind="primary"] *,
    div[data-testid="stButton"] button[kind="primary"] span {
        color:white !important;
        -webkit-text-fill-color:white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background:var(--violet-dark); border:0; transform:translateY(-1px);
        box-shadow:0 10px 22px rgba(116,88,245,.26);
    }
    div[data-testid="stProgress"] > div > div { background:var(--violet); }
    @media (max-width: 760px) {
        .block-container { padding:1.2rem 1rem 2rem; }
        .topbar { margin-bottom:1.8rem; }
        .top-tag { font-size:0.68rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="topbar">
      <div class="brand"><div class="brand-mark">🐾</div><div class="brand-name">PawSight</div></div>
      <div class="top-tag">AI IMAGE CLASSIFIER</div>
    </div>
    <div class="hero">
      <div class="eyebrow">A little AI, a lot of paws</div>
      <h1>Who’s in the picture?</h1>
      <p>Drop in a photo and let PawSight take a look. Your trained CNN will tell you whether it sees a cat or a dog.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if not MODEL_PATH.exists():
    st.error(f"Model file missing: `{MODEL_PATH.name}`")
    st.info("Place `cat_dog_model.keras` in the same folder as `app.py`, then refresh the app.")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_mobilenet():
    return tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3),
    )


def image_has_cat_or_dog(image):
    resized_image = image.resize(MOBILENET_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)[None, ...]
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    scores = load_mobilenet().predict(image_array, verbose=0)[0]
    top_five = np.argsort(scores)[-5:]
    top_five_has_pet = any(int(class_id) in CAT_DOG_CLASS_IDS for class_id in top_five)

    # MobileNet ka check heuristic hai; combined score se asli pet photo
    # galti se reject hone ka chance kam hota hai.
    pet_probability = float(np.sum(scores[list(CAT_DOG_CLASS_IDS)]))
    return top_five_has_pet or pet_probability >= 0.10


try:
    model = load_model()
except Exception as exc:
    st.error("PawSight could not load the trained model. Check that the `.keras` file is correct.")
    st.exception(exc)
    st.stop()


left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">01 · Add your photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">Choose a clear image of a cat or dog.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="JPG and PNG images are supported.",
    )
    st.markdown(
        '<div class="upload-hint">✦ &nbsp; Best results with one pet in focus and good lighting.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="panel-title">02 · Your result</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">Your prediction will appear here.</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown(
            '<div class="empty-preview"><div class="empty-icon">🐶</div>'
            '<div class="empty-title">Ready when you are</div>'
            '<div class="empty-copy">Upload a photo on the left to see the image and its prediction here.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("Could not open this image. Try another JPG or PNG file.")
            st.stop()

        st.image(image, use_container_width=True)
        analyze = st.button("Analyze photo  →", type="primary", use_container_width=True)

        if analyze:
            try:
                with st.spinner("Checking your photo..."):
                    is_cat_or_dog = image_has_cat_or_dog(image)
            except Exception as exc:
                st.error("The photo check failed. Please try another image.")
                st.exception(exc)
                st.stop()

            if not is_cat_or_dog:
                st.warning("It only detects cat or dog photos. Please upload a cat or dog photo.")
            else:
                # Model ke andar Rescaling(1./255) layer pehle se hai.
                resized_image = image.resize(IMAGE_SIZE)
                image_array = np.asarray(resized_image, dtype=np.float32)[None, ...]

                try:
                    score = float(np.asarray(model.predict(image_array, verbose=0)).reshape(-1)[0])
                except Exception as exc:
                    st.error("Prediction failed. Check that the model expects 128×128 RGB images.")
                    st.exception(exc)
                    st.stop()

                is_dog = score >= 0.5
                label = "Dog" if is_dog else "Cat"
                emoji = "🐶" if is_dog else "🐱"
                confidence = score if is_dog else 1.0 - score
                cat_probability = 1.0 - score

                st.markdown(
                    '<div class="result-label">Pawsitive identification</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-name">{emoji} {label}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="result-sub">PawSight’s best guess for this photo</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="confidence-row"><span class="confidence-title">Model confidence</span>'
                    f'<span class="confidence-number">{confidence * 100:.2f}%</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                st.markdown('<div class="prob-title">Class scores</div>', unsafe_allow_html=True)
                st.caption(f"Cat  {cat_probability * 100:.1f}%   ·   Dog  {score * 100:.1f}%")


st.markdown(
    '<div class="footer-note">Made with care for curious pet people · PawSight CNN</div>',
    unsafe_allow_html=True,
)
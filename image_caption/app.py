import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import cv2
import streamlit as st
import tempfile
import random
from object_detection import simple_object_detection
from image_segmentation import deeplabv3_segment

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="PixelGenius",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern, Clean UI ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;400&display=swap');
    html, body, .stApp {
        font-family: 'Montserrat', sans-serif !important;
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%) !important;
    }
    .stApp {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%) !important;
    }
    .header-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }
    .header-logo img {
        width: 60px;
        height: 60px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(80,80,120,0.10);
    }
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(270deg, #43cea2, #185a9d, #f6d365, #fda085, #43cea2);
        background-size: 1000% 1000%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientMove 8s ease infinite;
        margin-bottom: 0.1rem;
    }
    @keyframes gradientMove {
        0% {background-position:0% 50%}
        50% {background-position:100% 50%}
        100% {background-position:0% 50%}
    }
    .header-tagline {
        text-align: center;
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .result-card {
        background: rgba(255,255,255,0.95);
        border-radius: 22px;
        box-shadow: 0 6px 32px rgba(80, 80, 120, 0.13);
        padding: 1.5rem 2.2rem 1.5rem 2.2rem;
        margin: 1.5rem 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(255,255,255,0.7);
        color: #22223b;
        text-align: center;
        padding: 0.7rem 0;
        font-size: 1.1rem;
        border-top: 1px solid #f6d365;
        font-family: 'Montserrat', sans-serif !important;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown(
    '''<div class="header-logo">
        <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" alt="logo"/>
        <span class="header-title">PixelGenius</span>
    </div>''', unsafe_allow_html=True)
st.markdown(
    '<div class="header-tagline">Your all-in-one AI image suite: Captioning, Segmentation, Detection</div>',
    unsafe_allow_html=True
)

# --- Sidebar: Input Method ---
with st.sidebar:
    st.markdown("## Input Image")
    mode = st.radio(
        "Choose input method:",
        ("Upload Image", "Capture from Webcam"),
        horizontal=False,
        index=0
    )
    img = None
    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
    elif mode == "Capture from Webcam":
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            img = Image.open(camera_image).convert('RGB')

# --- Paths ---
CAPTION_MODEL_PATH = os.path.join('image_caption_sub', 'output', 'model.h5')
TOKENIZER_PATH = os.path.join('image_caption_sub', 'output', 'tokenizer.pkl')

# --- Model URLs ---
EFFICIENTDET_URL = "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1"
SSDMNV2_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

# --- Loaders ---
@st.cache_resource
def load_captioning_model():
    if not os.path.exists(CAPTION_MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    try:
        model = load_model(CAPTION_MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading captioning model or tokenizer: {e}")
        return None, None

# --- Utility Functions ---
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_vgg16_feature(img):
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    vgg_model = VGG16()
    vgg_model = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    image = img.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def predict_caption(model, image_features, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([image_features, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# --- Main Area ---
if img is not None:
    st.markdown("<div style='display:flex; justify-content:center; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
    st.image(img, caption="Input Image", use_column_width=False, width=420)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Results ---
    # 1. Image Caption
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("1. Image Caption")
    with st.spinner('Generating caption...'):
        model, tokenizer = load_captioning_model()
        if model is not None and tokenizer is not None:
            max_length = 49  # Set according to your training
            features = extract_vgg16_feature(img)
            caption = predict_caption(model, features, tokenizer, max_length)
            st.success(f"Caption: {caption}")
            st.download_button("Download Caption", caption, file_name="caption.txt")
        else:
            st.warning("Captioning model or tokenizer not found. Please ensure 'output/model.h5' and 'output/tokenizer.pkl' exist.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Segmentation (DeepLabV3+)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("2. Image Segmentation (DeepLabV3+)")
    try:
        with st.spinner('Segmenting image with DeepLabV3+...'):
            mask, overlay = deeplabv3_segment(img)
            st.image(overlay, caption="DeepLabV3+ Segmentation Overlay", use_column_width=True)
            st.image(mask, caption="Segmentation Mask", use_column_width=True, channels="GRAY")
    except Exception as e:
        st.warning(f"DeepLabV3+ could not be loaded: {e}.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Object Detection
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("3. Object Detection")
    with st.spinner('Detecting objects...'):
        image_with_boxes, detections = simple_object_detection(img, threshold=0.5)
        st.image(image_with_boxes, caption="Object Detection", use_column_width=True)
        if detections:
            st.info(f"Detected objects: {', '.join(set([d['label'] for d in detections]))}")
        else:
            st.info("No objects detected.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please upload an image or use the webcam.")

# --- Footer ---
st.markdown(
    '<div class="footer">Made with ❤️ by <a href="https://github.com/your-github" target="_blank">PixelGenius</a></div>',
    unsafe_allow_html=True
) 
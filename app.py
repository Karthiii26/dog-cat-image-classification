"""
app.py
======
A clean, minimal Streamlit web application for Dog-Cat Image Classification
using a 3-layer Convolutional Neural Network.
"""

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import streamlit as st
import numpy as np

from predict import preprocess_image, VALID_IMAGE_EXTENSIONS

st.set_page_config(
    page_title="Dog and Cat Image Classification",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Loads and caches the trained Keras CNN model."""
    import tensorflow as tf
    model_path = Path("models/dog_cat_cnn.keras")
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(str(model_path))


def classify_image(image: Image.Image, model):
    """Preprocesses input image and returns predicted class and confidence."""
    batch_array = preprocess_image(image, target_size=(128, 128))
    raw_output = float(model.predict(batch_array, verbose=0)[0][0])

    if raw_output >= 0.5:
        predicted_class = "Dog"
        confidence = raw_output * 100.0
    else:
        predicted_class = "Cat"
        confidence = (1.0 - raw_output) * 100.0

    return predicted_class, confidence, raw_output


# Page Title and Description
st.title("Dog and Cat Image Classification")
st.write(
    "This application classifies images as either a Dog or a Cat using a custom "
    "Convolutional Neural Network built with three convolutional layers."
)

model = load_trained_model()
if model is None:
    st.error("Trained model not found at 'models/dog_cat_cnn.keras'. Please run 'python train.py' first.")
    st.stop()

# Image Input Section
selected_image = None
image_caption = ""

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid image.")
        else:
            img = Image.open(uploaded_file)
            img.verify()
            uploaded_file.seek(0)
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_caption = uploaded_file.name
    except (UnidentifiedImageError, OSError) as e:
        st.error(f"Invalid image file: {e}")
else:
    sample_dir = Path("sample_images")
    if sample_dir.exists():
        samples = sorted([p for p in sample_dir.glob("*.*") if p.suffix.lower() in VALID_IMAGE_EXTENSIONS])
        if samples and st.checkbox("Or select a sample image from the test set"):
            sample_names = [p.name for p in samples]
            selected_name = st.selectbox("Sample image", sample_names)
            selected_image = Image.open(sample_dir / selected_name).convert("RGB")
            image_caption = selected_name

# Display and Prediction Section
if selected_image is not None:
    st.write("")
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("Image")
        st.image(selected_image, caption=image_caption, use_column_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Classifying..."):
            predicted_class, confidence, raw_score = classify_image(selected_image, model)

        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.progress(float(confidence / 100.0))
        st.caption(f"Raw sigmoid output: {raw_score:.4f}")

st.write("---")

# Collapsible Technical Information
with st.expander("Model Architecture"):
    st.markdown(
        """
        - **Convolutional Layer 1:** Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
        - **Convolutional Layer 2:** Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
        - **Convolutional Layer 3:** Conv2D (128 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
        - **Flatten**
        - **Dense Layer:** 128 neurons, ReLU activation
        - **Dropout:** 0.5 rate
        - **Output Layer:** 1 neuron, Sigmoid activation
        - **Loss:** Binary Cross-Entropy
        - **Optimizer:** Adam
        - **Input Dimensions:** 128x128 pixels normalized to [0, 1]
        """
    )

with st.expander("Training & Evaluation Results"):
    history_plot = Path("plots/training_history.png")
    cm_plot = Path("plots/confusion_matrix.png")
    report_file = Path("plots/classification_report.txt")

    if history_plot.exists():
        st.subheader("Training and Validation Curves")
        st.image(str(history_plot), use_column_width=True)

    if cm_plot.exists():
        st.subheader("Confusion Matrix (Test Split)")
        st.image(str(cm_plot), use_column_width=True)

    if report_file.exists():
        st.subheader("Classification Report (Test Split)")
        with open(report_file, "r") as f:
            st.code(f.read(), language="text")

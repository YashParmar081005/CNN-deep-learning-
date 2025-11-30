# app.py
import io
import os
import tempfile
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)

st.set_page_config(page_title="CNN Predictor", layout="centered")


# -------------------------
# Helpers: Pillow resampling compatibility
# -------------------------
# Pillow >=10: Image.Resampling.LANCZOS
# older Pillow: Image.LANCZOS or Image.ANTIALIAS
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except Exception:
    # fallback for older Pillow versions
    try:
        RESAMPLE_LANCZOS = Image.LANCZOS
    except Exception:
        RESAMPLE_LANCZOS = Image.ANTIALIAS


# -------------------------
# Model builder + caching (stable under Streamlit)
# -------------------------
def clear_keras_session():
    """Clear previous TF/Keras session to avoid name-scope errors on re-run."""
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


def _build_cnn_model(input_shape=(28, 28, 1), n_classes=10):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource
def get_model(input_shape=(28, 28, 1), n_classes=10):
    """
    Returns a compiled model. This function is cached by Streamlit so
    the model is created once per session / parameter set.
    """
    # Ensure pure python types for caching stability
    input_shape = tuple(int(x) for x in input_shape)
    n_classes = int(n_classes)

    clear_keras_session()
    model = _build_cnn_model(input_shape=input_shape, n_classes=n_classes)
    return model


# -------------------------
# Preprocessing utilities
# -------------------------
def preprocess_image_pil(img_pil, target_size=(28, 28)):
    """
    Convert PIL image to grayscale target_size numpy array normalized to [0,1].
    Output shape: (1, h, w, 1)
    """
    # convert to grayscale
    img_gray = ImageOps.grayscale(img_pil)

    # resize using chosen resampling constant
    img_resized = img_gray.resize(target_size, RESAMPLE_LANCZOS)

    # convert to numpy array
    arr = np.array(img_resized).astype(np.float32)

    # normalize to 0..1
    arr = arr / 255.0

    # reshape for CNN input
    arr = arr.reshape(1, target_size[0], target_size[1], 1)
    return arr


# -------------------------
# UI
# -------------------------
st.title("📷 CNN Image Predictor (Streamlit)")
st.write(
    "Upload an image and (optionally) your trained `.h5` weights file.\n"
    "The app will preprocess the image to 28×28 grayscale and show the predicted class & probabilities."
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_image = st.file_uploader(
        "Upload an image (jpg/png). Prefer single-digit / centered images if using MNIST-like model.",
        type=["png", "jpg", "jpeg"]
    )
    st.markdown("**Optional:** If you have a trained Keras weights file (`.h5`), upload it below to get real predictions.")
    weights_file = st.file_uploader("Upload model weights (.h5)", type=["h5"])

with col2:
    st.write("Model settings")
    n_classes = st.number_input("Number of classes", value=10, min_value=2, max_value=100, step=1)
    invert_checkbox = st.checkbox("Invert colors (white bg → black bg) if predictions look wrong", value=False)
    show_probs = st.checkbox("Show class probabilities chart", value=True)

# Build (or fetch cached) model
model = get_model(input_shape=(28, 28, 1), n_classes=n_classes)

# Load weights if provided (save to a temp file first, then load)
if weights_file is not None:
    tmp_weights = None
    try:
        with st.spinner("Saving uploaded weights and loading into model..."):
            # write uploaded bytes to a temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            tmp.write(weights_file.read())
            tmp.flush()
            tmp.close()
            tmp_weights = tmp.name

            # load weights
            model.load_weights(tmp_weights)
        st.success("Loaded weights successfully ✅")
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        st.info("The model will still run with randomly initialized weights (predictions will not be meaningful).")
    finally:
        # remove temp file if it exists
        try:
            if tmp_weights and os.path.exists(tmp_weights):
                os.unlink(tmp_weights)
        except Exception:
            pass

# Prediction block
if uploaded_image is not None:
    # Read image
    image_bytes = uploaded_image.read()
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    # Show original
    st.subheader("Input Image")
    st.image(img_pil, use_column_width=True)

    # Preprocess
    arr = preprocess_image_pil(img_pil, target_size=(28, 28))
    if invert_checkbox:
        arr = 1.0 - arr

    # Show processed preview
    st.subheader("Preprocessed (28×28 grayscale) — model input")
    preview = (arr[0, :, :, 0] * 255).astype(np.uint8)
    st.image(preview, width=150, clamp=True, channels="GRAY")

    # Predict
    with st.spinner("Running model prediction..."):
        try:
            preds = model.predict(arr)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        preds = preds.reshape(-1)
        pred_class = int(np.argmax(preds))
        top_prob = float(np.max(preds))

    st.success(f"Predicted class: **{pred_class}**  —  Confidence: **{top_prob * 100:.2f}%**")

    if show_probs:
        # display probabilities bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        classes = list(range(len(preds)))
        ax.bar(classes, preds)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_title("Class probabilities")
        ax.set_xticks(classes)
        st.pyplot(fig)

    # Allow user to download the preprocessed array (optional)
    if st.button("Download preprocessed array (.npy)"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        try:
            np.save(tmp.name, arr)
            tmp.close()
            with open(tmp.name, "rb") as f:
                st.download_button(
                    label="Download .npy",
                    data=f,
                    file_name="preprocessed_input.npy",
                    mime="application/octet-stream"
                )
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

else:
    st.info("Upload an image to see predictions. Or upload model weights to use a trained model.")

# Footer notes
st.markdown("---")
st.markdown(
    """
    **Notes & Tips**
    - This frontend expects a CNN trained on 28×28 grayscale images (e.g. MNIST).  
    - If your notebook used different preprocessing (normalization, inversion, centering), adjust `preprocess_image_pil()` accordingly.  
    - To use your trained model: in your notebook call `model.save_weights("my_weights.h5")` and upload the `.h5` file here.  
    - If you prefer uploading a full saved model (`model.save("model.h5")`), I can give an alternate version of this app that uses `tf.keras.models.load_model()` directly.
    """
)

st.markdown("Developed with ❤️ using Streamlit and TensorFlow/Keras.")

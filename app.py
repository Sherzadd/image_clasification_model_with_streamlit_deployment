import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.caption("Upload a leaf image and this app will predict the plant disease class using your trained TensorFlow/Keras model.")

# -----------------------------
# Beginner-friendly explanation
# -----------------------------
with st.expander("📘 Beginner explanation (click to open)"):
    st.markdown(
        """
**What happens when you upload an image?**

1. **You upload** a JPG/PNG leaf photo.
2. The app **loads your saved model** (the same model you trained in your notebook).
3. The app **resizes** the image to the size your model expects (256×256).
4. The app runs `model.predict(...)` to get **probabilities** for each class.
5. The app picks the biggest probability (using `argmax`) and shows you the **predicted label**.
        """
    )

# -----------------------------
# Helper functions
# -----------------------------
def load_class_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list, e.g. ['class1','class2',...]")
    return names


@st.cache_resource
def load_model_cached(model_path: str):
    # Cache the model so it doesn't reload on every Streamlit rerun
    return tf.keras.models.load_model(model_path, compile=False)


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    """Return True if the model (even nested) contains a Rescaling layer.

    Your training notebook builds the model with:
      resize_and_rescale = Sequential([Resizing(...), Rescaling(1/255)])
    which is nested inside the main Sequential.
    """
    def _has(layer) -> bool:
        # Direct match (works across Keras/TensorFlow versions)
        if layer.__class__.__name__.lower() == "rescaling":
            return True

        # If this layer itself contains sub-layers (e.g., Sequential inside Sequential), search recursively
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if _has(sub):
                    return True
        return False

    return _has(model)


def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """
    Convert PIL image -> NumPy batch (1, H, W, 3)

    We resize to the model input size because your saved model expects (256, 256, 3).
    We do NOT divide by 255 if the model already contains Rescaling(1/255).
    """
    img = img.convert("RGB")

    # Read model input shape like: (None, 256, 256, 3)
    in_shape = getattr(model, "input_shape", None)

    # Resize if we have a fixed size
    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        target_h, target_w = in_shape[1], in_shape[2]
        if target_h is not None and target_w is not None:
            img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img)              # (H, W, 3), uint8
    x = np.expand_dims(x, 0)       # (1, H, W, 3)
    x = x.astype("float32")        # match tf.data pipeline dtype

    # Notebook model already includes Rescaling(1/255) inside the model.
    # So we ONLY scale here if the model does NOT contain a Rescaling layer.
    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    """
    Ensure the output behaves like probabilities.
    If it doesn't sum to ~1, apply softmax.
    """
    pred_vector = np.asarray(pred_vector).astype("float32")
    s = float(pred_vector.sum())
    if not (0.98 <= s <= 1.02) or (pred_vector.min() < 0):
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("⚙️ Settings")

default_model_path = "models/image_classification_model.keras"
default_classes_path = "class_names.json"

model_path = st.sidebar.text_input("Model path", value=default_model_path)
classes_path = st.sidebar.text_input("Class names file", value=default_classes_path)

st.sidebar.markdown(
    """
**Where should my model be?**

- Your training notebook uses a path like: `../models/image_classification_model.keras`
- In this repo, that means the model should be at: `models/image_classification_model.keras`

If your model file is somewhere else, change the path above.
    """
)

# Load model
model = None
if os.path.exists(model_path):
    try:
        model = load_model_cached(model_path)
        st.sidebar.success("Model loaded ✅")
        st.sidebar.caption(f"Input shape: {getattr(model, 'input_shape', None)}")
    except Exception as e:
        st.sidebar.error("Model found, but failed to load ❌")
        st.sidebar.write(str(e))
else:
    st.sidebar.warning("Model file not found ❗")
    st.sidebar.caption("Train + save your model first, or change the model path here.")

# Load class names
class_names = None
if os.path.exists(classes_path):
    try:
        class_names = load_class_names(classes_path)
        st.sidebar.success(f"Loaded {len(class_names)} class names ✅")

        # Sanity check: model output units should match number of class names
        try:
            if model is not None and hasattr(model, "output_shape"):
                out_dim = model.output_shape[-1]
                if out_dim is not None and int(out_dim) != len(class_names):
                    st.sidebar.error(
                        f"Mismatch: model outputs {int(out_dim)} classes, but class_names.json has {len(class_names)}."
                    )
        except Exception:
            pass

    except Exception as e:
        st.sidebar.error("class_names.json found, but failed to load ❌")
        st.sidebar.write(str(e))
else:
    st.sidebar.warning("class_names file not found ❗")
    st.sidebar.caption("Keep class_names.json next to app.py (repo root) or change the path above.")

st.divider()

# -----------------------------
# Upload + Predict
# -----------------------------
import io
import hashlib

# Stop early if model/classes not ready
if model is None:
    st.error("Model is not loaded. Please fix the model path in the sidebar.")
    st.stop()

if class_names is None:
    st.error("class_names.json is not loaded. Please fix the class names file path in the sidebar.")
    st.stop()

# --- Reset button ---
if st.sidebar.button("Reset / Clear image"):
    st.session_state["last_hash"] = None
    st.session_state["last_pred"] = None
    st.session_state["last_probs"] = None
    st.rerun()

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="uploader")

if uploaded is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

# Read bytes (this is important)
img_bytes = uploaded.getvalue()

# Compute a hash so we know when the file actually changed
img_hash = hashlib.md5(img_bytes).hexdigest()

# Open image from bytes
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

# ✅ Preprocess EXACTLY like your notebook training pipeline:
# - resize to model input (256x256)
# - DO NOT manually rescale if the model already contains Rescaling(1/255)
x = preprocess(img, model)

# Only predict if image changed (or first time)
if st.session_state.get("last_hash") != img_hash or st.session_state.get("last_probs") is None:
    preds = model.predict(x, verbose=0)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.asarray(preds)
    if preds.ndim == 2:
        preds = preds[0]  # (n_classes,)

    probs = to_probabilities(preds)

    pred_id = int(np.argmax(probs))
    if pred_id >= len(class_names):
        st.error(
            f"Prediction index {pred_id} is outside your class_names list (length {len(class_names)}).\\n\\n"
            "Fix: make sure `class_names.json` matches the model output order."
        )
        st.stop()

    st.session_state["last_hash"] = img_hash
    st.session_state["last_probs"] = probs
    st.session_state["last_pred"] = pred_id

# Read cached results
probs = st.session_state["last_probs"]
pred_id = int(st.session_state["last_pred"])
pred_label = class_names[pred_id]
confidence = float(probs[pred_id])

st.success(f"✅ Predicted class: **{pred_label}**")
st.write(f"Confidence: **{confidence:.2%}**")

# Top-K
st.subheader("3) Top predictions")
top_k = min(5, len(probs))
top_idx = np.argsort(probs)[::-1][:top_k]

for rank, i in enumerate(top_idx, start=1):
    st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

st.caption("Tip: If predictions look wrong, double-check your class_names.json order and preprocessing.")

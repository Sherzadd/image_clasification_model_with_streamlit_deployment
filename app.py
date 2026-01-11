import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# -----------------------------
# ✅ Page setup (same style, no Settings sidebar)
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification through Artificial Intelligence",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🌿 Plant Disease identification through Artificial Intelligence")
st.caption(
    "Upload a plant leaf image and this app will identify the plant disease using a trained "
    "Artificial Intelligence model (TensorFlow/Keras)."
)

# -----------------------------
# ✅ Hidden (internal) paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = (BASE_DIR / "models").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()

# Prefer these model names if present; otherwise pick the first .keras in models/
PREFERRED_MODEL_NAMES = [
    "image_classification_model_linux.keras",
    "image_classification_model.keras",
]


def pick_model_file() -> Path | None:
    """Pick a model file from models/ without exposing any UI setting."""
    for name in PREFERRED_MODEL_NAMES:
        p = (MODELS_DIR / name).resolve()
        if p.exists():
            return p

    if MODELS_DIR.exists():
        all_models = sorted(MODELS_DIR.glob("*.keras"))
        if all_models:
            return all_models[0].resolve()

    return None


# -----------------------------
# Helper functions
# -----------------------------
def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    # Cache model and reload automatically when the file changes (mtime changes)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    """Return True if the model (even nested) contains a Rescaling layer."""
    def _has(layer) -> bool:
        if layer.__class__.__name__.lower() == "rescaling":
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if _has(sub):
                    return True
        return False

    return _has(model)


def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """
    Convert PIL image -> NumPy batch (1, H, W, 3)

    Resize to the model input size.
    Do NOT divide by 255 if the model already contains Rescaling(1/255).
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
    x = x.astype("float32")

    # Only scale if the model does NOT contain a Rescaling layer
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
# Load model + class names (hidden, no sidebar)
# -----------------------------
model_path = pick_model_file()
if model_path is None:
    st.error("⚠️ Model file not found in the `models/` folder. Please contact the app owner.")
    st.stop()

if not CLASSES_PATH.exists():
    st.error("⚠️ `class_names.json` not found next to `app.py`. Please contact the app owner.")
    st.stop()

try:
    model = load_model_cached(str(model_path), model_path.stat().st_mtime)
except Exception:
    st.error("⚠️ The model could not be loaded in this environment. Please contact the app owner.")
    st.stop()

try:
    class_names = load_class_names(CLASSES_PATH)
except Exception:
    st.error("⚠️ Class labels could not be loaded. Please contact the app owner.")
    st.stop()

# Optional quiet sanity check
try:
    if hasattr(model, "output_shape") and model.output_shape[-1] is not None:
        out_dim = int(model.output_shape[-1])
        if out_dim != len(class_names):
            st.error("⚠️ Internal mismatch (model outputs vs class labels). Please contact the app owner.")
            st.stop()
except Exception:
    pass


# -----------------------------
# Layout: User manual LEFT, App RIGHT (no camera option)
# -----------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("📘 User Manual")

    st.markdown(
        """
**How to take a good photo (important):**
- Use **bright natural light** (avoid very dark photos).
- Keep the leaf **in focus** (no blur).
- Capture **one leaf clearly** (fill most of the frame).
- Use a **plain background** if possible.
- Avoid strong **shadows**, **reflections**, and **filters**.
- Don’t crop too tightly — include the **full infected area**.

**How to use the app:**
1. **Upload an image** (PNG / JPG / JPEG).
2. Wait a moment for the prediction.
3. Read the **predicted class** and **confidence**.
        """
    )

with right:
    st.subheader("🌿 Upload a leaf image")

    # --- Reset button (kept, but moved to main area) ---
    if st.button("Reset / Clear"):
        st.session_state["last_hash"] = None
        st.session_state["last_pred"] = None
        st.session_state["last_probs"] = None
        st.rerun()

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        st.stop()

    # Read bytes
    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    # Open image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption="Selected image", use_container_width=True)

    # Preprocess + predict
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
                "⚠️ Prediction index is outside the class label list. "
                "This means class_names.json does not match the model output order."
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

    st.subheader("Top predictions")
    top_k = min(5, len(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]

    for rank, i in enumerate(top_idx, start=1):
        st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

    st.caption("Tip: If results look wrong, try a brighter/sharper photo with a plain background.")

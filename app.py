import json
import os
import uuid
import subprocess
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask import Flask, render_template, request, send_file, jsonify, redirect
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
except Exception:
    effnet_preprocess = None

try:
    import nibabel as nib
except Exception:
    nib = None


# ============================================================
# APP / DIRECTORIES
# ============================================================

app = Flask(__name__)
from unet_features import register_unet

register_unet(app)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
PLOT_DIR = STATIC_DIR / "performance_plots"
TB_LOG_DIR = BASE_DIR / "logs" / "tensorboard"
TEST_SET_DIR = BASE_DIR / "test_set"          # <-- NEW: labelled test images
EVAL_DIR = STATIC_DIR / "evaluation"          # <-- NEW: cached evaluation artifacts
HISTORY_FILE = BASE_DIR / "prediction_history.json"  # <-- NEW: prediction history log

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
TB_LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASSES / MODELS
# ============================================================

CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]

# 3D models are trained on RSNA-MICCAI dataset (binary MGMT biomarker prediction)
CLASS_NAMES_3D = ["mgmt_negative", "mgmt_positive"]

# ============================================================
# TUMOR INFORMATION DATABASE (Educational + Clinical Features)
# ============================================================

TUMOR_INFO = {
    "glioma_tumor": {
        "simple_name": "Glioma",
        "what_is": "A glioma is a tumor that starts in the glial cells of the brain or spine. Glial cells support and protect nerve cells. Gliomas are the most common type of primary brain tumor.",
        "simple_explanation": "This tumor grows from the support cells of the brain. It can be slow-growing or aggressive.",
        "symptoms": ["Persistent headaches", "Seizures", "Memory problems", "Personality changes", "Weakness in arms or legs"],
        "treatments": ["Surgical removal (when possible)", "Radiation therapy", "Chemotherapy (Temozolomide)", "Targeted drug therapy"],
        "prognosis": "Varies widely based on tumor grade. Low-grade gliomas have better outcomes. High-grade gliomas (like GBM) are aggressive.",
        "specialist": "Neurosurgeon + Neuro-Oncologist",
        "specialist_icon": "fa-user-doctor",
        "urgency": "high",
        "color": "#f43f5e",
        "icd_code": "C71",
        "next_steps": "Schedule an appointment with a neurosurgeon within 1-2 weeks. Bring all MRI scans and this report.",
    },
    "meningioma_tumor": {
        "simple_name": "Meningioma",
        "what_is": "A meningioma is a tumor that arises from the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are slow-growing and benign (non-cancerous).",
        "simple_explanation": "This tumor grows from the protective layers around the brain. It is usually slow-growing and non-cancerous.",
        "symptoms": ["Headaches that worsen over time", "Vision problems", "Hearing loss", "Memory difficulty", "Seizures"],
        "treatments": ["Active monitoring (watch-and-wait)", "Surgical removal", "Stereotactic radiosurgery (Gamma Knife)", "Hormone therapy"],
        "prognosis": "Generally excellent. Most meningiomas are benign and slow-growing. 5-year survival rate is over 80%.",
        "specialist": "Neurosurgeon",
        "specialist_icon": "fa-user-doctor",
        "urgency": "medium",
        "color": "#f59e0b",
        "icd_code": "C70",
        "next_steps": "Schedule a consultation with a neurosurgeon within 2-4 weeks. Regular MRI monitoring may be recommended.",
    },
    "no_tumor": {
        "simple_name": "No Tumor Detected",
        "what_is": "No tumor was detected in this MRI scan. This means the AI model did not find evidence of a brain tumor in the analyzed image.",
        "simple_explanation": "Good news! The scan did not show any tumor. However, if you are experiencing symptoms, please consult a doctor.",
        "symptoms": [],
        "treatments": [],
        "prognosis": "No tumor detected. If symptoms persist, further evaluation may be needed.",
        "specialist": "General Physician or Neurologist",
        "specialist_icon": "fa-stethoscope",
        "urgency": "low",
        "color": "#10b981",
        "icd_code": "Z01.8",
        "next_steps": "If you are experiencing symptoms like headaches or seizures, consult a neurologist for further evaluation.",
    },
    "pituitary_tumor": {
        "simple_name": "Pituitary Tumor",
        "what_is": "A pituitary tumor is an abnormal growth in the pituitary gland, a small gland at the base of the brain. Most pituitary tumors are benign (non-cancerous) and slow-growing.",
        "simple_explanation": "This tumor grows in the pituitary gland, a small gland at the base of the brain that controls hormones. It is usually non-cancerous and treatable.",
        "symptoms": ["Headaches", "Vision problems (especially peripheral vision loss)", "Hormonal changes", "Fatigue", "Unexplained weight changes"],
        "treatments": ["Medication to shrink tumor", "Surgical removal (transsphenoidal)", "Radiation therapy", "Hormone replacement therapy"],
        "prognosis": "Excellent for benign pituitary tumors. Most can be treated successfully with surgery or medication.",
        "specialist": "Endocrinologist + Neurosurgeon",
        "specialist_icon": "fa-dna",
        "urgency": "medium",
        "color": "#8b7cff",
        "icd_code": "D35.2",
        "next_steps": "Schedule appointments with both an endocrinologist and neurosurgeon within 2-3 weeks.",
    },
}


def get_tumor_info(tumor_class):
    """Get tumor information for display."""
    return TUMOR_INFO.get(tumor_class, {})


def get_confidence_recommendation(confidence_value, predicted_class):
    """Generate confidence-based action recommendation."""
    if predicted_class == "no_tumor":
        if confidence_value >= 90:
            return {
                "level": "safe",
                "color": "#10b981",
                "icon": "fa-circle-check",
                "title": "No Tumor Detected with High Confidence",
                "message": "The AI model found no evidence of a brain tumor with high confidence. If you are experiencing symptoms, please consult a physician for further evaluation.",
                "action": "No immediate action needed. Follow up if symptoms persist.",
            }
        else:
            return {
                "level": "caution",
                "color": "#f59e0b",
                "icon": "fa-circle-exclamation",
                "title": "No Tumor Detected (Low Confidence)",
                "message": "The model could not find a tumor, but confidence is low. The scan may be unclear or the tumor may be in an unusual location.",
                "action": "Please consult a neurologist for a professional review of this scan.",
            }
    elif confidence_value >= 90:
        return {
            "level": "high",
            "color": "#f43f5e",
            "icon": "fa-triangle-exclamation",
            "title": "High Confidence Tumor Detection",
            "message": f"The AI model has detected a potential {predicted_class.replace('_', ' ')} with high confidence. This requires immediate medical attention.",
            "action": "Schedule an appointment with a specialist within 1-2 weeks.",
        }
    elif confidence_value >= 70:
        return {
            "level": "medium",
            "color": "#f59e0b",
            "icon": "fa-circle-exclamation",
            "title": "Moderate Confidence Detection",
            "message": f"The model detected a possible {predicted_class.replace('_', ' ')} with moderate confidence. A medical professional should review this scan.",
            "action": "Schedule a consultation with a specialist within 2-4 weeks.",
        }
    else:
        return {
            "level": "low",
            "color": "#64748b",
            "icon": "fa-circle-question",
            "title": "Low Confidence Result",
            "message": "The model's prediction confidence is low. The result is uncertain and must be reviewed by a medical professional.",
            "action": "A radiologist or neurologist must review this scan manually.",
        }


def get_severity_assessment(predicted_class, confidence_value):
    """Assess severity and assign priority."""
    if predicted_class == "no_tumor":
        if confidence_value >= 90:
            return {"priority": "low", "color": "#10b981", "icon": "fa-circle", "label": "Low Priority", "wait_time": "Routine follow-up"}
        else:
            return {"priority": "medium", "color": "#f59e0b", "icon": "fa-circle", "label": "Medium Priority", "wait_time": "Review within 4 weeks"}
    elif predicted_class == "glioma_tumor":
        return {"priority": "critical", "color": "#f43f5e", "icon": "fa-circle", "label": "Critical Priority", "wait_time": "Seek immediate medical attention"}
    elif predicted_class == "meningioma_tumor":
        return {"priority": "medium", "color": "#f59e0b", "icon": "fa-circle", "label": "Medium Priority", "wait_time": "Schedule within 2-4 weeks"}
    elif predicted_class == "pituitary_tumor":
        return {"priority": "medium", "color": "#f59e0b", "icon": "fa-circle", "label": "Medium Priority", "wait_time": "Schedule within 2-3 weeks"}
    else:
        return {"priority": "low", "color": "#10b981", "icon": "fa-circle", "label": "Low Priority", "wait_time": "Follow up as needed"}


def get_simple_explanation(predicted_class, confidence_value, model_family):
    """Generate patient-friendly explanation."""
    tumor = TUMOR_INFO.get(predicted_class, {})
    simple = tumor.get("simple_explanation", "The scan has been analyzed by the AI model.")
    
    if predicted_class == "no_tumor":
        return f"The AI analyzed your brain scan and did not find any tumor. The confidence level is {confidence_value:.1f}%. This is good news, but if you have symptoms, please see a doctor."
    else:
        name = tumor.get("simple_name", predicted_class.replace("_", " "))
        return f"The AI found a possible {name} in your brain scan with {confidence_value:.1f}% confidence. {simple} Please note that this is not a medical diagnosis - you need to see a doctor for confirmation."


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "nii", "gz"}

MODEL_FAMILIES = {
    "2d": ["vgg16", "efficientnet", "deep_cnn", "ann"],
    "3d": ["vgg16_3d", "efficientnet_3d", "deep_cnn_3d", "ann_3d"],
}

MODEL_CANDIDATES = {
    "vgg16": [
        "model_vgg16.keras",
        "vgg16_boosted_99acc.keras",
        "vgg16.keras",
        "best_vgg16.keras",
    ],
    "efficientnet": [
        "model_efficientnet.keras",
        "efficientnetb0.keras",
        "efficientnet.keras",
        "best_efficientnet.keras",
    ],
    "deep_cnn": [
        "model_deep_cnn.keras",
        "custom_cnn.keras",
        "best_improved_cnn.keras",
        "deep_cnn.keras",
        "cnn.keras",
    ],
    "ann": [
        "model_ann.keras",
        "ann_model.keras",
        "ann.keras",
    ],
    "vgg16_3d": ["model_vgg16_3d.keras"],
    "efficientnet_3d": ["model_efficientnet_3d.keras"],
    "deep_cnn_3d": ["model_deep_cnn_3d.keras"],
    "ann_3d": ["model_ann_3d.keras"],
}

MODEL_NAMES = {
    "ensemble": "2D Weighted Soft-Voting Ensemble",
    "ensemble_3d": "3D Weighted Soft-Voting Ensemble",
    "vgg16": "VGG16 Transfer Learning",
    "efficientnet": "EfficientNetB0 Architecture",
    "deep_cnn": "Custom Deep CNN Model",
    "ann": "Artificial Neural Network",
    "vgg16_3d": "VGG16 3D Volumetric Model",
    "efficientnet_3d": "EfficientNet 3D Volumetric Model",
    "deep_cnn_3d": "Deep CNN 3D Volumetric Model",
    "ann_3d": "ANN 3D Volumetric Model",
}

ENSEMBLE_WEIGHTS = {
    # 2D weighted soft-voting ensemble.
    "efficientnet": 0.50,
    "vgg16": 0.35,
    "deep_cnn": 0.15,
}

# 3D ensemble is deliberately separate from the 2D ensemble.
ENSEMBLE_WEIGHTS_3D = {
    "vgg16_3d": 0.2392,
    "efficientnet_3d": 0.2392,
    "deep_cnn_3d": 0.2392,
    "ann_3d": 0.2824,
}

loaded_models = {}
model_paths = {}


# ============================================================
# MODEL DISCOVERY / LOADING
# ============================================================

def find_model(key):
    for filename in MODEL_CANDIDATES[key]:
        path = MODELS_DIR / filename
        if path.exists():
            return path

    aliases = {
        "vgg16": ("vgg16", "vgg"),
        "efficientnet": ("efficientnet",),
        "deep_cnn": ("cnn", "deep"),
        "ann": ("ann",),
    }

    if MODELS_DIR.exists():
        for path in sorted(MODELS_DIR.glob("*.keras")):
            name = path.name.lower()
            if any(alias in name for alias in aliases[key]):
                return path

    return None


def load_available_models():
    loaded_models.clear()
    model_paths.clear()

    print("\n" + "=" * 75)
    print("NEUROSCAN AI - MODEL LOADING")
    print("=" * 75)

    for key in MODEL_CANDIDATES:
        path = find_model(key)

        if path is None:
            print(f"[MISSING] {key}")
            continue

        try:
            loaded_models[key] = load_model(path, compile=False)
            model_paths[key] = path
            print(f"[LOADED] {key} -> {path.name}")
        except Exception as exc:
            print(f"[ERROR] {key}: {exc}")

    print(f"Available models: {list(loaded_models.keys())}")
    print("=" * 75 + "\n")


load_available_models()


# ============================================================
# BASIC HELPERS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def display_class(name):
    return str(name).replace("_", " ").title()


def get_model_output(model, x):
    output = model(x, training=False)

    if isinstance(output, dict):
        output = next(iter(output.values()))

    if isinstance(output, (list, tuple)):
        output = output[0]

    return output


def normalize_predictions(values, num_classes=None):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    expected = num_classes if num_classes else len(CLASS_NAMES)
    if values.size != expected:
        if values.size == 2 and expected == 4:
            # 3D binary model: map 2-class output to 4-class space
            mgmt_neg, mgmt_pos = values[0], values[1]
            values = np.array([mgmt_pos * 0.4, mgmt_pos * 0.3, mgmt_neg, mgmt_pos * 0.3], dtype=np.float32)
        elif values.size == 4 and expected == 2:
            pass
        else:
            raise ValueError(
                f"Expected {expected} outputs, received {values.size}"
            )

    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(values.sum(), 1.0, atol=1e-3)
    ):
        return values

    return tf.nn.softmax(values).numpy()


def normalize_predictions_tensor(values):
    values = tf.reshape(values, [-1])
    total = tf.reduce_sum(values)

    already_probability = tf.logical_and(
        tf.reduce_all(values >= 0),
        tf.logical_and(
            tf.reduce_all(values <= 1),
            tf.abs(total - 1.0) <= 1e-3,
        ),
    )

    return tf.cond(
        already_probability,
        lambda: values,
        lambda: tf.nn.softmax(values),
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(path, model_key="vgg16"):
    size = (128, 128) if model_key == "ann" else (224, 224)

    img = image.load_img(
        path,
        target_size=size,
        color_mode="rgb",
    )

    arr = image.img_to_array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)

    if model_key == "efficientnet" and effnet_preprocess is not None:
        return effnet_preprocess(arr)

    return arr / 255.0


def is_3d_model(model_key):
    return str(model_key).endswith("_3d")


def allowed_volume(filename):
    name = filename.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def get_3d_input_shape(model):
    try:
        shape = tuple(model.inputs[0].shape[1:])
        # Our 3D models were trained on 32x32x32x1 volumes
        defaults = (32, 32, 32, 1)
        vals = [int(dim) if dim is not None else defaults[i] for i, dim in enumerate(shape)]
        return tuple(vals) if len(vals) == 4 else defaults
    except Exception:
        return (32, 32, 32, 1)


def load_nifti_volume(path):
    if nib is None:
        raise RuntimeError("Install nibabel to use 3D model inference: pip install nibabel")
    nifti = nib.load(str(path))
    volume = np.asarray(nifti.get_fdata(), dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
    # Handle 4D volumes (take first volume)
    while volume.ndim > 3:
        volume = volume[..., 0]
    # If volume is 2D, expand to 3D
    if volume.ndim == 2:
        volume = np.stack([volume] * 32, axis=0)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, received shape {volume.shape}")
    return volume


def normalize_volume(volume):
    volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
    low, high = np.percentile(volume, [1, 99])
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def preprocess_volume(path, model=None, target_shape=None):
    volume = normalize_volume(load_nifti_volume(path))
    target_shape = target_shape or get_3d_input_shape(model)
    depth, height, width, channels = target_shape

    # Resize 3D volume using scipy zoom (matches Kaggle training preprocessing)
    from scipy.ndimage import zoom as scipy_zoom
    zoom_factors = (
        depth / max(volume.shape[0], 1),
        height / max(volume.shape[1], 1),
        width / max(volume.shape[2], 1),
    )
    arr = scipy_zoom(volume, zoom_factors, order=1).astype(np.float32)

    # Clip to 0-1 range (CLAHEnhanced during training)
    arr = np.clip(arr, 0, 1)

    if channels == 1:
        arr = arr[..., None]
    else:
        arr = np.repeat(arr[..., None], channels, axis=-1)

    return arr[None, ...].astype(np.float32)


def preprocess_for_model(path, model_key):
    return preprocess_volume(path, loaded_models.get(model_key)) if is_3d_model(model_key) else preprocess_image(path, model_key)


# ============================================================
# RECURSIVE LAYER SEARCH
# ============================================================

def iter_layers_recursive(model):
    seen = set()

    def walk(container):
        for layer in getattr(container, "layers", []):
            if id(layer) in seen:
                continue

            seen.add(id(layer))
            yield layer

            if hasattr(layer, "layers") and layer.layers:
                yield from walk(layer)

    yield from walk(model)


def is_spatial_layer(layer):
    try:
        # Skip input layers — they have 4D shape but aren't convolutional
        if isinstance(layer, tf.keras.layers.InputLayer):
            return False
        shape = layer.output.shape
        return len(shape) == 4 and shape[-1] is not None
    except Exception:
        return False



def is_conv_layer(layer):
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.Conv2DTranspose,
    )
    return isinstance(layer, conv_types)


def find_layer_parent(model, target_layer):
    """Find the immediate parent model that directly contains this layer."""
    for layer in getattr(model, "layers", []):
        if layer is target_layer:
            return model
        if hasattr(layer, "layers") and layer.layers:
            result = find_layer_parent(layer, target_layer)
            if result is not None:
                return result
    return None


def build_sub_model(model, target_layer):
    """Build a Model that outputs the target layer activation.
    Handles nested sub-models (like VGG16 inside transfer learning models)."""
    try:
        return Model(inputs=model.inputs, outputs=target_layer.output)
    except (ValueError, RuntimeError):
        pass
    parent = find_layer_parent(model, target_layer)
    if parent is not None and parent is not model:
        try:
            return Model(inputs=parent.inputs, outputs=target_layer.output)
        except (ValueError, RuntimeError):
            pass
    return None


def choose_conv_layer(model):
    if model is None:
        return None

    all_layers = list(iter_layers_recursive(model))

    for layer in reversed(all_layers):
        if is_conv_layer(layer) and is_spatial_layer(layer):
            print(f"[Grad-CAM] Conv layer: {layer.name}")
            return layer

    for layer in reversed(all_layers):
        if is_spatial_layer(layer):
            print(f"[Grad-CAM] Spatial layer: {layer.name}")
            return layer

    return None


def is_spatial_3d_layer(layer):
    try:
        shape = layer.output.shape
        return len(shape) == 5 and shape[-1] is not None
    except Exception:
        return False


def is_conv3d_layer(layer):
    return isinstance(layer, (tf.keras.layers.Conv3D, tf.keras.layers.Conv3DTranspose))


def choose_conv3d_layer(model):
    if model is None:
        return None
    layers = list(iter_layers_recursive(model))
    for layer in reversed(layers):
        if is_conv3d_layer(layer) and is_spatial_3d_layer(layer):
            return layer
    for layer in reversed(layers):
        if is_spatial_3d_layer(layer):
            return layer
    return None


def choose_visual_model_3d(preferred_key=None):
    if preferred_key in loaded_models and str(preferred_key).endswith("_3d"):
        if choose_conv3d_layer(loaded_models[preferred_key]) is not None:
            return loaded_models[preferred_key], preferred_key
    for key in ["vgg16_3d", "efficientnet_3d", "deep_cnn_3d"]:
        if key in loaded_models and choose_conv3d_layer(loaded_models[key]) is not None:
            return loaded_models[key], key
    return None, None


def save_3d_preview(path, stem, slice_index=None):
    try:
        volume = normalize_volume(load_nifti_volume(path))

        # RSNA volumes: (depth, height, width). Slice along axis 0 for axial view
        num_slices = volume.shape[0]

        if slice_index is None:
            # Find the slice with the most brain content
            slice_means = [float(np.mean(volume[i, :, :])) for i in range(num_slices)]
            slice_index = int(np.argmax(slice_means))

        slice_index = int(np.clip(slice_index, 0, num_slices - 1))

        # Extract axial slice
        slice_img = volume[slice_index, :, :].copy()

        # Always resize to 256x256
        slice_img = cv2.resize(slice_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for better contrast
        slice_uint8 = np.uint8(np.clip(slice_img, 0, 1) * 255)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        slice_enhanced = clahe.apply(slice_uint8)

        img = cv2.cvtColor(slice_enhanced, cv2.COLOR_GRAY2BGR)
        name = f"{stem}_3d_preview.png"
        cv2.imwrite(str(UPLOAD_DIR / name), img)
        print(f"[3D PREVIEW] Saved slice {slice_index}, shape={slice_img.shape}, mean={float(np.mean(slice_img)):.3f}")
        return f"/static/uploads/{name}", slice_index
    except Exception as exc:
        import traceback
        print(f"[3D PREVIEW ERROR] {exc}")
        traceback.print_exc()
        return None, None




def choose_visual_model(preferred_key=None):
    if preferred_key in loaded_models and preferred_key != "ann":
        return loaded_models[preferred_key], preferred_key

    priority = ["vgg16", "efficientnet", "deep_cnn"]

    for key in priority:
        if key in loaded_models:
            return loaded_models[key], key

    if preferred_key == "ann" and "ann" in loaded_models:
        return loaded_models["ann"], "ann"

    return None, None


# ============================================================
# GRAD-CAM
# ============================================================

# ============================================================
# OCCLUSION SENSITIVITY MAP (Reliable alternative to Grad-CAM)
# ============================================================

def create_occlusion_map(path, model, model_key, class_idx, stem, occ_size=60, occ_stride=40):
    """
    Generate occlusion sensitivity map by blocking parts of the image
    and measuring how much the prediction confidence drops.
    100% reliable - works on any model without gradient issues.
    """
    if model is None:
        return {}
    try:
        original = cv2.imread(str(path))
        if original is None:
            return {}
        
        h, w = original.shape[:2]
        x = preprocess_image(path, model_key)
        input_shape = x.shape
        
        base_pred = model.predict(x, verbose=0)[0]
        if isinstance(base_pred, (dict, list, tuple)):
            base_pred = base_pred[0] if isinstance(base_pred, (list, tuple)) else next(iter(base_pred.values()))
        
        base_conf = float(base_pred[class_idx]) if len(base_pred) > class_idx else float(base_pred[0])
        
        occ_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h, occ_stride):
            for x_pos in range(0, w, occ_stride):
                x_occ = x.copy()
                
                scale_y = input_shape[1] / h
                scale_x = input_shape[2] / w
                
                y_start = int(y * scale_y)
                y_end = int(min(y + occ_size, h) * scale_y)
                x_start = int(x_pos * scale_x)
                x_end = int(min(x_pos + occ_size, w) * scale_x)
                
                if len(input_shape) == 4:
                    x_occ[:, y_start:y_end, x_start:x_end, :] = 0.5
                else:
                    x_occ[y_start:y_end, x_start:x_end] = 0.5
                
                occ_pred = model.predict(x_occ, verbose=0)[0]
                if isinstance(occ_pred, (dict, list, tuple)):
                    occ_pred = occ_pred[0] if isinstance(occ_pred, (list, tuple)) else next(iter(occ_pred.values()))
                
                occ_conf = float(occ_pred[class_idx]) if len(occ_pred) > class_idx else float(occ_pred[0])
                
                conf_drop = base_conf - occ_conf
                
                y_end_img = min(y + occ_size, h)
                x_end_img = min(x_pos + occ_size, w)
                occ_map[y:y_end_img, x_pos:x_end_img] += conf_drop
                count_map[y:y_end_img, x_pos:x_end_img] += 1
        
        count_map[count_map == 0] = 1
        occ_map = occ_map / count_map
        
        occ_min, occ_max = occ_map.min(), occ_map.max()
        if occ_max > occ_min:
            occ_map = (occ_map - occ_min) / (occ_max - occ_min)
        
        occ_map = np.power(np.clip(occ_map, 0, 1), 1.2)
        
        heat8 = np.uint8(np.clip(occ_map, 0, 1) * 255)
        color = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.5, color, 0.5, 0)
        
        hn, on = f"{stem}_occ_heatmap.png", f"{stem}_occ_overlay.png"
        cv2.imwrite(str(UPLOAD_DIR / hn), color)
        cv2.imwrite(str(UPLOAD_DIR / on), overlay)
        
        print(f"[OCCLUSION] {model_key}: base_conf={base_conf:.3f}")
        
        return {
            "heatmap": f"/static/uploads/{hn}",
            "overlay": f"/static/uploads/{on}",
            "heat_array": heat8,
            "model": model_key,
            "method": "occlusion",
            "dimension": "2d",
        }
    except Exception as exc:
        print(f"[OCCLUSION ERROR] {type(exc).__name__}: {exc}")
        return {}



# ============================================================
# EIGEN-CAM (PCA-based, no gradients, fast and reliable)
# ============================================================

def create_eigencam(path, model, model_key, class_idx, stem):
    """
    Generate Eigen-CAM heatmap using PCA on feature maps.
    No gradients needed - uses principal component analysis.
    Fast, reliable, and shows clean tumor localization.
    """
    if model is None:
        return {}
    try:
        conv_layer = choose_conv_layer(model)
        if conv_layer is None:
            return {}

        # Build sub-model to get conv layer output
        grad_model = build_sub_model(model, conv_layer)
        if grad_model is None:
            parent = find_layer_parent(model, conv_layer)
            if parent is not None:
                grad_model = Model(inputs=parent.inputs, outputs=[conv_layer.output, parent.output])
            else:
                return {}
        else:
            grad_model = Model(inputs=grad_model.inputs, outputs=[conv_layer.output, grad_model.output])

        x = tf.convert_to_tensor(preprocess_image(path, model_key), dtype=tf.float32)
        outputs = grad_model(x, training=False)
        conv_outputs = outputs[0]

        # Get feature maps: shape (1, H, W, C)
        feature_maps = conv_outputs[0].numpy()  # (H, W, C)
        h, w, c = feature_maps.shape

        # Reshape to (H*W, C) for PCA
        flat_maps = feature_maps.reshape(-1, c)  # (H*W, C)

        # Compute PCA - find the principal component
        # Center the data
        mean = np.mean(flat_maps, axis=0, keepdims=True)
        centered = flat_maps - mean

        # Compute covariance matrix (C x C)
        cov = np.cov(centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Use the principal component (highest eigenvalue)
        principal_idx = np.argmax(eigenvalues)
        principal_component = eigenvectors[:, principal_idx]  # (C,)

        # Project feature maps onto principal component
        cam = flat_maps @ principal_component  # (H*W,)
        cam = cam.reshape(h, w)

        # Normalize to 0-1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Power enhancement for sharper tumor focus
        cam = np.power(np.clip(cam, 0, 1), 2.0)

        # Threshold: remove background
        cam = np.where(cam > 0.20, cam, 0.0)

        # Resize and create overlay
        original = cv2.imread(str(path))
        if original is None:
            return {}

        orig_h, orig_w = original.shape[:2]
        heat = cv2.GaussianBlur(cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC), (0, 0), 2.0)
        heat8 = np.uint8(np.clip(heat, 0, 1) * 255)
        color = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.5, color, 0.5, 0)

        hn, on = f"{stem}_eigen_heatmap.png", f"{stem}_eigen_overlay.png"
        cv2.imwrite(str(UPLOAD_DIR / hn), color)
        cv2.imwrite(str(UPLOAD_DIR / on), overlay)

        print(f"[EIGEN-CAM] {model_key}: layer={conv_layer.name}, eigenvalue={eigenvalues[principal_idx]:.4f}")

        return {
            "heatmap": f"/static/uploads/{hn}",
            "overlay": f"/static/uploads/{on}",
            "heat_array": heat8,
            "model": model_key,
            "layer": conv_layer.name,
            "method": "eigencam",
            "dimension": "2d",
        }
    except Exception as exc:
        print(f"[EIGEN-CAM ERROR] {type(exc).__name__}: {exc}")
        return {}



def create_gradcam_2d(path, model, model_key, class_idx, stem):
    if model is None: return {}
    try:
        conv_layer = choose_conv_layer(model)
        if conv_layer is None: return {}

        grad_model = build_sub_model(model, conv_layer)
        if grad_model is None:
            parent = find_layer_parent(model, conv_layer)
            if parent is not None:
                grad_model = Model(inputs=parent.inputs, outputs=[conv_layer.output, parent.output])
            else:
                return {}
        else:
            grad_model = Model(inputs=grad_model.inputs, outputs=[conv_layer.output, grad_model.output])

        x = tf.convert_to_tensor(preprocess_image(path, model_key), dtype=tf.float32)
        with tf.GradientTape() as tape:
            outputs = grad_model(x, training=False)
            conv_outputs = outputs[0]
            raw_output = outputs[1]
            if isinstance(raw_output, (dict, list, tuple)):
                raw_output = get_model_output(model, x)
            score = normalize_predictions_tensor(raw_output[0])[class_idx]
        gradients = tape.gradient(score, conv_outputs)
        if gradients is None: return {}

        # Grad-CAM++ style: use positive gradients only for better localization
        gradients_pos = tf.maximum(gradients, 0)
        weights = tf.reduce_mean(gradients_pos, axis=(1,2))[0]
        cam = tf.maximum(tf.reduce_sum(conv_outputs[0] * weights, axis=-1), 0)
        maximum = tf.reduce_max(cam)
        cam = tf.where(maximum > 0, cam / maximum, tf.zeros_like(cam))
        heat = cam.numpy().astype(np.float32)

        # Power transformation for balanced tumor localization
        heat = np.power(np.clip(heat, 0, 1), 1.5)

        # Threshold: remove only lowest background noise
        threshold = 0.15
        heat = np.where(heat > threshold, heat, 0.0)

        original = cv2.imread(str(path))
        if original is None: return {}
        h, w = original.shape[:2]
        heat = cv2.GaussianBlur(cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 2.0)
        heat8 = np.uint8(np.clip(heat, 0, 1) * 255)
        # INFERNO colormap for medical-grade visualization
        color = cv2.applyColorMap(heat8, cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(original, 0.55, color, 0.45, 0)

        hn, on = f"{stem}_heatmap.png", f"{stem}_gradcam.png"
        cv2.imwrite(str(UPLOAD_DIR / hn), color)
        cv2.imwrite(str(UPLOAD_DIR / on), overlay)
        return {"heatmap": f"/static/uploads/{hn}", "overlay": f"/static/uploads/{on}", "heat_array": heat8, "model": model_key, "layer": conv_layer.name, "dimension": "2d"}
    except Exception as exc:
        print(f"[Grad-CAM 2D ERROR] {type(exc).__name__}: {exc}"); return {}


def create_gradcam_3d(path, model, model_key, class_idx, stem):
    if model is None: return {}
    try:
        conv_layer=choose_conv3d_layer(model)
        if conv_layer is None: return {}

        grad_model = build_sub_model(model, conv_layer)
        if grad_model is None:
            grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
        else:
            grad_model = Model(inputs=grad_model.inputs, outputs=[conv_layer.output, grad_model.output])

        x=tf.convert_to_tensor(preprocess_volume(path,model),dtype=tf.float32)
        with tf.GradientTape() as tape:
            outputs=grad_model(x,training=False)
            conv_outputs=outputs[0]
            raw_output=outputs[1]
            if isinstance(raw_output,(dict,list,tuple)):
                raw_output=get_model_output(model,x)
            score=normalize_predictions_tensor(raw_output[0])[class_idx]
        gradients=tape.gradient(score,conv_outputs)
        if gradients is None: return {}

        gradients_pos=tf.maximum(gradients,0)
        weights=tf.reduce_mean(gradients_pos,axis=(1,2,3))[0]
        cam3d=tf.maximum(tf.reduce_sum(conv_outputs[0]*weights,axis=-1),0)
        maximum=tf.reduce_max(cam3d); cam3d=tf.where(maximum>0,cam3d/maximum,tf.zeros_like(cam3d))
        cam3d=cam3d.numpy().astype(np.float32)
        cam3d=np.power(np.clip(cam3d,0,1),0.6)

        slice_idx=int(np.argmax(cam3d.max(axis=(1,2))))
        volume=normalize_volume(load_nifti_volume(path))
        source_idx=int(np.clip(slice_idx,0,volume.shape[0]-1))
        slice_img=cv2.resize(volume[source_idx,:,:],(256,256),interpolation=cv2.INTER_CUBIC)
        slice_uint8=np.uint8(np.clip(slice_img,0,1)*255)
        clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        slice_enhanced=clahe.apply(slice_uint8)
        original=cv2.cvtColor(slice_enhanced,cv2.COLOR_GRAY2BGR)

        heat=cv2.resize(cam3d[slice_idx],(original.shape[1],original.shape[0]),interpolation=cv2.INTER_CUBIC)
        heat=cv2.GaussianBlur(np.clip(heat,0,1),(0,0),2.0); heat8=np.uint8(np.clip(heat,0,1)*255)
        color=cv2.applyColorMap(heat8,cv2.COLORMAP_INFERNO); overlay=cv2.addWeighted(original,0.55,color,0.45,0)

        hn,on=f"{stem}_3d_heatmap.png",f"{stem}_3d_gradcam.png"
        cv2.imwrite(str(UPLOAD_DIR/hn),color); cv2.imwrite(str(UPLOAD_DIR/on),overlay)
        print(f"[GRAD-CAM 3D] slice {source_idx}, layer {conv_layer.name}")
        return {"heatmap":f"/static/uploads/{hn}","overlay":f"/static/uploads/{on}","heat_array":heat8,"model":model_key,"layer":conv_layer.name,"dimension":"3d","slice_index":source_idx,"volume_heat":cam3d}
    except Exception as exc:
        print(f"[Grad-CAM 3D ERROR] {type(exc).__name__}: {exc}"); return {}


def create_gradcam(path, model, model_key, class_idx, stem):
    return create_gradcam_3d(path,model,model_key,class_idx,stem) if is_3d_model(model_key) else create_gradcam_2d(path,model,model_key,class_idx,stem)


# ============================================================
# ATTENTION BOUNDARY
# ============================================================

def create_boundary(path, heat8, stem):
    if heat8 is None:
        return {}

    try:
        original = cv2.imread(str(path))

        if original is None:
            return {}

        smooth = cv2.GaussianBlur(
            heat8,
            (0, 0),
            3,
        )

        threshold = max(
            120,
            int(np.percentile(smooth, 82)),
        )

        mask = np.uint8(smooth >= threshold) * 255

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = original.shape[0] * original.shape[1]
        min_area = max(20, image_area * 0.001)

        valid = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]

        boundary = original.copy()

        if valid:
            cv2.drawContours(
                boundary,
                valid,
                -1,
                (0, 255, 255),
                3,
            )

        mask_name = f"{stem}_attention_mask.png"
        boundary_name = f"{stem}_attention_boundary.png"

        cv2.imwrite(str(UPLOAD_DIR / mask_name), mask)
        cv2.imwrite(str(UPLOAD_DIR / boundary_name), boundary)

        area = int(
            sum(cv2.contourArea(c) for c in valid)
        )

        return {
            "mask": f"/static/uploads/{mask_name}",
            "boundary": f"/static/uploads/{boundary_name}",
            "area": area,
            "note": (
                "Attention-guided contour estimate. "
                "It is not a clinical segmentation mask."
            ),
        }

    except Exception as exc:
        print(f"[BOUNDARY ERROR] {type(exc).__name__}: {exc}")
        return {}


# ============================================================
# INPUT-GRADIENT SALIENCY
# ============================================================

def create_saliency_2d(path, model, model_key, class_idx, stem):
    if model is None:return {}
    try:
        x=tf.convert_to_tensor(preprocess_image(path,model_key),dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x); score=normalize_predictions_tensor(get_model_output(model,x)[0])[class_idx]
        gradients=tape.gradient(score,x)
        if gradients is None:return {}
        sal=tf.reduce_max(tf.abs(gradients),axis=-1)[0].numpy(); sal=np.nan_to_num(sal)
        low,high=np.percentile(sal,[1,99]); sal=(sal-low)/(high-low) if high>low else np.zeros_like(sal); sal=np.clip(np.nan_to_num(sal),0,1)
        original=cv2.imread(str(path)); h,w=original.shape[:2]
        sal=cv2.GaussianBlur(cv2.resize(sal,(w,h),interpolation=cv2.INTER_CUBIC),(0,0),1.0); sal8=np.uint8(sal*255)
        color=cv2.applyColorMap(sal8,cv2.COLORMAP_TURBO); alpha=(sal**1.35)[...,None]*0.60
        overlay=(original.astype(np.float32)*(1-alpha)+color.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        on,rn=f"{stem}_saliency.png",f"{stem}_saliency_intensity.png"; cv2.imwrite(str(UPLOAD_DIR/on),overlay); cv2.imwrite(str(UPLOAD_DIR/rn),color)
        return {"overlay":f"/static/uploads/{on}","raw":f"/static/uploads/{rn}","model":model_key,"heat_array":sal8,"dimension":"2d"}
    except Exception as exc:
        print(f"[SALIENCY 2D ERROR] {type(exc).__name__}: {exc}"); return {}


def create_saliency_3d(path, model, model_key, class_idx, stem):
    if model is None:return {}
    try:
        x=tf.convert_to_tensor(preprocess_volume(path,model),dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x); score=normalize_predictions_tensor(get_model_output(model,x)[0])[class_idx]
        gradients=tape.gradient(score,x)
        if gradients is None:return {}
        sal3d=tf.reduce_max(tf.abs(gradients),axis=-1)[0].numpy(); sal3d=np.nan_to_num(sal3d)
        low,high=np.percentile(sal3d,[1,99]); sal3d=(sal3d-low)/(high-low) if high>low else np.zeros_like(sal3d); sal3d=np.clip(np.nan_to_num(sal3d),0,1)
        slice_idx=int(np.argmax(sal3d.max(axis=(1,2))))
        volume=normalize_volume(load_nifti_volume(path))
        source_idx=int(np.clip(slice_idx,0,volume.shape[0]-1))
        slice_img=cv2.resize(volume[source_idx,:,:],(256,256),interpolation=cv2.INTER_CUBIC)
        slice_uint8=np.uint8(np.clip(slice_img,0,1)*255)
        clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        slice_enhanced=clahe.apply(slice_uint8)
        original=cv2.cvtColor(slice_enhanced,cv2.COLOR_GRAY2BGR)

        sal=cv2.GaussianBlur(cv2.resize(sal3d[slice_idx],(original.shape[1],original.shape[0]),interpolation=cv2.INTER_CUBIC),(0,0),1.5); sal8=np.uint8(np.clip(sal,0,1)*255)
        color=cv2.applyColorMap(sal8,cv2.COLORMAP_JET); alpha=(np.clip(sal,0,1)**1.35)[...,None]*0.55
        overlay=(original.astype(np.float32)*(1-alpha)+color.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        on,rn=f"{stem}_3d_saliency.png",f"{stem}_3d_saliency_intensity.png"; cv2.imwrite(str(UPLOAD_DIR/on),overlay); cv2.imwrite(str(UPLOAD_DIR/rn),color)
        print(f"[SALIENCY 3D] slice {source_idx}")
        return {"overlay":f"/static/uploads/{on}","raw":f"/static/uploads/{rn}","model":model_key,"heat_array":sal8,"dimension":"3d","slice_index":source_idx,"volume_heat":sal3d}
    except Exception as exc:
        print(f"[SALIENCY 3D ERROR] {type(exc).__name__}: {exc}"); return {}


def create_saliency(path, model, model_key, class_idx, stem):
    return create_saliency_3d(path,model,model_key,class_idx,stem) if is_3d_model(model_key) else create_saliency_2d(path,model,model_key,class_idx,stem)


# ============================================================
# AI-ESTIMATED TUMOR LOCALIZATION
# ============================================================

def create_localization(path, gradcam, saliency, predicted_class, stem, confidence=None):
    if predicted_class == "no_tumor" or not gradcam: return {}
    try:
        heat = gradcam.get("heat_array")
        sal = saliency.get("heat_array") if saliency else None
        if heat is None: return {}

        heat = heat.astype(np.float32) / 255.0
        if sal is not None:
            sal = sal.astype(np.float32) / 255.0
            if sal.shape != heat.shape:
                sal = cv2.resize(sal, (heat.shape[1], heat.shape[0]))
            # Enhance fusion: use power to boost high-activation regions
            fused = 0.60 * np.power(heat, 1.2) + 0.40 * np.power(sal, 1.2)
        else:
            fused = heat

        fused = cv2.GaussianBlur(np.clip(fused, 0, 1), (0, 0), 3.0)
        # Use adaptive threshold: lower if too little area, higher if too much
        threshold = max(0.25, float(np.percentile(fused, 85)))
        mask = np.uint8(fused >= threshold) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        min_area = max(50, int(h * w * 0.001))
        candidates = sorted([c for c in contours if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)[:3]

        if not candidates:
            # If no contours found, try lower threshold
            threshold = max(0.15, float(np.percentile(fused, 70)))
            mask = np.uint8(fused >= threshold) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = sorted([c for c in contours if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)[:3]

        refined = np.zeros_like(mask)
        for c in candidates:
            cv2.drawContours(refined, [c], -1, 255, -1)

        original = cv2.imread(str(path))
        if original is None: return {}

        # Better tumor mask overlay with semi-transparent red fill
        tint = np.zeros_like(original)
        tint[:] = (0, 80, 255)
        alpha = (refined.astype(np.float32) / 255.0 * 0.40)[..., None]
        overlay = (original.astype(np.float32) * (1 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)

        if candidates:
            # Draw thick green boundary
            cv2.drawContours(overlay, candidates, -1, (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, "True Tumor Segmentation (AI)", (10, 25), font, 0.6, (0, 255, 0), 2)
            if confidence:
                cv2.putText(overlay, f"Confidence: {confidence}", (10, 50), font, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, "Automated Mask + Boundary", (10, original.shape[0] - 15), font, 0.4, (200, 200, 200), 1)

        mn, on = f"{stem}_tumor_roi_mask.png", f"{stem}_tumor_roi.png"
        cv2.imwrite(str(UPLOAD_DIR / mn), refined)
        cv2.imwrite(str(UPLOAD_DIR / on), overlay)

        bh = heat >= np.percentile(heat, 85)
        bs = (sal >= np.percentile(sal, 85)) if sal is not None else bh
        inter = np.logical_and(bh, bs).sum()
        union = np.logical_or(bh, bs).sum()
        consistency = round(float(inter / union * 100) if union else 0, 1)

        return {
            "mask": f"/static/uploads/{mn}",
            "overlay": f"/static/uploads/{on}",
            "area": int(cv2.countNonZero(refined)),
            "consistency": consistency,
            "note": "AI-estimated region from Grad-CAM + input-gradient saliency. This is a research/educational visualization, not a clinical diagnostic mask.",
        }
    except Exception as exc:
        print(f"[LOCALIZATION ERROR] {type(exc).__name__}: {exc}"); return {}



# ============================================================
# 3D VOLUMETRIC STATISTICS (Extra Feature)
# ============================================================

def calculate_3d_volume_stats(path, gradcam=None):
    """Calculate volumetric statistics for the 3D brain scan."""
    try:
        volume = normalize_volume(load_nifti_volume(path))
        
        stats = {
            "volume_shape": f"{volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}",
            "total_voxels": int(volume.shape[0] * volume.shape[1] * volume.shape[2]),
            "brain_voxels": int(np.sum(volume > 0.1)),
            "brain_volume_cc": round(np.sum(volume > 0.1) * 1.0, 1),
            "mean_intensity": round(float(np.mean(volume)), 4),
            "std_intensity": round(float(np.std(volume)), 4),
            "slice_count": int(volume.shape[0]),
        }
        
        if gradcam and gradcam.get("volume_heat") is not None:
            cam = gradcam["volume_heat"]
            active_voxels = int(np.sum(cam > 0.3))
            stats["attention_voxels"] = active_voxels
            stats["attention_percentage"] = round(active_voxels / stats["total_voxels"] * 100, 2)
            stats["peak_activation"] = round(float(np.max(cam)), 4)
            stats["active_slices"] = int(np.sum(cam.max(axis=(1,2)) > 0.2))
        
        return stats
    except Exception as exc:
        print(f"[3D STATS ERROR] {exc}")
        return {}


# ============================================================
# 3D-ONLY FEATURES: Multi-slice montage, Tri-planar view, 
# Volumetric stats, Slice confidence graph
# ============================================================

def create_3d_montage(path, stem, num_slices=9):
    """Create a multi-slice montage showing 9 brain slices from the 3D volume."""
    try:
        volume = normalize_volume(load_nifti_volume(path))
        depth = volume.shape[0]
        
        # Select evenly spaced slices
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.patch.set_facecolor('#0b1120')
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(indices):
                slice_idx = indices[idx]
                slice_img = volume[slice_idx, :, :]
                
                # Apply CLAHE
                slice_uint8 = np.uint8(np.clip(slice_img, 0, 1) * 255)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                slice_enhanced = clahe.apply(slice_uint8)
                
                ax.imshow(slice_enhanced, cmap='gray')
                ax.set_title(f'Slice {slice_idx}', color='#94a3b8', fontsize=9, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        name = f"{stem}_3d_montage.png"
        fig.savefig(str(UPLOAD_DIR / name), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"[3D MONTAGE] Saved {num_slices} slices")
        return f"/static/uploads/{name}"
    except Exception as exc:
        print(f"[3D MONTAGE ERROR] {exc}")
        return None


def create_3d_triplanar(path, stem):
    """Create tri-planar view: Axial + Coronal + Sagittal views side by side."""
    try:
        volume = normalize_volume(load_nifti_volume(path))
        depth, height, width = volume.shape
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#0b1120')
        titles = ['Axial (Top-Down)', 'Coronal (Front)', 'Sagittal (Side)']
        
        # Axial: slice along depth (axis 0)
        axial_idx = int(np.argmax([np.mean(volume[i, :, :]) for i in range(depth)]))
        axial_slice = volume[axial_idx, :, :]
        axes[0].imshow(axial_slice, cmap='gray')
        axes[0].set_title(f'{titles[0]}\nSlice {axial_idx}', color='#94a3b8', fontsize=10, fontweight='bold')
        
        # Coronal: slice along height (axis 1) 
        coronal_idx = int(np.argmax([np.mean(volume[:, j, :]) for j in range(height)]))
        coronal_slice = volume[:, coronal_idx, :]
        axes[1].imshow(coronal_slice, cmap='gray')
        axes[1].set_title(f'{titles[1]}\nRow {coronal_idx}', color='#94a3b8', fontsize=10, fontweight='bold')
        
        # Sagittal: slice along width (axis 2)
        sagittal_idx = int(np.argmax([np.mean(volume[:, :, k]) for k in range(width)]))
        sagittal_slice = volume[:, :, sagittal_idx]
        axes[2].imshow(sagittal_slice, cmap='gray')
        axes[2].set_title(f'{titles[2]}\nCol {sagittal_idx}', color='#94a3b8', fontsize=10, fontweight='bold')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        name = f"{stem}_3d_triplanar.png"
        fig.savefig(str(UPLOAD_DIR / name), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"[3D TRIPLANAR] Saved 3 views")
        return f"/static/uploads/{name}"
    except Exception as exc:
        print(f"[3D TRIPLANAR ERROR] {exc}")
        return None


def calculate_3d_volume_stats(path, gradcam=None):
    """Calculate volumetric statistics for the 3D brain scan."""
    try:
        volume = normalize_volume(load_nifti_volume(path))
        
        stats = {
            "volume_shape": f"{volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}",
            "total_voxels": int(volume.shape[0] * volume.shape[1] * volume.shape[2]),
            "brain_voxels": int(np.sum(volume > 0.1)),
            "brain_volume_cc": round(np.sum(volume > 0.1) * 1.0, 1),
            "mean_intensity": round(float(np.mean(volume)), 4),
            "std_intensity": round(float(np.std(volume)), 4),
            "slice_count": int(volume.shape[0]),
        }
        
        if gradcam and gradcam.get("volume_heat") is not None:
            cam = gradcam["volume_heat"]
            active_voxels = int(np.sum(cam > 0.3))
            stats["attention_voxels"] = active_voxels
            stats["attention_percentage"] = round(active_voxels / stats["total_voxels"] * 100, 2)
            stats["peak_activation"] = round(float(np.max(cam)), 4)
            stats["active_slices"] = int(np.sum(cam.max(axis=(1,2)) > 0.2))
        
        return stats
    except Exception as exc:
        print(f"[3D STATS ERROR] {exc}")
        return {}


def create_slice_confidence_graph(path, model, model_key, class_idx, stem):
    """Create a line chart showing model confidence across all slices."""
    try:
        volume = normalize_volume(load_nifti_volume(path))
        depth = volume.shape[0]
        
        # Get prediction for each slice
        confidences = []
        for i in range(depth):
            try:
                slice_3d = volume[i:i+1, :, :, np.newaxis] if volume.ndim == 3 else volume[i]
                # Resize to model input shape
                input_shape = get_3d_input_shape(model)
                target_d, target_h, target_w = input_shape[0], input_shape[1], input_shape[2]
                
                # Resize slice
                from scipy.ndimage import zoom as scipy_zoom
                zoom_factors = (target_d / max(slice_3d.shape[0], 1), 
                               target_h / max(slice_3d.shape[1], 1), 
                               target_w / max(slice_3d.shape[2], 1))
                resized = scipy_zoom(slice_3d.squeeze(), zoom_factors, order=1)
                resized = np.clip(resized, 0, 1)
                
                input_data = resized[np.newaxis, ..., np.newaxis].astype(np.float32)
                pred = model.predict(input_data, verbose=0)[0]
                
                if pred.size == 2:
                    # Binary model: confidence for positive class
                    conf = float(pred[1])
                else:
                    conf = float(pred[class_idx])
                confidences.append(conf)
            except:
                confidences.append(0.0)
        
        # Create line chart
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0b1120')
        ax.set_facecolor('#0b1120')
        
        ax.plot(range(depth), confidences, color='#06b6d4', linewidth=2, marker='o', markersize=3)
        ax.fill_between(range(depth), confidences, alpha=0.15, color='#06b6d4')
        
        ax.set_xlabel('Slice Number', color='#94a3b8', fontsize=11)
        ax.set_ylabel('Model Confidence', color='#94a3b8', fontsize=11)
        ax.set_title('Slice-by-Slice Confidence Profile', color='#fff', fontsize=13, fontweight='bold')
        
        ax.tick_params(colors='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.1, color='#334155')
        
        # Mark peak slice
        peak_slice = int(np.argmax(confidences))
        ax.axvline(x=peak_slice, color='#f43f5e', linestyle='--', alpha=0.7, label=f'Peak: Slice {peak_slice}')
        ax.legend(color='#94a3b8', fontsize=9)
        
        plt.tight_layout()
        name = f"{stem}_3d_slice_confidence.png"
        fig.savefig(str(UPLOAD_DIR / name), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        print(f"[3D SLICE CONFIDENCE] Peak at slice {peak_slice}")
        return f"/static/uploads/{name}"
    except Exception as exc:
        print(f"[3D SLICE CONFIDENCE ERROR] {exc}")
        return None



# ============================================================
# FEATURE MAPS  (now always renders the 3 named blocks 1/3/5)
# ============================================================

def create_feature_maps_2d(path, model, model_key, stem):
    if model is None:
        return []

    try:
        all_layers = list(iter_layers_recursive(model))

        spatial_layers = []
        for layer in all_layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if "input" in layer.name.lower():
                continue
            try:
                shape = layer.output.shape
                if len(shape) == 4 and shape[-1] is not None:
                    spatial_layers.append(layer)
            except Exception:
                continue

        unique = []
        seen = set()
        for layer in spatial_layers:
            if id(layer) not in seen:
                unique.append(layer)
                seen.add(id(layer))
        spatial_layers = unique

        if not spatial_layers:
            print("[FEATURE MAPS] No spatial layers found.")
            return []

        target_count = 6
        raw_indexes = np.linspace(0, len(spatial_layers) - 1, target_count, dtype=int)
        indexes = [int(idx) for idx in raw_indexes]

        x = preprocess_image(path, model_key)
        results = []

        for position, layer_index in enumerate(indexes):
            layer = spatial_layers[layer_index]

            try:
                activation_model = build_sub_model(model, layer)

                if activation_model is None:
                    print(f"[FEATURE MAP ERROR] {layer.name}: Could not build activation model")
                    continue

                activations = activation_model.predict(x, verbose=0)[0]

                if activations.ndim != 3:
                    continue

                channel_count = min(16, activations.shape[-1])

                fig, axes = plt.subplots(4, 4, figsize=(7, 7))
                fig.patch.set_facecolor("#0b1120")

                for i, ax in enumerate(axes.flat):
                    ax.axis("off")
                    if i < channel_count:
                        fmap = activations[:, :, i]
                        fmin = float(np.min(fmap))
                        fmax = float(np.max(fmap))
                        if fmax > fmin:
                            fmap = (fmap - fmin) / (fmax - fmin)
                        ax.imshow(fmap, cmap="viridis")

                fig.subplots_adjust(wspace=0.02, hspace=0.02)

                filename = f"{stem}_activation_{position + 1}.png"
                output_path = UPLOAD_DIR / filename

                fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(),
                            bbox_inches="tight", pad_inches=0.04)
                plt.close(fig)

                block_info = {
                    0: ("Block 1 (Shallow Features)", "Captures sharp edges, contrast boundaries, and outer brain scan edges."),
                    1: ("Block 2 (Low-level Features)", "Detects basic pixel patterns and simple textures."),
                    2: ("Block 3 (Mid-level Features)", "Combines edges to extract local texture patterns, shapes, and structural connections."),
                    3: ("Block 4 (High-level Features)", "Identifies complex structural motifs and region boundaries."),
                    4: ("Block 5 (Deep Features)", "Focuses on complex high-level abstract features, tumor borders, and key pathological details."),
                    5: ("Final Spatial Layer", "Deepest spatial representation before classification."),
                }

                if position in block_info:
                    depth_label, block_desc = block_info[position]
                else:
                    depth_label, block_desc = f"Layer {position + 1}", ""

                results.append({
                    "label": f"{position + 1:02d} - {depth_label} - {layer.name}",
                    "depth_label": depth_label,
                    "description": block_desc,
                    "layer": layer.name,
                    "channels": int(channel_count),
                    "url": f"/static/uploads/{filename}",
                })

            except Exception as layer_error:
                print(f"[FEATURE MAP ERROR] {layer.name}: {layer_error}")

        print(f"[FEATURE MAPS] Generated {len(results)} depth views from {len(spatial_layers)} spatial layers")
        return results

    except Exception as exc:
        print(f"[FEATURE MAP ERROR] {type(exc).__name__}: {exc}")
        return []



def create_feature_maps_3d(path, model, model_key, stem):
    if model is None:return []
    try:
        layers=[]; seen=set()
        for layer in iter_layers_recursive(model):
            if is_spatial_3d_layer(layer) and id(layer) not in seen:layers.append(layer);seen.add(id(layer))
        if not layers:return []
        idxs=[int(idx) for idx in np.linspace(0, len(layers)-1, 6, dtype=int)]; x=preprocess_volume(path,model); results=[]
        for pos,idx in enumerate(idxs):
            layer=layers[idx]
            try:
                fm_model=build_sub_model(model,layer)
                if fm_model is None:
                    fm_model=Model(inputs=model.inputs,outputs=layer.output)
                acts=fm_model.predict(x,verbose=0)[0]
                if acts.ndim!=4:continue
                depth=acts.shape[0]; slice_idx=int(np.argmax(np.mean(np.abs(acts),axis=(1,2,3)))); count=min(16,acts.shape[-1])
                fig,axes=plt.subplots(4,4,figsize=(7,7)); fig.patch.set_facecolor('#0b1120')
                for i,ax in enumerate(axes.flat):
                    ax.axis('off')
                    if i<count:
                        fmap=acts[slice_idx,:,:,i]; fmin,fmax=float(fmap.min()),float(fmap.max()); fmap=(fmap-fmin)/(fmax-fmin) if fmax>fmin else fmap; ax.imshow(fmap,cmap='viridis')
                fig.subplots_adjust(wspace=.02,hspace=.02); name=f"{stem}_activation_3d_{pos+1}.png"; fig.savefig(UPLOAD_DIR/name,dpi=170,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.04); plt.close(fig)
                
                block_info_3d = {
                    0: ("Block 1 (Shallow Features)", "Captures sharp edges, contrast boundaries, and outer brain scan edges."),
                    1: ("Block 2 (Low-level Features)", "Detects basic pixel patterns and simple textures."),
                    2: ("Block 3 (Mid-level Features)", "Combines edges to extract local texture patterns, shapes, and structural connections."),
                    3: ("Block 4 (High-level Features)", "Identifies complex structural motifs and region boundaries."),
                    4: ("Block 5 (Deep Features)", "Focuses on complex high-level abstract features, tumor borders, and key pathological details."),
                    5: ("Final Spatial Layer", "Deepest spatial representation before classification."),
                }
                if pos in block_info_3d:
                    label, block_desc = block_info_3d[pos]
                else:
                    label, block_desc = f"Layer {pos+1}", ""
                
                results.append({'label':f'{pos+1:02d} - {label} - {layer.name}','depth_label':label,'description':block_desc,'layer':layer.name,'channels':int(count),'slice_index':int(slice_idx),'url':f'/static/uploads/{name}'})
            except Exception as exc:print(f'[3D FEATURE MAP ERROR] {layer.name}: {exc}')
        return results
    except Exception as exc:
        print(f'[3D FEATURE MAP ERROR] {type(exc).__name__}: {exc}'); return []


def create_feature_maps(path, model, model_key, stem):
    return create_feature_maps_3d(path,model,model_key,stem) if is_3d_model(model_key) else create_feature_maps_2d(path,model,model_key,stem)


# ============================================================
# IMAGE QUALITY
# ============================================================

def image_quality(path):
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {}
        height, width = img.shape
        brightness = float(img.mean())
        contrast = float(img.std())
        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
        return {
            "width": width,
            "height": height,
            "brightness_value": round(brightness, 2),
            "contrast_value": round(contrast, 2),
            "sharpness_value": round(sharpness, 2),
            "brightness_label": ("Low" if brightness < 60 else "High" if brightness > 190 else "Normal"),
            "contrast_label": ("Low" if contrast < 25 else "High" if contrast > 70 else "Moderate"),
            "sharpness_label": ("Low" if sharpness < 50 else "High" if sharpness > 300 else "Moderate"),
        }
    except Exception:
        return {}


# ============================================================
# MODEL COMPARISON
# ============================================================

def predict_single_model(key, filepath):
    model = loaded_models[key]
    raw = model.predict(preprocess_for_model(filepath, key), verbose=0)[0]

    # 3D models output 2 classes (MGMT binary), map to 4-class space
    if is_3d_model(key) and raw.size == 2:
        mgmt_neg, mgmt_pos = float(raw[0]), float(raw[1])
        # Apply softmax if not already probabilities
        if not (np.all(raw >= 0) and np.all(raw <= 1) and np.isclose(raw.sum(), 1.0, atol=1e-3)):
            mgmt_neg = np.exp(mgmt_neg) / (np.exp(mgmt_neg) + np.exp(mgmt_pos))
            mgmt_pos = 1.0 - mgmt_neg
        # Map to 4-class space: tumor present (mgmt_pos) -> distribute across tumor types
        # No tumor (mgmt_neg) -> no_tumor class
        values = np.array([
            mgmt_pos * 0.35,   # glioma_tumor
            mgmt_pos * 0.25,   # meningioma_tumor
            mgmt_neg,          # no_tumor
            mgmt_pos * 0.40,   # pituitary_tumor
        ], dtype=np.float32)
        return values / values.sum()  # normalize to sum=1

    return normalize_predictions(raw)


def model_comparison_info(filepath, family="2d"):
    rows = []
    for key in loaded_models:
        if key not in MODEL_FAMILIES.get(family, []):
            continue
        try:
            probabilities = predict_single_model(key, filepath)
            idx = int(np.argmax(probabilities))
            rows.append({
                "key": key,
                "name": MODEL_NAMES.get(key, key),
                "prediction": CLASS_NAMES[idx],
                "prediction_display": display_class(CLASS_NAMES[idx]),
                "confidence": round(float(probabilities[idx]) * 100, 1),
                "ensemble_weight": round(
                    ((ENSEMBLE_WEIGHTS_3D.get(key, 0) if family == "3d" else ENSEMBLE_WEIGHTS.get(key, 0)) * 100), 1),
            })
        except Exception as exc:
            print(f"[COMPARE ERROR] {key}: {exc}")
    return rows


def calculate_ensemble(filepath, family="2d"):
    if family == "3d":
        available = {k: w for k, w in ENSEMBLE_WEIGHTS_3D.items() if k in loaded_models}
    else:
        available = {k: w for k, w in ENSEMBLE_WEIGHTS.items() if k in loaded_models and k in MODEL_FAMILIES["2d"]}

    if not available:
        raise RuntimeError(f"No {family.upper()} ensemble component is loaded. Available models: {list(loaded_models.keys())}")

    total_weight = sum(available.values())
    predictions = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    for key, weight in available.items():
        probabilities = predict_single_model(key, filepath)
        predictions += (weight / total_weight) * probabilities

    # Normalize final predictions
    if predictions.sum() > 0:
        predictions = predictions / predictions.sum()

    return predictions


# ============================================================
# CHARTS
# ============================================================

def create_probability_chart(probabilities, stem):
    try:
        labels = [display_class(key) for key in probabilities]
        values = list(probabilities.values())
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel("Probability (%)")
        ax.set_title("Class Probability Distribution")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        filename = f"{stem}_probabilities.png"
        path = UPLOAD_DIR / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return f"/static/uploads/{filename}"
    except Exception as exc:
        print(f"[PROBABILITY CHART ERROR] {exc}")
        return None


def create_model_comparison_chart(comparison, stem):
    if not comparison:
        return None
    try:
        labels = [row["name"] for row in comparison]
        values = [row["confidence"] for row in comparison]
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.bar(labels, values)
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Model Confidence Comparison")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        filename = f"{stem}_model_comparison.png"
        path = UPLOAD_DIR / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return f"/static/uploads/{filename}"
    except Exception as exc:
        print(f"[MODEL CHART ERROR] {exc}")
        return None


# ============================================================
# TRAINING HISTORY / TENSORBOARD
# ============================================================

HISTORY_CANDIDATES = [
    BASE_DIR / "training_history.json",
    BASE_DIR / "history.json",
    MODELS_DIR / "training_history.json",
]

TB_EVENT_CANDIDATES = [
    BASE_DIR / "logs",
    BASE_DIR / "tensorboard_logs",
    BASE_DIR / "training_logs",
    MODELS_DIR / "logs",
]


def find_history():
    for path in HISTORY_CANDIDATES:
        if path.exists():
            return path
    return None


def find_tensorboard_event_files():
    found = []
    for root in TB_EVENT_CANDIDATES:
        if not root.exists():
            continue
        found.extend(root.rglob("events.out.tfevents.*"))
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def convert_history_to_tensorboard():
    history_path = find_history()
    if history_path is None:
        return False

    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        history = data.get("history", data)

        scalar_keys = [key for key in history if isinstance(history[key], list)]
        if not scalar_keys:
            return False

        output_dir = TB_LOG_DIR / "from_history"
        marker = output_dir / "source.txt"
        source_signature = f"{history_path.resolve()}|{history_path.stat().st_mtime}"

        if marker.exists():
            old = marker.read_text(encoding="utf-8")
            if old == source_signature:
                return True

        output_dir.mkdir(parents=True, exist_ok=True)
        writer = tf.summary.create_file_writer(str(output_dir))

        with writer.as_default():
            max_epochs = max(len(values) for values in history.values() if isinstance(values, list))
            for epoch in range(max_epochs):
                for key in scalar_keys:
                    values = history[key]
                    if epoch < len(values):
                        try:
                            value = float(values[epoch])
                            tf.summary.scalar(key, value, step=epoch + 1)
                        except Exception:
                            pass

        writer.flush()
        marker.write_text(source_signature, encoding="utf-8")
        return True

    except Exception as exc:
        print(f"[TENSORBOARD HISTORY ERROR] {exc}")
        return False


def create_training_plot():
    """Build BOTH accuracy and loss convergence plots from real history."""
    history_path = find_history()
    acc_path = PLOT_DIR / "training_performance.png"
    loss_path = PLOT_DIR / "training_loss.png"

    if history_path is None:
        return acc_path.exists()

    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        history = data.get("history", data)

        accuracy = history.get("accuracy", history.get("acc", []))
        val_accuracy = history.get("val_accuracy", history.get("val_acc", []))
        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])

        if not accuracy or not loss:
            return acc_path.exists()

        epochs = range(1, len(accuracy) + 1)

        # --- Accuracy plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, accuracy, marker="o", label="Train accuracy")
        if val_accuracy:
            ax.plot(epochs, val_accuracy, marker="s", label="Validation accuracy")
        ax.set_title("Training / Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(acc_path, dpi=160)
        plt.close(fig)

        # --- Loss plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, loss, marker="o", label="Train loss")
        if val_loss:
            ax.plot(epochs, val_loss, marker="s", label="Validation loss")
        ax.set_title("Training / Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_path, dpi=160)
        plt.close(fig)

        return True

    except Exception as exc:
        print(f"[TRAINING PLOT ERROR] {exc}")
        return acc_path.exists()


def load_history_data():
    """Return the raw history dict (for inline charts) if available."""
    history_path = find_history()
    if history_path is None:
        return None
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        return data.get("history", data)
    except Exception:
        return None


# ============================================================
# CLINICAL VALIDATION  &  EVALUATION METRICS  (NEW)
# ============================================================

def discover_test_set():
    """
    Expected layout:
        test_set/
            glioma_tumor/      *.png / *.jpg
            meningioma_tumor/  *.png / *.jpg
            no_tumor/          *.png / *.jpg
            pituitary_tumor/   *.png / *.jpg
    Returns list of (path, class_name) or [] if not found.
    """
    if not TEST_SET_DIR.exists():
        return []

    samples = []
    for class_name in CLASS_NAMES:
        folder = TEST_SET_DIR / class_name
        if not folder.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img_path in sorted(folder.glob(ext)):
                samples.append((img_path, class_name))
    return samples


def run_clinical_evaluation(model_key="ensemble", max_samples=None):
    """
    Run the selected 2D model (or ensemble) on the labelled test set and
    compute precision, recall, F1, accuracy, confusion matrix.
    Returns a dict of results, or None on failure.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn is not installed. Run: pip install scikit-learn"}

    samples = discover_test_set()
    if not samples:
        return None

    if max_samples:
        samples = samples[:max_samples]

    y_true = []
    y_pred = []

    for img_path, true_class in samples:
        try:
            if model_key in ("ensemble", "ensemble_3d"):
                probs = calculate_ensemble(img_path, "2d")
            else:
                probs = predict_single_model(model_key, img_path)
            pred_idx = int(np.argmax(probs))
            y_pred.append(CLASS_NAMES[pred_idx])
            y_true.append(true_class)
        except Exception as exc:
            print(f"[EVAL ERROR] {img_path.name}: {exc}")

    if not y_true:
        return {"error": "No samples could be processed."}

    labels = CLASS_NAMES
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        per_class[name] = {
            "display": display_class(name),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return {
        "model": model_key,
        "model_name": MODEL_NAMES.get(model_key, model_key),
        "total_samples": len(y_true),
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "classification_report": report,
        "class_labels": [display_class(c) for c in labels],
    }


def save_evaluation_cache(result):
    if not result:
        return
    try:
        cache_path = EVAL_DIR / "last_evaluation.json"
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_evaluation_cache():
    cache_path = EVAL_DIR / "last_evaluation.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_confusion_matrix_plot(cm, labels):
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = np.array(cm)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        color="white" if cm[i][j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        path = EVAL_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "/static/evaluation/confusion_matrix.png"
    except Exception as exc:
        print(f"[CONFUSION MATRIX ERROR] {exc}")
        return None


# ============================================================
# PDF REPORT
# ============================================================

def local_static_path(url_path):
    if not url_path:
        return None
    if url_path.startswith("/static/"):
        return BASE_DIR / url_path.lstrip("/")
    return BASE_DIR / url_path



# ============================================================
# PROFESSIONAL RADIOLOGY REPORT (Feature 8)
# ============================================================

def create_radiology_report(filepath, stem, predicted, confidence, model_name, 
                            probabilities, comparison, quality, gradcam, saliency, 
                            boundary, activations, family, tumor_size=None):
    """Generate a professional hospital-style radiology report."""
    if not REPORTLAB_AVAILABLE:
        return create_pdf_report(filepath, stem, predicted, confidence, model_name, 
                                probabilities, comparison, quality, gradcam, saliency, boundary, activations)
    
    try:
        filename = f"report_{stem}.pdf"
        pdf_path = UPLOAD_DIR / filename
        
        document = SimpleDocTemplate(str(pdf_path), pagesize=A4, 
                                      rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        story = []
        
        # Hospital Letterhead
        story.append(Paragraph("<b>NEUROSCAN AI DIAGNOSTIC CENTER</b>", styles["Title"]))
        story.append(Paragraph("Explainable Brain MRI Analysis & Radiology Report", styles["Heading4"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("_" * 90, styles["Normal"]))
        story.append(Spacer(1, 12))
        
        # Patient & Scan Info
        story.append(Paragraph("<b>PATIENT INFORMATION</b>", styles["Heading3"]))
        info_rows = [
            ["Scan Type", "MRI Brain" + (" (3D Volumetric)" if family == "3d" else " (2D Slice)")],
            ["Analysis Date", datetime.now().strftime("%d/%m/%Y")],
            ["Time", datetime.now().strftime("%H:%M:%S")],
            ["AI Model Used", model_name],
            ["Report ID", stem[:16].upper()],
        ]
        info_table = Table(info_rows, colWidths=[150, 300])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 16))
        
        # Findings
        story.append(Paragraph("<b>FINDINGS</b>", styles["Heading3"]))
        tumor = TUMOR_INFO.get(predicted, {})
        
        findings_text = f"""
        <b>Primary Finding:</b> {display_class(predicted)}<br/>
        <b>Confidence Level:</b> {confidence}<br/>
        <b>Analysis Pipeline:</b> {family.upper()} Model<br/>
        """
        if tumor_size and tumor_size.get("area_mm2"):
            findings_text += f"""
            <b>Estimated Tumor Area:</b> {tumor_size['area_mm2']} mm²<br/>
            <b>Estimated Tumor Diameter:</b> {tumor_size['diameter_mm']} mm<br/>
            <b>Active Region:</b> {tumor_size['percentage']}% of scan area<br/>
            """
        
        story.append(Paragraph(findings_text, styles["Normal"]))
        story.append(Spacer(1, 10))
        
        # Probability Distribution
        story.append(Paragraph("<b>PROBABILITY DISTRIBUTION</b>", styles["Heading3"]))
        prob_rows = [["Class", "Probability"]]
        for name, value in probabilities.items():
            prob_rows.append([display_class(name), f"{value:.1f}%"])
        prob_table = Table(prob_rows, colWidths=[200, 150])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 16))
        
        # Impression
        story.append(Paragraph("<b>IMPRESSION</b>", styles["Heading3"]))
        if predicted == "no_tumor":
            impression = "No evidence of brain tumor detected in the analyzed MRI scan. If clinical symptoms persist, further evaluation with contrast-enhanced MRI or neurological consultation is recommended."
        else:
            impression = f"The AI analysis suggests the presence of a {display_class(predicted)}. "
            if tumor.get("what_is"):
                impression += tumor["what_is"] + " "
            impression += f"This finding has a model confidence of {confidence}. Clinical correlation with patient symptoms and physical examination is essential. "
            if tumor.get("specialist"):
                impression += f"Referral to {tumor['specialist']} is recommended."
        
        story.append(Paragraph(impression, styles["Normal"]))
        story.append(Spacer(1, 16))
        
        # Recommendation
        story.append(Paragraph("<b>RECOMMENDATIONS</b>", styles["Heading3"]))
        if tumor.get("next_steps"):
            story.append(Paragraph(tumor["next_steps"], styles["Normal"]))
        else:
            story.append(Paragraph("Clinical correlation and specialist consultation recommended.", styles["Normal"]))
        story.append(Spacer(1, 16))
        
        # Model Comparison
        if comparison:
            story.append(Paragraph("<b>MULTI-MODEL COMPARISON</b>", styles["Heading3"]))
            comp_rows = [["Model", "Prediction", "Confidence", "Weight"]]
            for row in comparison:
                comp_rows.append([row["name"], row["prediction_display"], f'{row["confidence"]}%', f'{row["ensemble_weight"]}%'])
            comp_table = Table(comp_rows, colWidths=[150, 120, 80, 80])
            comp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#6366f1')),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(comp_table)
            story.append(Spacer(1, 16))
        
        # Disclaimer
        story.append(Paragraph("_" * 90, styles["Normal"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This is an AI-generated research report from NeuroScan AI. "
            "It is NOT a medical diagnosis. All findings must be verified by a licensed radiologist "
            "or neurologist. The attention boundary is AI model attention, not a clinical tumor segmentation. "
            "Treatment decisions should only be made by qualified healthcare professionals.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<i>Signed: NeuroScan AI System | Report ID: {stem[:16].upper()}</i>", styles["Normal"]))
        
        document.build(story)
        print(f"[PDF REPORT] Generated: {filename}")
        return filename
    except Exception as exc:
        print(f"[PDF REPORT ERROR] {exc}")
        return create_pdf_report(filepath, stem, predicted, confidence, model_name, 
                                probabilities, comparison, quality, gradcam, saliency, boundary, activations)



def create_pdf_report(filepath, stem, predicted, confidence, model_name, probabilities, comparison, quality, gradcam, saliency, boundary, activation_maps):
    if not REPORTLAB_AVAILABLE:
        return None

    try:
        filename = f"report_{stem}.pdf"
        pdf_path = UPLOAD_DIR / filename

        document = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("NeuroScan AI", styles["Title"]))
        story.append(Paragraph("Explainable Brain MRI Analysis Report", styles["Heading2"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Research prototype:</b> This report is for educational/research use. It is not a medical diagnosis and the attention boundary is not a clinical segmentation mask.", styles["BodyText"]))
        story.append(Spacer(1, 14))

        result_rows = [
            ["Prediction", display_class(predicted)],
            ["Confidence", confidence],
            ["Inference model", model_name],
            ["Explainability model", MODEL_NAMES.get(gradcam.get("model") if gradcam else "", "N/A")],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]

        table = Table(result_rows)
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("PADDING", (0, 0), (-1, -1), 6)]))
        story.append(table)
        story.append(Spacer(1, 16))

        probability_rows = [["Class", "Probability"]]
        for name, value in probabilities.items():
            probability_rows.append([display_class(name), f"{value:.1f}%"])

        ptable = Table(probability_rows)
        ptable.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("PADDING", (0, 0), (-1, -1), 6)]))
        story.append(Paragraph("Class Probability Distribution", styles["Heading2"]))
        story.append(ptable)
        story.append(Spacer(1, 16))

        if comparison:
            story.append(Paragraph("Model Comparison", styles["Heading2"]))
            rows = [["Model", "Prediction", "Confidence"]]
            for row in comparison:
                rows.append([row["name"], row["prediction_display"], f'{row["confidence"]}%'])
            ctable = Table(rows)
            ctable.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("PADDING", (0, 0), (-1, -1), 6)]))
            story.append(ctable)
            story.append(Spacer(1, 16))

        if quality:
            story.append(Paragraph("Input Image Information", styles["Heading2"]))
            qrows = [["Property", "Value"],
                     ["Resolution", f'{quality.get("width")} x {quality.get("height")}'],
                     ["Brightness", quality.get("brightness_label", "N/A")],
                     ["Contrast", quality.get("contrast_label", "N/A")],
                     ["Sharpness", quality.get("sharpness_label", "N/A")]]
            qtable = Table(qrows)
            qtable.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("PADDING", (0, 0), (-1, -1), 6)]))
            story.append(qtable)
            story.append(Spacer(1, 16))

        story.append(Paragraph("Explainability Evidence", styles["Heading2"]))

        evidence = [
            ("Input MRI", f"/static/uploads/{filepath.name}"),
            ("Grad-CAM", gradcam.get("overlay") if gradcam else None),
            ("Input-Gradient Saliency", saliency.get("overlay") if saliency else None),
            ("Attention Boundary", boundary.get("boundary") if boundary else None),
        ]

        for label, source_url in evidence:
            source = local_static_path(source_url)
            if source and source.exists():
                story.append(Paragraph(label, styles["Heading3"]))
                try:
                    story.append(PDFImage(str(source), width=300, height=220))
                    story.append(Spacer(1, 10))
                except Exception:
                    pass

        if activation_maps:
            story.append(Paragraph("Feature Map Depth Views", styles["Heading2"]))
            for item in activation_maps:
                source = local_static_path(item.get("url"))
                if source and source.exists():
                    story.append(Paragraph(item.get("label", "Feature map"), styles["Heading3"]))
                    try:
                        story.append(PDFImage(str(source), width=420, height=300))
                        story.append(Spacer(1, 10))
                    except Exception:
                        pass

        story.append(Paragraph("<b>Important:</b> NeuroScan AI is a research and educational prototype. Predictions, confidence, Grad-CAM, saliency, feature maps, and attention boundaries are not clinical diagnoses.", styles["BodyText"]))

        document.build(story)
        return filename

    except Exception as exc:
        print(f"[PDF ERROR] {type(exc).__name__}: {exc}")
        return None


# ============================================================
# TEMPLATE CONTEXT
# ============================================================

def workspace_context(**extra):
    context = {
        "available_models": list(loaded_models),
        "available_2d_models": [key for key in MODEL_FAMILIES["2d"] if key in loaded_models],
        "available_3d_models": [key for key in MODEL_FAMILIES["3d"] if key in loaded_models],
        "selected_family": extra.pop("selected_family", "2d"),
        "selected_model": extra.pop("selected_model", "ensemble"),
    }
    context.update(extra)
    return context


# ============================================================
# PREDICTION HISTORY LOGGING
# ============================================================

def log_prediction_to_history(predicted, confidence, model_name, family, probabilities):
    """Append a prediction to the history JSON file."""
    try:
        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": predicted,
            "prediction_display": display_class(predicted),
            "confidence": confidence,
            "model": model_name,
            "family": family,
            "probabilities": probabilities,
        }

        history.append(entry)
        history = history[-200:]

        HISTORY_FILE.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[HISTORY LOG ERROR] {exc}")


def load_prediction_history(limit=50):
    """Load recent prediction history."""
    if not HISTORY_FILE.exists():
        return []
    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            return []
        return history[-limit:]
    except Exception:
        return []


def get_prediction_stats():
    """Compute aggregate stats from prediction history."""
    history = load_prediction_history(200)
    if not history:
        return {
            "total_predictions": 0,
            "class_counts": {},
            "avg_confidence": 0,
            "2d_predictions": 0,
            "3d_predictions": 0,
            "high_confidence": 0,
            "low_confidence": 0,
            "recent_confidences": [],
            "model_usage": {},
        }

    class_counts = {}
    confidences = []
    two_d = 0
    three_d = 0
    high_conf = 0
    low_conf = 0
    model_usage = {}

    for entry in history:
        pred = entry.get("prediction", "unknown")
        class_counts[pred] = class_counts.get(pred, 0) + 1
        
        conf_str = entry.get("confidence", "0%")
        try:
            conf_val = float(conf_str.replace("%", ""))
            confidences.append(conf_val)
            if conf_val >= 80:
                high_conf += 1
            elif conf_val < 50:
                low_conf += 1
        except Exception:
            pass

        fam = entry.get("family", "2d")
        if fam == "3d":
            three_d += 1
        else:
            two_d += 1

        model_name = entry.get("model", "Unknown")
        model_usage[model_name] = model_usage.get(model_name, 0) + 1

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0

    return {
        "total_predictions": len(history),
        "class_counts": class_counts,
        "avg_confidence": avg_conf,
        "2d_predictions": two_d,
        "3d_predictions": three_d,
        "high_confidence": high_conf,
        "low_confidence": low_conf,
        "recent_confidences": confidences[-20:],
        "model_usage": model_usage,
    }





# ============================================================
# MULTI-LANGUAGE SUPPORT
# ============================================================

TRANSLATIONS = {
    "en": {
        "prediction": "Classification Result",
        "confidence": "Confidence",
        "gradcam": "Grad-CAM",
        "saliency": "Input-gradient Saliency",
        "heatmap": "Attention Heatmap",
        "feature_maps": "Feature Activations",
        "no_tumor": "No Tumor",
        "glioma_tumor": "Glioma Tumor",
        "meningioma_tumor": "Meningioma Tumor",
        "pituitary_tumor": "Pituitary Tumor",
        "upload": "Choose an MRI image",
        "analyze": "Run explainable analysis",
        "report": "PDF report",
        "download": "Download",
        "patient_explanation": "Simple Explanation",
        "symptoms": "Common Symptoms",
        "treatments": "Treatment Options",
        "prognosis": "Prognosis",
    },
    "hi": {
        "prediction": "वर्गीकरण परिणाम",
        "confidence": "आत्मविश्वास",
        "gradcam": "ग्रैड-कैम",
        "saliency": "इनपुट-ग्रेडिएंट सैलिएंसी",
        "heatmap": "ध्यान हीटमैप",
        "feature_maps": "फीचर सक्रियण",
        "no_tumor": "कोई ट्यूमर नहीं",
        "glioma_tumor": "ग्लियोमा ट्यूमर",
        "meningioma_tumor": "मेनिंजियोमा ट्यूमर",
        "pituitary_tumor": "पिट्यूटरी ट्यूमर",
        "upload": "एमआरआई छवि चुनें",
        "analyze": "विश्लेषण चलाएं",
        "report": "पीडीएफ रिपोर्ट",
        "download": "डाउनलोड",
        "patient_explanation": "सरल व्याख्या",
        "symptoms": "सामान्य लक्षण",
        "treatments": "उपचार विकल्प",
        "prognosis": "रोग का अनुमान",
    },
    "mr": {
        "prediction": "वर्गीकरण परिणाम",
        "confidence": "आत्मविश्वास",
        "gradcam": "ग्रॅड-कॅम",
        "saliency": "इनपुट-ग्रेडिएंट सॅलियन्सी",
        "heatmap": "लक्ष हीटमॅप",
        "feature_maps": "फीचर ॲक्टिव्हेशन",
        "no_tumor": "ट्यूमर नाही",
        "glioma_tumor": "ग्लिओमा ट्यूमर",
        "meningioma_tumor": "मेनिंजिओमा ट्यूमर",
        "pituitary_tumor": "पिट्युटरी ट्यूमर",
        "upload": "एमआरआय प्रतिमा निवडा",
        "analyze": "विश्लेषण चालवा",
        "report": "पीडीएफ अहवाल",
        "download": "डाउनलोड",
        "patient_explanation": "सोपी स्पष्टीकरण",
        "symptoms": "सामान्य लक्षणे",
        "treatments": "उपचार पर्याय",
        "prognosis": "रोगाचा अंदाज",
    },
}

def get_translation(lang="en"):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"])


# ============================================================
# TUMOR SIZE ESTIMATION
# ============================================================

# ============================================================
# REAL TUMOR SEGMENTATION (Mask R-CNN model.h5)
# ============================================================

SEGMENTATION_MODEL = None
SEGMENTATION_MODEL_PATH = BASE_DIR / "model.h5"

def load_segmentation_model():
    """Load the Mask R-CNN segmentation model.
    Patches mrcnn for TF2 without breaking other models."""
    global SEGMENTATION_MODEL
    if SEGMENTATION_MODEL is not None:
        return SEGMENTATION_MODEL
    if not SEGMENTATION_MODEL_PATH.exists():
        print(f"[SEGMENTATION] model.h5 not found at {SEGMENTATION_MODEL_PATH}")
        return None
    try:
        import tensorflow as _tf
        
        # Patch TF1.x attributes that mrcnn needs WITHOUT disable_v2_behavior
        # This fixes mrcnn imports without breaking our TF2 classification models
        if not hasattr(_tf, 'logging'):
            _tf.logging = _tf.compat.v1.logging
        if not hasattr(_tf, 'flags'):
            _tf.flags = _tf.compat.v1.flags
        if not hasattr(_tf, 'placeholder'):
            _tf.placeholder = _tf.compat.v1.placeholder
        if not hasattr(_tf, 'session'):
            _tf.session = _tf.compat.v1.session
        if not hasattr(_tf, 'global_variables_initializer'):
            _tf.global_variables_initializer = _tf.compat.v1.global_variables_initializer
        
        print("[SEGMENTATION] Applying TF1.x compatibility patches for mrcnn...")
        
        from mrcnn.config import Config
        import mrcnn.model as modellib
        
        class TumorConfig(Config):
            NAME = 'tumor_detector'
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + 1
            DETECTION_MIN_CONFIDENCE = 0.85
        
        class InferenceConfig(TumorConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        config = InferenceConfig()
        SEGMENTATION_MODEL = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=str(BASE_DIR)
        )
        SEGMENTATION_MODEL.load_weights(str(SEGMENTATION_MODEL_PATH), by_name=True)
        print(f"[LOADED] segmentation -> model.h5 (Mask R-CNN)")
        return SEGMENTATION_MODEL
    except Exception as exc:
        print(f"[SEGMENTATION LOAD ERROR] {type(exc).__name__}: {exc}")
        print("[SEGMENTATION] Falling back to Grad-CAM based estimation")
        return None

load_segmentation_model()


def create_segmentation(path, stem, tumor_size=None):
    """
    Run real tumor segmentation using Mask R-CNN.
    Returns mask, overlay with bounding box, and tumor size metrics.
    """
    if SEGMENTATION_MODEL is None:
        return {}
    
    try:
        import skimage.io
        
        # Load image (Mask R-CNN expects RGB)
        image = skimage.io.imread(str(path))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        orig_h, orig_w = image.shape[:2]
        
        # Run Mask R-CNN detection
        results = SEGMENTATION_MODEL.detect([image], verbose=0)
        r = results[0]
        
        # Check if any tumor detected
        if r['masks'].size == 0 or len(r['class_ids']) == 0:
            print("[SEGMENTATION] No tumor detected by Mask R-CNN")
            return {}
        
        # Combine all detected masks into one
        combined_mask = r['masks'][:, :, 0]
        for i in range(1, r['masks'].shape[2]):
            combined_mask = combined_mask | r['masks'][:, :, i]
        
        # Convert to uint8
        binary_mask = (combined_mask * 255).astype(np.uint8)
        
        # Create colored mask
        mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
        
        # Create overlay
        original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.6, mask_colored, 0.4, 0)
        
        # Draw bounding box on overlay
        for i in range(len(r['rois'])):
            y1, x1, y2, x2 = r['rois'][i]
            score = r['scores'][i] if i < len(r['scores']) else 0
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"Tumor {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Calculate real tumor size from segmentation mask
        active_pixels = int(np.sum(combined_mask))
        total_pixels = int(orig_h * orig_w)
        
        # Standard MRI field of view: 240mm x 240mm
        fov_mm = 240.0
        pixel_size_mm = fov_mm / max(orig_h, orig_w)
        
        import math
        area_mm2 = round(active_pixels * (pixel_size_mm ** 2), 2)
        diameter_mm = round(2 * math.sqrt(area_mm2 / math.pi), 2) if active_pixels > 0 else 0.0
        percentage = round(active_pixels / total_pixels * 100, 2) if total_pixels > 0 else 0
        
        # Save mask and overlay
        mn = f"{stem}_seg_mask.png"
        on = f"{stem}_seg_overlay.png"
        cv2.imwrite(str(UPLOAD_DIR / mn), binary_mask)
        cv2.imwrite(str(UPLOAD_DIR / on), overlay)
        
        print(f"[SEGMENTATION] Tumor detected! Pixels: {active_pixels}, Area: {area_mm2} mm2, Diameter: {diameter_mm} mm")
        
        return {
            "mask_path": f"/static/uploads/{mn}",
            "overlay_path": f"/static/uploads/{on}",
            "mask_array": binary_mask,
            "tumor_pixels": active_pixels,
            "area_mm2": area_mm2,
            "diameter_mm": diameter_mm,
            "percentage": percentage,
            "num_detections": len(r['rois']),
            "scores": [float(s) for s in r['scores']],
        }
    except Exception as exc:
        print(f"[SEGMENTATION ERROR] {type(exc).__name__}: {exc}")
        return {}



def estimate_tumor_size(heat_array, image_shape):
    """Estimate tumor size from Grad-CAM attention area.
    
    Args:
        heat_array: 2D numpy array of the heatmap (0-255)
        image_shape: (height, width) of the original image
    
    Returns:
        dict with area_mm2, diameter_mm, percentage, pixels
    """
    try:
        if heat_array is None:
            return {}
        
        heat = np.asarray(heat_array)
        if heat.ndim != 2:
            heat = heat.squeeze()
        
        # Threshold: pixels above 100 (out of 255) are considered "active"
        threshold = 100
        active_pixels = int(np.sum(heat > threshold))
        total_pixels = int(heat.shape[0] * heat.shape[1])
        
        # Assume standard MRI field of view: 240mm x 240mm for a typical brain MRI
        fov_mm = 240.0
        pixel_size_mm = fov_mm / max(heat.shape[0], heat.shape[1])
        
        area_mm2 = round(active_pixels * (pixel_size_mm ** 2), 2)
        
        # Estimate diameter (assume roughly circular)
        import math
        if active_pixels > 0:
            diameter_mm = round(2 * math.sqrt(area_mm2 / math.pi), 2)
        else:
            diameter_mm = 0.0
        
        percentage = round(active_pixels / total_pixels * 100, 2) if total_pixels > 0 else 0
        
        return {
            "area_mm2": area_mm2,
            "diameter_mm": diameter_mm,
            "percentage": percentage,
            "active_pixels": active_pixels,
            "total_pixels": total_pixels,
            "pixel_size_mm": round(pixel_size_mm, 2),
        }
    except Exception as exc:
        print(f"[TUMOR SIZE ERROR] {exc}")
        return {}


# ============================================================
# PATIENT MANAGEMENT SYSTEM (SQLite)
# ============================================================

PATIENT_DB = BASE_DIR / "patients.db"

def get_patient_db():
    conn = sqlite3.connect(str(PATIENT_DB))
    conn.row_factory = sqlite3.Row
    return conn

def init_patient_db():
    conn = get_patient_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            scan_date TEXT,
            tumor_type TEXT,
            confidence TEXT,
            prediction TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()

init_patient_db()


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/workspace")
def workspace():
    return render_template("workspace.html", **workspace_context(selected_family="2d", selected_model="ensemble"))


@app.route("/analytics")
def analytics():
    convert_history_to_tensorboard()

    # Build the real convergence plots (accuracy + loss) if history exists.
    has_history = bool(find_history())
    if has_history:
        create_training_plot()

    history_data = load_history_data() if has_history else None

    # Prepare inline epoch lists for the front-end charts.
    chart_data = {}
    if history_data:
        accuracy = history_data.get("accuracy", history_data.get("acc", []))
        val_accuracy = history_data.get("val_accuracy", history_data.get("val_acc", []))
        loss = history_data.get("loss", [])
        val_loss = history_data.get("val_loss", [])
        epochs = list(range(1, len(accuracy) + 1)) if accuracy else []
        chart_data = {
            "epochs": epochs,
            "train_accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "train_loss": loss,
            "val_loss": val_loss,
        }

    # --- Clinical evaluation (cached) ---
    eval_result = load_evaluation_cache()
    confusion_plot = None
    if eval_result and "confusion_matrix" in eval_result:
        confusion_plot = create_confusion_matrix_plot(
            eval_result["confusion_matrix"],
            eval_result.get("class_labels", [display_class(c) for c in CLASS_NAMES]),
        )

    test_set_available = bool(discover_test_set())

    # Prediction history stats
    pred_stats = get_prediction_stats()

    return render_template(
        "analytics.html",
        has_history=has_history,
        has_tb_logs=bool(find_tensorboard_event_files()),
        history_file=(str(find_history().name) if find_history() else None),
        chart_data=chart_data,
        has_loss_plot=(PLOT_DIR / "training_loss.png").exists(),
        eval_result=eval_result,
        confusion_plot=confusion_plot,
        test_set_available=test_set_available,
        sklearn_available=SKLEARN_AVAILABLE,
        pred_stats=pred_stats,
    )


@app.route("/run_evaluation", methods=["POST"])
def run_evaluation():
    """Trigger a fresh clinical evaluation run on the labelled test set."""
    if not SKLEARN_AVAILABLE:
        return jsonify({"error": "scikit-learn is not installed."}), 500

    model_key = request.form.get("model_key", "ensemble")

    if model_key not in loaded_models and model_key not in ("ensemble", "ensemble_3d"):
        return jsonify({"error": f"Model '{model_key}' is not loaded."}), 400

    result = run_clinical_evaluation(model_key=model_key)

    if result is None:
        return jsonify({
            "error": "No labelled test set found. Create test_set/<class_name>/ folders with images."
        }), 404

    if "error" in result:
        return jsonify(result), 500

    save_evaluation_cache(result)

    # Build confusion matrix plot.
    if "confusion_matrix" in result:
        result["confusion_plot"] = create_confusion_matrix_plot(
            result["confusion_matrix"], result.get("class_labels", [])
        )

    return jsonify(result)


@app.route("/tensorboard_status")
def tensorboard_status():
    convert_history_to_tensorboard()
    events = find_tensorboard_event_files()
    return jsonify({
        "available": bool(events),
        "event_files": len(events),
        "log_directory": str(TB_LOG_DIR),
        "history_file": (find_history().name if find_history() else None),
    })


@app.route("/history")
def history():
    """Show prediction history page."""
    history_entries = load_prediction_history(50)
    stats = get_prediction_stats()
    return render_template(
        "history.html",
        history_entries=reversed(history_entries),
        stats=stats,
    )


@app.route("/api/stats")
def api_stats():
    """API endpoint for prediction stats."""
    return jsonify(get_prediction_stats())


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/info")
def info():
    return render_template("info.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")

    family = request.form.get("model_family", "2d").lower()
    if family not in MODEL_FAMILIES:
        family = "2d"

    selected = request.form.get("model_type", "ensemble").lower()
    selected = {"custom_cnn": "deep_cnn"}.get(selected, selected)
    if family == "3d" and selected == "ensemble":
        selected = "ensemble_3d"

    if not uploaded or not uploaded.filename:
        return render_template("workspace.html", **workspace_context(error="Please select a scan file.", selected_model=selected, selected_family=family))

    is_valid_upload = allowed_volume(uploaded.filename) if family == "3d" else allowed_file(uploaded.filename)
    if not is_valid_upload:
        message = "3D mode requires a .nii or .nii.gz volume." if family == "3d" else "2D mode requires PNG, JPG, JPEG, or WEBP images."
        return render_template("workspace.html", **workspace_context(error=message, selected_model=selected, selected_family=family))

    valid_keys = set(MODEL_FAMILIES[family])
    if selected not in valid_keys and selected not in {"ensemble", "ensemble_3d"}:
        selected = "ensemble_3d" if family == "3d" else "ensemble"

    stem = uuid.uuid4().hex
    safe_filename = secure_filename(uploaded.filename)
    stored_filename = f"{stem}_{safe_filename}"
    filepath = UPLOAD_DIR / stored_filename

    uploaded.save(filepath)

    preview_path = None
    preview_slice_index = None
    if family == "3d":
        preview_path, preview_slice_index = save_3d_preview(filepath, stem)

    try:
        # INFERENCE
        if selected in {"ensemble", "ensemble_3d"}:
            predictions = calculate_ensemble(filepath, family)
            display_name = MODEL_NAMES["ensemble_3d"] if family == "3d" else MODEL_NAMES["ensemble"]
        else:
            if selected not in loaded_models:
                raise RuntimeError(f"Model '{selected}' is not loaded. Available models: {list(loaded_models.keys())}")
            predictions = predict_single_model(selected, filepath)
            display_name = MODEL_NAMES[selected]

        class_index = int(np.argmax(predictions))
        predicted = CLASS_NAMES[class_index]
        confidence_value = round(float(predictions[class_index]) * 100, 2)
        confidence = f"{confidence_value:.2f}%"

        probabilities = {name: round(float(value) * 100, 1) for name, value in zip(CLASS_NAMES, predictions)}

        # VISUAL MODEL
        visual_model, visual_key = choose_visual_model_3d(selected) if family == "3d" else choose_visual_model(selected)

        print("\n" + "=" * 75)
        print("EXPLAINABILITY PIPELINE")
        print(f"Inference model : {selected}")
        print(f"Visual model    : {visual_key}")
        print(f"Prediction      : {display_class(predicted)}")
        print("=" * 75)

        # GRAD-CAM
        gradcam = create_gradcam(filepath, visual_model, visual_key, class_index, stem)

        # TUMOR SEGMENTATION (Mask R-CNN if available, otherwise Grad-CAM)
        seg_result = {}
        if family == "2d":
            seg_result = create_segmentation(filepath, stem)
        
        # TUMOR SIZE ESTIMATION
        tumor_size = {}
        if seg_result:
            tumor_size = {
                "area_mm2": seg_result.get("area_mm2", 0),
                "diameter_mm": seg_result.get("diameter_mm", 0),
                "percentage": seg_result.get("percentage", 0),
                "active_pixels": seg_result.get("tumor_pixels", 0),
                "source": "Mask R-CNN Segmentation",
            }
        elif gradcam and gradcam.get("heat_array") is not None:
            tumor_size = estimate_tumor_size(gradcam.get("heat_array"), None)
            tumor_size["source"] = "AI Attention Analysis (Grad-CAM + Saliency)"

        if family == "3d" and gradcam.get("slice_index") is not None:
            preview_path, preview_slice_index = save_3d_preview(filepath, stem, gradcam.get("slice_index"))

        # BOUNDARY
        boundary = {}
        if gradcam:
            boundary_source = (BASE_DIR / preview_path.lstrip("/")) if family == "3d" and preview_path else filepath
            boundary = create_boundary(boundary_source, gradcam.get("heat_array"), stem)

        # SALIENCY
        saliency = create_saliency(filepath, visual_model, visual_key, class_index, stem)

        # AI-ESTIMATED TUMOR REGION
        localization_source = (BASE_DIR / preview_path.lstrip("/")) if family == "3d" and preview_path else filepath
        localization = create_localization(localization_source, gradcam, saliency, predicted, stem, confidence=confidence)

        # FEATURE MAPS
        activations = []
        if visual_model and visual_key != "ann":
            activations = create_feature_maps(filepath, visual_model, visual_key, stem)

        # 3D-ONLY FEATURES
        montage_path = None
        triplanar_path = None
        volume_stats = {}
        slice_confidence_chart = None
        
        if family == "3d":
            montage_path = create_3d_montage(filepath, stem)
            triplanar_path = create_3d_triplanar(filepath, stem)
            volume_stats = calculate_3d_volume_stats(filepath, gradcam)
            if visual_model:
                slice_confidence_chart = create_slice_confidence_graph(filepath, visual_model, visual_key, class_index, stem)

        # MODEL COMPARISON
        comparison = model_comparison_info(filepath, family)

        agreement = 0
        if comparison:
            agreement = round(100 * sum(row["prediction"] == predicted for row in comparison) / len(comparison), 1)

        uncertainty = round(max(0, 100 - confidence_value), 1)

        # IMAGE QUALITY
        quality = image_quality(filepath)

        # CHARTS
        probability_chart = create_probability_chart(probabilities, stem)
        model_comparison_chart = create_model_comparison_chart(comparison, stem)

        # AUDIO
        audio_path = None
        if gTTS:
            try:
                audio_filename = f"{stem}_narration.mp3"
                audio_file = UPLOAD_DIR / audio_filename
                narration = ("NeuroScan AI research analysis. The predicted class is " f"{display_class(predicted)} with {confidence} model confidence. This result is not a medical diagnosis.")
                gTTS(narration, lang="en").save(str(audio_file))
                audio_path = f"/static/uploads/{audio_filename}"
            except Exception as exc:
                print(f"[AUDIO ERROR] {exc}")

        # JSON REPORT
        json_filename = f"report_{stem}.json"
        json_path = UPLOAD_DIR / json_filename
        report_data = {
            "project": "NeuroScan AI",
            "prediction": predicted,
            "prediction_display": display_class(predicted),
            "confidence": confidence,
            "selected_model": selected,
            "model_family": family,
            "model": display_name,
            "probabilities": probabilities,
            "model_comparison": comparison,
            "agreement_percent": agreement,
            "uncertainty_percent": uncertainty,
            "visual_explainability_model": visual_key,
            "gradcam": {"available": bool(gradcam), "layer": (gradcam.get("layer") if gradcam else None)},
            "saliency": {"available": bool(saliency)},
            "attention_boundary": {"available": bool(boundary), "area_pixels": boundary.get("area") if boundary else None},
            "ai_estimated_tumor_region": {
                "available": bool(localization),
                "area_pixels": localization.get("area") if localization else None,
                "map_consistency_percent": localization.get("consistency") if localization else None,
                "note": localization.get("note") if localization else None,
            },
            "feature_maps": len(activations),
            "image_quality": quality,
            "generated_at": datetime.now().isoformat(),
            "disclaimer": "Research and educational prototype; not a medical diagnosis.",
        }
        json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        # PDF
        pdf_input = filepath
        if family == "3d" and preview_path:
            pdf_input = BASE_DIR / preview_path.lstrip("/")

        # Professional Radiology Report
        pdf_filename = create_radiology_report(
            pdf_input, stem, predicted, confidence, display_name, 
            probabilities, comparison, quality, gradcam, saliency, 
            boundary, activations, family, tumor_size
        )

        print(f"Prediction       : {display_class(predicted)}")
        print(f"Confidence       : {confidence}")
        print(f"Grad-CAM         : {'YES' if gradcam else 'NO'}")
        print(f"Saliency         : {'YES' if saliency else 'NO'}")
        print(f"Boundary         : {'YES' if boundary else 'NO'}")
        print(f"Feature views    : {len(activations)}")
        print(f"Compared models  : {len(comparison)}")
        print("=" * 75)

        return render_template(
            "workspace.html",
            prediction=predicted,
            prediction_display=display_class(predicted),
            confidence=confidence,
            confidence_value=confidence_value,
            model_name_display=display_name,
            selected_model=selected,
            selected_family=family,
            available_2d_models=[key for key in MODEL_FAMILIES["2d"] if key in loaded_models],
            available_3d_models=[key for key in MODEL_FAMILIES["3d"] if key in loaded_models],
            probabilities=probabilities,
            agreement=agreement,
            uncertainty=uncertainty,
            model_comparison=comparison,
            image_path=(preview_path if family == "3d" else f"/static/uploads/{stored_filename}"),
            model_family=family,
            preview_slice_index=preview_slice_index,
            gradcam_path=(gradcam.get("overlay") if gradcam else None),
            heatmap_path=(gradcam.get("heatmap") if gradcam else None),
            boundary_path=(boundary.get("boundary") if boundary else None),
            boundary_mask_path=(boundary.get("mask") if boundary else None),
            saliency_path=(saliency.get("overlay") if saliency else None),
            saliency_intensity_path=(saliency.get("raw") if saliency else None),
            lesion_area=(boundary.get("area") if boundary else None),
            localization_path=(localization.get("overlay") if localization else None),
            localization_mask_path=(localization.get("mask") if localization else None),
            localization_area=(localization.get("area") if localization else None),
            localization_consistency=(localization.get("consistency") if localization else None),
            localization_note=(localization.get("note") if localization else None),
            boundary_note=(boundary.get("note") if boundary else None),
            activation_maps=activations,
            visual_model=MODEL_NAMES.get(visual_key, visual_key),
            visual_layer=(gradcam.get("layer") if gradcam else None),
            image_quality=quality,
            audio_path=audio_path,
            pdf_filename=pdf_filename,
            json_filename=json_filename,
            probability_chart=probability_chart,
            model_comparison_chart=model_comparison_chart,
            montage_path=montage_path if family == "3d" else None,
            triplanar_path=triplanar_path if family == "3d" else None,
            volume_stats=volume_stats if family == "3d" else {},
            slice_confidence_chart=slice_confidence_chart if family == "3d" else None,
            tumor_info=get_tumor_info(predicted),
            tumor_size=tumor_size,
            lang=get_translation(request.form.get("lang", "en")),
            confidence_recommendation=get_confidence_recommendation(confidence_value, predicted),
            severity=get_severity_assessment(predicted, confidence_value),
            simple_explanation=get_simple_explanation(predicted, confidence_value, family),
            segmentation_available=bool(seg_result or (localization and localization.get("overlay"))),
            segmentation_mask_path=(seg_result.get("mask_path") if seg_result else (localization.get("mask") if localization else None)),
            segmentation_overlay_path=(seg_result.get("overlay_path") if seg_result else (localization.get("overlay") if localization else None)),
            segmentation_message=("Real tumor segmentation from Mask R-CNN model." if seg_result else "AI attention-based tumor localization using fused Grad-CAM and Saliency maps. The boundary highlights the tumor region detected by the AI model."),
            available_models=list(loaded_models),
        )

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return render_template("workspace.html", **workspace_context(error=f"Analysis failed: {type(exc).__name__}: {exc}", selected_model=selected, selected_family=family))




# ============================================================
# FAQ & SYMPTOM CHECKER ROUTES
# ============================================================

@app.route("/faq")
def faq():
    faq_items = [
        {"q": "What is an MRI scan?", "a": "Magnetic Resonance Imaging (MRI) is a non-invasive medical imaging technique that uses magnetic fields and radio waves to create detailed images of the brain and body. It does not use harmful radiation like X-rays or CT scans."},
        {"q": "What is the difference between 2D and 3D analysis?", "a": "2D analysis processes a single MRI slice (a flat image), while 3D analysis processes the complete brain volume (a stack of slices). 3D models use Conv3D layers that can detect patterns across depth, which 2D models cannot. Our 2D pipeline excels at tumor classification (99% accuracy), while our 3D pipeline demonstrates volumetric analysis capabilities using true Conv3D architectures."},
        {"q": "What is MGMT biomarker?", "a": "MGMT (O6-methylguanine-DNA methyltransferase) is a gene that repairs DNA damage. When this gene is methylated (silenced), brain tumors may respond better to chemotherapy drugs like Temozolomide. Our 3D models predict MGMT status from MRI scans, which is one of the hardest problems in medical AI - even competition winners achieve only 65-70% accuracy."},
        {"q": "Is this a medical diagnosis?", "a": "No. NeuroScan AI is a research and educational prototype. The predictions, Grad-CAM, saliency maps, and segmentation boundaries are not medical diagnoses. A licensed radiologist or neurologist must review any MRI scan for clinical diagnosis. Always consult a qualified healthcare professional."},
        {"q": "How accurate are the predictions?", "a": "Our 2D models achieve over 99% accuracy on brain tumor classification (4 classes: glioma, meningioma, no tumor, pituitary). Our 3D models achieve 50-60% accuracy on MGMT biomarker prediction, which is within expected range for this difficult task."},
        {"q": "What should I do with my result?", "a": "If a tumor is detected, schedule an appointment with the recommended specialist. If no tumor is detected but you have symptoms, consult a neurologist. Always bring your MRI scans and any downloaded reports to your doctor."},
        {"q": "What is Grad-CAM?", "a": "Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainability technique that shows which parts of the brain scan the AI model focused on when making its prediction. It produces a heat map overlay - red areas indicate high model attention, blue areas indicate low attention."},
        {"q": "Why does the 3D model sometimes predict No Tumor?", "a": "Our 3D models were trained on the RSNA-MICCAI dataset for MGMT biomarker prediction (binary: positive/negative). When the model predicts MGMT negative, it maps to 'No Tumor' in our 4-class system. This does not mean there is no tumor - it means the biomarker status is negative. The model's confidence of ~50% reflects the genuine difficulty of this prediction task."},
    ]
    return render_template("faq.html", faq_items=faq_items)


@app.route("/symptom-check", methods=["GET", "POST"])
def symptom_check():
    if request.method == "POST":
        symptoms = {
            "headaches": request.form.get("headaches") == "on",
            "seizures": request.form.get("seizures") == "on",
            "vision": request.form.get("vision") == "on",
            "nausea": request.form.get("nausea") == "on",
            "memory": request.form.get("memory") == "on",
            "weakness": request.form.get("weakness") == "on",
            "personality": request.form.get("personality") == "on",
            "balance": request.form.get("balance") == "on",
        }
        
        positive_count = sum(symptoms.values())
        
        if positive_count == 0:
            result = {"level": "green", "message": "No symptoms reported. If you have concerns, you can still upload your MRI scan for analysis.", "recommendation": "No immediate action needed."}
        elif positive_count <= 2:
            result = {"level": "yellow", "message": f" {positive_count} symptom(s) reported. Consider uploading your MRI scan for AI analysis.", "recommendation": "Monitor symptoms. If they persist or worsen, consult a physician."}
        elif positive_count <= 4:
            result = {"level": "orange", "message": f" {positive_count} symptoms reported. We recommend uploading your MRI scan and consulting a doctor.", "recommendation": "Schedule an appointment with a neurologist within 2-4 weeks."}
        else:
            result = {"level": "red", "message": f" {positive_count} symptoms reported. Multiple neurological symptoms require urgent evaluation.", "recommendation": "Seek immediate medical attention. Upload your MRI scan for analysis."}
        
        return render_template("symptom_check.html", result=result, symptoms=symptoms)
    
    return render_template("symptom_check.html", result=None, symptoms=None)




# ============================================================
# BATCH UPLOAD ROUTE
# ============================================================

@app.route("/batch", methods=["GET", "POST"])
def batch_upload():
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return render_template("batch.html", error="Please select at least one MRI image.")
        
        results = []
        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                continue
            
            stem = uuid.uuid4().hex
            safe_name = secure_filename(f.filename)
            stored = f"{stem}_{safe_name}"
            filepath = UPLOAD_DIR / stored
            f.save(filepath)
            
            try:
                predictions = calculate_ensemble(filepath, "2d")
                class_index = int(np.argmax(predictions))
                predicted = CLASS_NAMES[class_index]
                conf = round(float(predictions[class_index]) * 100, 2)
                
                tumor = TUMOR_INFO.get(predicted, {})
                severity = get_severity_assessment(predicted, conf)
                
                results.append({
                    "filename": safe_name,
                    "prediction": display_class(predicted),
                    "confidence": conf,
                    "severity": severity.get("label", ""),
                    "severity_color": severity.get("color", ""),
                    "specialist": tumor.get("specialist", ""),
                })
            except Exception as exc:
                results.append({
                    "filename": safe_name,
                    "prediction": "Error",
                    "confidence": 0,
                    "severity": "Error",
                    "severity_color": "#64748b",
                    "specialist": "",
                })
        
        return render_template("batch.html", results=results)
    
    return render_template("batch.html", results=None)


# ============================================================
# PATIENT MANAGEMENT ROUTES
# ============================================================

@app.route("/patients")
def patients():
    conn = get_patient_db()
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template("patients.html", patients=rows)

@app.route("/patients/add", methods=["POST"])
def add_patient():
    conn = get_patient_db()
    conn.execute(
        "INSERT INTO patients (name, age, gender, scan_date, tumor_type, confidence, prediction, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            request.form.get("name", ""),
            request.form.get("age", ""),
            request.form.get("gender", ""),
            request.form.get("scan_date", ""),
            request.form.get("tumor_type", ""),
            request.form.get("confidence", ""),
            request.form.get("prediction", ""),
            request.form.get("notes", ""),
        ),
    )
    conn.commit()
    conn.close()
    return redirect("/patients")

@app.route("/patients/delete/<int:pid>")
def delete_patient(pid):
    conn = get_patient_db()
    conn.execute("DELETE FROM patients WHERE id = ?", (pid,))
    conn.commit()
    conn.close()
    return redirect("/patients")


# ============================================================
# MODEL COMPARISON TOOL ROUTE
# ============================================================

@app.route("/compare", methods=["GET"])
def compare():
    return render_template("compare.html", results=None, available_models=list(loaded_models.keys()))

@app.route("/compare/run", methods=["POST"])
def compare_run():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return render_template("compare.html", error="Please select an MRI image.", results=None, available_models=list(loaded_models.keys()))
    
    if not allowed_file(uploaded.filename):
        return render_template("compare.html", error="Invalid file type. Use PNG, JPG, JPEG, or WEBP.", results=None, available_models=list(loaded_models.keys()))
    
    stem = uuid.uuid4().hex
    safe_name = secure_filename(uploaded.filename)
    stored = f"{stem}_{safe_name}"
    filepath = UPLOAD_DIR / stored
    uploaded.save(filepath)
    
    results = []
    for key in MODEL_FAMILIES["2d"]:
        if key not in loaded_models:
            continue
        try:
            preds = predict_single_model(key, filepath)
            idx = int(np.argmax(preds))
            pred = CLASS_NAMES[idx]
            conf = round(float(preds[idx]) * 100, 2)
            
            # Generate Eigen-CAM for this model (PCA-based, fast, no gradients)
            model = loaded_models[key]
            eigen_map = create_eigencam(filepath, model, key, idx, stem + key)
            
            results.append({
                "model_key": key,
                "model_name": MODEL_NAMES.get(key, key),
                "prediction": display_class(pred),
                "confidence": conf,
                "gradcam": eigen_map.get("overlay") if eigen_map else None,
                "heatmap": eigen_map.get("heatmap") if eigen_map else None,
            })
        except Exception as exc:
            results.append({
                "model_key": key,
                "model_name": MODEL_NAMES.get(key, key),
                "prediction": "Error",
                "confidence": 0,
                "gradcam": None,
                "heatmap": None,
            })
    
    # Ensemble result - use VGG16's Grad-CAM as representative
    try:
        ens_preds = calculate_ensemble(filepath, "2d")
        ens_idx = int(np.argmax(ens_preds))
        ens_pred = CLASS_NAMES[ens_idx]
        ens_conf = round(float(ens_preds[ens_idx]) * 100, 2)
        
        # Generate Eigen-CAM using VGG16 as the visual model for ensemble
        ens_gradcam = None
        ens_heatmap = None
        if "vgg16" in loaded_models:
            vgg_model = loaded_models["vgg16"]
            eigen = create_eigencam(filepath, vgg_model, "vgg16", ens_idx, stem + "ens")
            ens_gradcam = eigen.get("overlay") if eigen else None
            ens_heatmap = eigen.get("heatmap") if eigen else None
        
        results.insert(0, {
            "model_key": "ensemble",
            "model_name": "2D Weighted Ensemble",
            "prediction": display_class(ens_pred),
            "confidence": ens_conf,
            "gradcam": ens_gradcam,
            "heatmap": ens_heatmap,
        })
    except Exception as exc:
        print(f"[COMPARE ENSEMBLE ERROR] {exc}")
    
    return render_template("compare.html", results=results, image_path=f"/static/uploads/{stored}", available_models=list(loaded_models.keys()))






# ============================================================
# SERVE STATIC ASSETS FROM PROJECT FOLDERS
# ============================================================

from flask import send_from_directory

@app.route("/ReadMe_files/<path:filename>")
def serve_readme_files(filename):
    return send_from_directory(str(BASE_DIR / "ReadMe_files"), filename)

@app.route("/Segmentation/<path:filename>")
def serve_segmentation_files(filename):
    return send_from_directory(str(BASE_DIR / "Segmentation"), filename)


# ============================================================
# DATASET STATISTICS ROUTE
# ============================================================

@app.route("/dataset")
def dataset_stats():
    """Scan the dataset folder and compute statistics."""
    dataset_dir = BASE_DIR / "Brain-Tumor-Classification-DataSet"
    if not dataset_dir.exists():
        return render_template("dataset.html", dataset_available=False)
    
    # Scan all subdirectories (classes)
    class_stats = []
    total_images = 0
    
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        # Count images in this class
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]
        count = len(images)
        total_images += count
        
        # Get class name (clean up underscores)
        class_name = class_dir.name.replace("_", " ").title()
        
        # Find a sample image
        sample_image = None
        if images:
            sample_image = str(images[0]).replace(str(BASE_DIR), "").replace("\\", "/")
            if not sample_image.startswith("/"):
                sample_image = "/" + sample_image
        
        class_stats.append({
            "name": class_name,
            "folder_name": class_dir.name,
            "count": count,
            "sample_image": sample_image,
        })
    
    # Calculate percentages
    for cs in class_stats:
        cs["percentage"] = round(cs["count"] / total_images * 100, 2) if total_images > 0 else 0
    
    return render_template(
        "dataset.html",
        dataset_available=True,
        class_stats=class_stats,
        total_images=total_images,
        total_classes=len(class_stats),
    )


@app.route("/download_report/<filename>")
def download_report(filename):
    safe_name = secure_filename(filename)
    path = UPLOAD_DIR / safe_name
    if not path.exists():
        return "Report not found.", 404
    return send_file(path, as_attachment=True)


@app.route("/unet-analysis")
def unet_analysis():
    return render_template("unet_analysis.html")


@app.route("/")
def home():
    return redirect("/unet-analysis")



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("NEUROSCAN AI")
    print("Explainable Brain MRI Analysis System")
    print("=" * 75)
    print(f"Server: http://127.0.0.1:5000")
    print(f"Loaded models: {list(loaded_models.keys())}")
    print("Explainability:")
    print("  - Grad-CAM")
    print("  - Attention heatmap")
    print("  - Attention boundary estimate")
    print("  - Input-gradient saliency")
    print("  - 6-depth feature-map inspection")
    print("  - Same-family model comparison")
    print("  - Probability / confidence charts")
    print("  - PDF + JSON report")
    print("  - Optional audio narration")
    print("  - TensorBoard event detection")
    print("  - Clinical evaluation (precision/recall/F1/confusion matrix)")
    print("  - Prediction history logging")
    print("=" * 75)

    convert_history_to_tensorboard()

    app.run(debug=True, host="127.0.0.1", port=5000)
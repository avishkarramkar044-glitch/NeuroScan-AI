# ================================================================
# NEUROSCAN AI - U-NET FEATURES MODULE
# ================================================================
#
# U-Net segmentation / explainability module.
#
# IMPORTANT:
# - Keeps probability, heatmap, segmentation, thresholds, boundary,
#   measurements, saliency and feature maps.
# - Adds image-guided tumor-region refinement.
# - The refinement can expand a small U-Net seed to the full connected
#   lesion when the MRI intensity/texture supports that expansion.
# - Raw U-Net probability is preserved separately for transparency.
# - Adds a brain-area PLAUSIBILITY GUARD: even a raw U-Net mask (before
#   any image-guided refinement) that covers an implausibly large
#   fraction of the estimated brain region is flagged instead of being
#   silently reported as "TUMOR REGION FOUND". This closes a gap where
#   MAX_REFINED_IMAGE_FRACTION only protected against the refinement
#   step growing a mask, but never checked the raw U-Net output itself.
#
# MEDICAL NOTE:
# This is an AI research tool, not a medical diagnostic system.
# The refined region is NOT ground-truth and must be clinically
# validated. Physical size in mm requires DICOM pixel spacing.
# ================================================================

import os
import uuid
import traceback

import cv2
import numpy as np
import tensorflow as tf

from flask import request, jsonify, url_for
from werkzeug.utils import secure_filename


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "neuroscan_unet_improved.keras"
)

RESULT_DIR = os.path.join(
    BASE_DIR, "static", "unet_results"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR, "static", "uploads"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

THRESHOLD_VALUES = [0.50, 0.30, 0.20, 0.10]

MIN_COMPONENT_AREA = 20
MORPH_KERNEL_SIZE = 3

# Original component settings are retained.
RELATIVE_PEAK_KEEP = 0.45
MAX_COMPONENTS_TO_KEEP = 5

# ================================================================
# NEW IMAGE-GUIDED REFINEMENT SETTINGS
# ================================================================
#
# The U-Net prediction is used as the seed.  The algorithm is then
# allowed to expand ONLY into image-supported tissue around that seed.
#
# This is deliberately separate from the raw probability mask so the
# user can see both:
#   RAW U-NET -> what the network itself predicted
#   REFINED   -> U-Net seed + image-guided lesion refinement
#
REFINEMENT_ENABLED = True

# Seed pixels must be sufficiently confident.
SEED_PROBABILITY = 0.70

# Search around the U-Net seed.  This allows a small prediction to
# grow to a larger connected lesion.
SEARCH_DILATION_PIXELS = 45

# Minimum candidate intensity relative to the seed.
# A lower value allows heterogeneous tumor tissue to be included.
INTENSITY_LOWER_FROM_SEED = 0.35

# Extra local smoothing before candidate extraction.
REFINEMENT_BLUR_SIZE = 5

# Prevent the refinement from occupying an unreasonable fraction of
# the complete image.
MAX_REFINED_IMAGE_FRACTION = 0.45

# ================================================================
# NEW: PLAUSIBILITY GUARD (applies to the RAW mask too, not just
# the refinement step)
# ================================================================
#
# MAX_REFINED_IMAGE_FRACTION above only guards the image-guided
# refinement's *expansion* step. It does nothing if the raw U-Net
# output itself (before refinement, or the fallback path used when
# refinement is rejected/disabled) already covers most of the brain.
#
# A real lesion very rarely occupies a large fraction of the brain in
# a single slice. A near-solid / saturated prediction is much more
# likely to indicate a degenerate model output (bad checkpoint,
# saturated sigmoid, wrong preprocessing, domain-shifted input, etc.)
# than an actual tumor. Rather than silently reporting these as
# "TUMOR REGION FOUND", we flag them for human review.
#
# Fraction is measured relative to the ESTIMATED BRAIN AREA (not the
# whole image), since brain area itself varies by slice/crop.
MAX_PLAUSIBLE_LESION_FRACTION_OF_BRAIN = 0.20

# ================================================================
# MODEL GLOBALS
# ================================================================

UNET_MODEL = None
FEATURE_MODEL = None

UNET_MODEL_INPUT_SIZE = (128, 128)
UNET_INPUT_CHANNELS = 3

FEATURE_LAYER_INFO = []
SELECTED_FEATURE_INFO = []


# ================================================================
# PRINT HEADER
# ================================================================

def _print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ================================================================
# LOAD U-NET MODEL
# ================================================================

def load_unet_model(force_reload=False):
    global UNET_MODEL
    global UNET_MODEL_INPUT_SIZE
    global UNET_INPUT_CHANNELS

    if UNET_MODEL is not None and not force_reload:
        return UNET_MODEL

    _print_header("LOADING NEUROSCAN U-NET")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"U-Net model was not found:\n{MODEL_PATH}\n\n"
            "Make sure neuroscan_unet_improved.keras is inside "
            "the models folder."
        )

    try:
        print(f"Model path: {MODEL_PATH}")

        UNET_MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        input_shape = UNET_MODEL.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if len(input_shape) == 4:
            h, w, c = input_shape[1], input_shape[2], input_shape[3]

            if h is not None and w is not None:
                UNET_MODEL_INPUT_SIZE = (int(h), int(w))

            if c is not None:
                UNET_INPUT_CHANNELS = int(c)

        print("U-Net loaded successfully.")
        print(f"Input shape : {UNET_MODEL.input_shape}")
        print(f"Output shape: {UNET_MODEL.output_shape}")
        print(f"Input size  : {UNET_MODEL_INPUT_SIZE}")
        print(f"Channels    : {UNET_INPUT_CHANNELS}")
        print(f"Parameters  : {UNET_MODEL.count_params():,}")

        return UNET_MODEL

    except Exception as e:
        UNET_MODEL = None
        print("ERROR LOADING U-NET")
        print(str(e))
        print(traceback.format_exc())
        raise


# ================================================================
# FIND CONVOLUTIONAL FEATURE LAYERS
# ================================================================

def find_feature_layers(model=None):
    global FEATURE_LAYER_INFO

    if model is None:
        model = load_unet_model()

    layers = []

    for layer in model.layers:
        try:
            output_shape = layer.output.shape

            if output_shape is None or len(output_shape) != 4:
                continue

            layer_name = layer.name
            layer_type = layer.__class__.__name__

            if (
                "conv" in layer_type.lower()
                or "conv" in layer_name.lower()
            ):
                channels = output_shape[-1]

                if channels is not None:
                    layers.append({
                        "name": layer_name,
                        "channels": int(channels),
                        "index": layer.index,
                        "layer": layer
                    })

        except Exception:
            continue

    FEATURE_LAYER_INFO = layers
    return layers


# ================================================================
# BUILD FEATURE MODEL
# ================================================================

def build_feature_model():
    global FEATURE_MODEL
    global SELECTED_FEATURE_INFO

    if FEATURE_MODEL is not None:
        return FEATURE_MODEL

    model = load_unet_model()
    layers = find_feature_layers(model)

    if not layers:
        print("WARNING: No convolutional feature layers found.")
        return None

    n = len(layers)

    shallow_index = 0
    middle_index = max(
        0, min(n - 1, int(round((n - 1) * 0.50)))
    )
    deep_index = n - 1

    selected = [
        layers[shallow_index],
        layers[middle_index],
        layers[deep_index]
    ]

    unique = []
    seen = set()

    for item in selected:
        if item["name"] not in seen:
            unique.append(item)
            seen.add(item["name"])

    selected = unique
    SELECTED_FEATURE_INFO = selected

    outputs = [item["layer"].output for item in selected]

    FEATURE_MODEL = tf.keras.Model(
        inputs=model.input,
        outputs=outputs,
        name="neuroscan_unet_feature_extractor"
    )

    print()
    print("U-Net feature extractor created.")

    for i, item in enumerate(selected):
        level = ["SHALLOW", "MIDDLE", "DEEP"][min(i, 2)]
        print(
            f"{level}: {item['name']} "
            f"({item['channels']} channels)"
        )

    return FEATURE_MODEL


# ================================================================
# READ MRI IMAGE
# ================================================================

def read_mri_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read MRI image: {image_path}"
        )

    image = image.astype(np.uint8)

    if np.max(image) > np.min(image):
        normalized = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX
        )
    else:
        normalized = image.copy()

    return image, normalized


# ================================================================
# PREPROCESS MRI FOR U-NET
# ================================================================

def preprocess_mri(image_path):
    original_gray, normalized_gray = read_mri_image(image_path)

    original_h, original_w = normalized_gray.shape[:2]
    target_h, target_w = UNET_MODEL_INPUT_SIZE

    resized_gray = cv2.resize(
        normalized_gray,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    image_float = resized_gray.astype(np.float32) / 255.0

    if UNET_INPUT_CHANNELS == 1:
        model_image = np.expand_dims(image_float, axis=-1)

    elif UNET_INPUT_CHANNELS == 3:
        model_image = cv2.cvtColor(
            resized_gray, cv2.COLOR_GRAY2RGB
        )
        model_image = model_image.astype(np.float32) / 255.0

    else:
        model_image = np.repeat(
            image_float[..., np.newaxis],
            UNET_INPUT_CHANNELS,
            axis=-1
        )

    model_input = np.expand_dims(model_image, axis=0)

    return {
        "original_gray": original_gray,
        "normalized_gray": normalized_gray,
        "resized_gray": resized_gray,
        "model_input": model_input,
        "original_size": (original_w, original_h)
    }


# ================================================================
# CONVERT MODEL OUTPUT TO PROBABILITY MAP
# ================================================================

def extract_probability_map(prediction):
    pred = np.asarray(prediction)

    if pred.ndim == 4:
        pred = pred[0]

    if pred.ndim == 3:
        if pred.shape[-1] == 1:
            pred = pred[..., 0]
        else:
            pred = np.max(pred, axis=-1)

    if pred.ndim != 2:
        raise ValueError(
            f"Unexpected U-Net output shape: {np.asarray(prediction).shape}"
        )

    pred = pred.astype(np.float32)

    if np.min(pred) < 0.0 or np.max(pred) > 1.0:
        pred = 1.0 / (
            1.0 + np.exp(-np.clip(pred, -50, 50))
        )

    return np.clip(pred, 0.0, 1.0)


# ================================================================
# GENERATE PROBABILITY MAP
# ================================================================

def predict_probability(image_path):
    model = load_unet_model()
    data = preprocess_mri(image_path)

    prediction = model.predict(
        data["model_input"],
        verbose=0
    )

    probability_small = extract_probability_map(prediction)

    original_w, original_h = data["original_size"]

    probability = cv2.resize(
        probability_small,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    probability = np.clip(probability, 0.0, 1.0)

    return probability, data, prediction


# ================================================================
# ROBUST VISUAL CONTRAST
# ================================================================

def _visualize_probability(probability):
    p = probability.astype(np.float32)

    low = float(np.percentile(p, 1.0))
    high = float(np.percentile(p, 99.5))

    if high - low <= 1e-8:
        low = float(np.min(p))
        high = float(np.max(p))

    if high - low > 1e-8:
        visual = (p - low) / (high - low)
    else:
        visual = np.zeros_like(p)

    return np.clip(visual, 0.0, 1.0).astype(np.float32)


# ================================================================
# CREATE VISUAL PROBABILITY HEATMAP
# ================================================================

def create_probability_heatmap(probability, output_path):
    visual = _visualize_probability(probability)

    visual_u8 = (
        np.clip(visual * 255.0, 0, 255)
        .astype(np.uint8)
    )

    heatmap = cv2.applyColorMap(
        visual_u8,
        cv2.COLORMAP_TURBO
    )

    cv2.imwrite(output_path, heatmap)
    return heatmap


# ================================================================
# CREATE PROBABILITY OVERLAY
# ================================================================

def create_probability_overlay(mri_gray, probability, output_path):
    base = cv2.cvtColor(
        mri_gray,
        cv2.COLOR_GRAY2BGR
    )

    p_vis = _visualize_probability(probability)

    p_u8 = (
        np.clip(p_vis * 255.0, 0, 255)
        .astype(np.uint8)
    )

    heatmap = cv2.applyColorMap(
        p_u8,
        cv2.COLORMAP_TURBO
    )

    alpha_map = np.clip(
        p_vis * 0.75,
        0.0,
        0.75
    )[..., np.newaxis]

    overlay = (
        base.astype(np.float32) * (1.0 - alpha_map)
        + heatmap.astype(np.float32) * alpha_map
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    cv2.imwrite(output_path, overlay)
    return overlay


# ================================================================
# BINARY MASK
# ================================================================

def make_binary_mask(probability, threshold=0.50):
    return (
        probability >= float(threshold)
    ).astype(np.uint8) * 255


# ================================================================
# FILL HOLES
# ================================================================

def _fill_mask_holes(mask):
    binary = (mask > 0).astype(np.uint8)

    if not np.any(binary):
        return mask.copy()

    flood = binary.copy()

    flood_mask = np.zeros(
        (binary.shape[0] + 2, binary.shape[1] + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        1
    )

    outside = flood == 1
    holes = (~outside) & (binary == 0)

    filled = binary.copy()
    filled[holes] = 1

    return (filled * 255).astype(np.uint8)


# ================================================================
# REMOVE SMALL NOISE
# ================================================================

def clean_mask(mask, min_component_area=20):
    binary = (mask > 0).astype(np.uint8)

    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area >= int(min_component_area):
            cleaned[labels == label] = 1

    cleaned_mask = (cleaned * 255).astype(np.uint8)

    if np.any(cleaned_mask):
        cleaned_mask = _fill_mask_holes(cleaned_mask)

    return cleaned_mask


# ================================================================
# FIND TUMOR COMPONENTS
# ================================================================

def analyze_components(mask, probability=None):
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        cx = float(centroids[label][0])
        cy = float(centroids[label][1])

        item = {
            "label": int(label),
            "area_pixels": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "centroid_x": round(cx, 2),
            "centroid_y": round(cy, 2)
        }

        if probability is not None:
            region = labels == label

            if np.any(region):
                item["peak_probability"] = round(
                    float(np.max(probability[region])),
                    6
                )
                item["mean_probability"] = round(
                    float(np.mean(probability[region])),
                    6
                )
            else:
                item["peak_probability"] = 0.0
                item["mean_probability"] = 0.0

        components.append(item)

    components.sort(
        key=lambda x: x["area_pixels"],
        reverse=True
    )

    return components


# ================================================================
# PROBABILITY-GUIDED COMPONENT FILTER
# ================================================================

def select_relevant_components(
    mask,
    probability,
    min_component_area=20,
    max_components=MAX_COMPONENTS_TO_KEEP,
    relative_peak_keep=RELATIVE_PEAK_KEEP
):
    binary = (mask > 0).astype(np.uint8)

    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < int(min_component_area):
            continue

        region = labels == label
        peak = float(np.max(probability[region]))

        candidates.append({
            "label": label,
            "area": area,
            "peak": peak
        })

    if not candidates:
        return np.zeros_like(mask, dtype=np.uint8)

    strongest_peak = max(
        item["peak"] for item in candidates
    )

    candidates = [
        item
        for item in candidates
        if item["peak"] >= (
            strongest_peak * float(relative_peak_keep)
        )
    ]

    candidates.sort(
        key=lambda x: (x["peak"], x["area"]),
        reverse=True
    )

    candidates = candidates[:int(max_components)]

    selected = np.zeros_like(binary)

    for item in candidates:
        selected[labels == item["label"]] = 1

    return (selected * 255).astype(np.uint8)


# ================================================================
# NEW: BRAIN REGION ESTIMATION
# ================================================================

def estimate_brain_region(mri_gray):
    """
    Estimate the brain/head tissue region so intensity-based refinement
    does not grow into the black background or skull.

    This is only a computational ROI, not anatomical segmentation.
    """

    image = mri_gray.astype(np.uint8)

    # Ignore very dark background.
    nonzero = image[image > 5]

    if nonzero.size < 100:
        return np.ones_like(image, dtype=np.uint8) * 255

    threshold = max(
        5,
        int(np.percentile(nonzero, 8))
    )

    candidate = (image >= threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return candidate * 255

    largest_label = 1 + int(
        np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
    )

    brain = (labels == largest_label).astype(np.uint8)

    # Slightly close gaps.
    brain = cv2.morphologyEx(
        brain,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        ),
        iterations=1
    )

    return (brain * 255).astype(np.uint8)


# ================================================================
# NEW: MASK-VS-BRAIN PLAUSIBILITY CHECK
# ================================================================

def check_lesion_plausibility(
    mask,
    brain_mask,
    max_fraction=MAX_PLAUSIBLE_LESION_FRACTION_OF_BRAIN
):
    """
    Compare a candidate lesion mask's area against the estimated brain
    area. This is intentionally independent of MAX_REFINED_IMAGE_FRACTION,
    which only bounds the image-guided *refinement* step. This check
    applies to ANY final mask -- raw U-Net output, the refined mask, or
    the fallback path -- so a degenerate/saturated prediction can't
    silently pass through just because refinement was skipped or
    rejected.

    Returns:
        (is_implausible, fraction_of_brain)
    """

    brain_area = int(np.sum(brain_mask > 0))
    mask_area = int(np.sum(mask > 0))

    if brain_area <= 0:
        # No usable brain ROI estimate; can't judge plausibility.
        return False, 0.0

    fraction = mask_area / float(brain_area)

    is_implausible = fraction > float(max_fraction)

    return is_implausible, round(fraction, 6)


# ================================================================
# NEW: GET STRONG U-NET SEED
# ================================================================

def create_unet_seed(probability):
    """
    Find high-confidence U-Net pixels.

    A dynamic percentile is used in addition to the fixed probability
    threshold. This prevents a tiny but genuine 99% response from being
    lost simply because the surrounding lesion is lower-confidence.
    """

    p = probability.astype(np.float32)

    fixed_seed = p >= float(SEED_PROBABILITY)

    positive = p[p > 0.05]

    if positive.size > 20:
        dynamic_value = float(
            np.percentile(positive, 98.5)
        )
    else:
        dynamic_value = float(
            np.percentile(p, 99.0)
        )

    dynamic_value = float(
        np.clip(dynamic_value, 0.55, 0.95)
    )

    dynamic_seed = p >= dynamic_value

    seed = fixed_seed | dynamic_seed

    # Remove isolated single pixels.
    seed_u8 = seed.astype(np.uint8)

    seed_u8 = cv2.morphologyEx(
        seed_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    )

    return seed_u8 * 255


# ================================================================
# NEW: IMAGE-GUIDED LESION REFINEMENT
# ================================================================

def refine_tumor_mask_with_image(
    mri_gray,
    probability,
    threshold=0.50
):
    """
    Refine a small U-Net prediction into a connected lesion region.

    Design:
      1. U-Net supplies the seed.
      2. Brain ROI limits the search.
      3. A local search area is created around the seed.
      4. Intensity and gradient information identify compatible tissue.
      5. Only candidate pixels connected to the U-Net seed are retained.
      6. The result is bounded to avoid taking most of the image.

    IMPORTANT:
    This is an image-guided refinement, not a second trained neural
    network and not a clinically validated tumor segmentation.
    """

    p = probability.astype(np.float32)
    image = mri_gray.astype(np.uint8)

    if not np.any(p > 0.05):
        return np.zeros_like(image, dtype=np.uint8), {
            "status": "NO_U_NET_SEED",
            "seed_pixels": 0,
            "refined": False
        }

    seed = create_unet_seed(p)

    # If the strict seed is empty, use the selected threshold.
    if not np.any(seed):
        seed = make_binary_mask(p, threshold)

    brain = estimate_brain_region(image)

    seed = cv2.bitwise_and(
        seed,
        brain
    )

    if not np.any(seed):
        return np.zeros_like(image, dtype=np.uint8), {
            "status": "U_NET_SEED_OUTSIDE_BRAIN_ROI",
            "seed_pixels": 0,
            "refined": False
        }

    # ------------------------------------------------------------
    # Find the strongest/most useful seed component.
    # ------------------------------------------------------------

    seed_components = analyze_components(
        seed,
        probability=p
    )

    if not seed_components:
        return np.zeros_like(image, dtype=np.uint8), {
            "status": "NO_CONNECTED_U_NET_SEED",
            "seed_pixels": 0,
            "refined": False
        }

    # Prefer a component with high probability and reasonable area.
    seed_components.sort(
        key=lambda x: (
            x.get("peak_probability", 0.0),
            x["area_pixels"]
        ),
        reverse=True
    )

    strongest = seed_components[0]

    x = strongest["x"]
    y = strongest["y"]
    w = strongest["width"]
    h = strongest["height"]

    # ------------------------------------------------------------
    # Search region.
    # ------------------------------------------------------------

    pad = int(
        max(
            SEARCH_DILATION_PIXELS,
            0.75 * max(w, h)
        )
    )

    H, W = image.shape[:2]

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    search_region = np.zeros_like(image, dtype=np.uint8)
    search_region[y1:y2, x1:x2] = 255

    search_region = cv2.bitwise_and(
        search_region,
        brain
    )

    # ------------------------------------------------------------
    # Smooth intensity only for candidate estimation.
    # ------------------------------------------------------------

    blur_size = int(REFINEMENT_BLUR_SIZE)
    if blur_size % 2 == 0:
        blur_size += 1

    smooth = cv2.GaussianBlur(
        image,
        (blur_size, blur_size),
        0
    )

    seed_bool = (
        (seed > 0)
        & (brain > 0)
        & (search_region > 0)
    )

    seed_values = smooth[seed_bool]

    if seed_values.size == 0:
        return seed, {
            "status": "SEED_RETAINED_NO_INTENSITY_REFERENCE",
            "seed_pixels": int(np.sum(seed > 0)),
            "refined": False
        }

    # ------------------------------------------------------------
    # Robust seed intensity.
    # ------------------------------------------------------------

    seed_median = float(
        np.median(seed_values)
    )

    q25, q75 = np.percentile(
        seed_values,
        [25, 75]
    )

    iqr = float(q75 - q25)

    # The lower limit allows heterogeneous lesion tissue.
    lower = (
        seed_median
        - max(
            10.0,
            iqr * 1.8,
            seed_median * float(INTENSITY_LOWER_FROM_SEED)
        )
    )

    # We intentionally do not impose a strict upper intensity limit:
    # enhancing/bright portions of the lesion should remain eligible.
    lower = float(np.clip(lower, 0.0, 255.0))

    # ------------------------------------------------------------
    # Local intensity candidate.
    # ------------------------------------------------------------

    intensity_candidate = (
        (smooth >= lower)
        & (smooth <= 255)
        & (search_region > 0)
    )

    # ------------------------------------------------------------
    # Add a weak probability prior.
    #
    # Pixels with meaningful U-Net probability are always allowed.
    # For lower-probability pixels, intensity/connectedness must support
    # their inclusion.
    # ------------------------------------------------------------

    probability_support = p >= max(
        0.10,
        float(threshold) * 0.40
    )

    candidate = (
        intensity_candidate
        | (
            probability_support
            & (search_region > 0)
        )
    )

    candidate = candidate.astype(np.uint8)

    # Small closing helps bridge tiny gaps inside a heterogeneous mass.
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),
        iterations=1
    )

    # ------------------------------------------------------------
    # Keep only candidate pixels connected to the U-Net seed.
    # ------------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8
        )
    )

    seed_labels = np.unique(
        labels[seed > 0]
    )

    seed_labels = set(
        int(v)
        for v in seed_labels
        if int(v) != 0
    )

    if not seed_labels:
        return seed, {
            "status": "U_NET_SEED_NOT_CONNECTED_TO_IMAGE_CANDIDATE",
            "seed_pixels": int(np.sum(seed > 0)),
            "refined": False
        }

    refined_binary = np.isin(
        labels,
        list(seed_labels)
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # Prevent pathological whole-image expansion.
    # ------------------------------------------------------------

    refined_area = int(
        np.sum(refined_binary)
    )

    max_area = int(
        image.size * float(MAX_REFINED_IMAGE_FRACTION)
    )

    if refined_area > max_area:
        # Fall back to the raw selected mask rather than accepting a
        # clearly unreasonable expansion.
        fallback = make_binary_mask(
            p,
            threshold
        )

        fallback = cv2.bitwise_and(
            fallback,
            brain
        )

        fallback = clean_mask(
            fallback,
            min_component_area=MIN_COMPONENT_AREA
        )

        return fallback, {
            "status": "REFINEMENT_REJECTED_EXCESSIVE_AREA",
            "seed_pixels": int(np.sum(seed > 0)),
            "candidate_pixels": refined_area,
            "refined": False
        }

    refined = (refined_binary * 255).astype(np.uint8)

    # Fill internal holes and gently smooth contour.
    refined = _fill_mask_holes(refined)

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),
        iterations=1
    )

    # Never lose the actual high-confidence U-Net seed.
    refined = cv2.bitwise_or(
        refined,
        seed
    )

    return refined, {
        "status": "IMAGE_GUIDED_REFINEMENT_APPLIED",
        "seed_pixels": int(np.sum(seed > 0)),
        "candidate_pixels": refined_area,
        "refined_pixels": int(np.sum(refined > 0)),
        "seed_median_intensity": round(seed_median, 2),
        "lower_intensity_limit": round(lower, 2),
        "search_box": {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1)
        },
        "refined": True
    }


# ================================================================
# TUMOR GEOMETRY
# ================================================================

def calculate_tumor_geometry(
    mask,
    probability,
    pixel_spacing=None
):
    binary = mask > 0

    area_pixels = int(np.sum(binary))
    total_pixels = int(binary.size)

    area_percentage = (
        (area_pixels / total_pixels) * 100.0
        if total_pixels > 0 else 0.0
    )

    max_probability = float(np.max(probability))

    if area_pixels > 0:
        selected = probability[binary]
        mean_probability = float(np.mean(selected))
    else:
        mean_probability = 0.0

    components = analyze_components(
        mask,
        probability=probability
    )

    if components:
        largest = components[0]

        bbox = {
            "x": largest["x"],
            "y": largest["y"],
            "width": largest["width"],
            "height": largest["height"]
        }

        centroid = {
            "x": largest["centroid_x"],
            "y": largest["centroid_y"]
        }

        equivalent_diameter_pixels = (
            2.0 * np.sqrt(area_pixels / np.pi)
            if area_pixels > 0 else 0.0
        )

        max_dimension_pixels = float(
            max(
                largest["width"],
                largest["height"]
            )
        )
    else:
        bbox = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0
        }

        centroid = {"x": 0, "y": 0}
        equivalent_diameter_pixels = 0.0
        max_dimension_pixels = 0.0

    physical_area_mm2 = None

    physical_size_note = (
        "Pixel spacing was not supplied; physical tumor size "
        "in millimeters cannot be reliably calculated."
    )

    if pixel_spacing is not None:
        try:
            sx = float(pixel_spacing[0])
            sy = float(pixel_spacing[1])

            if sx > 0 and sy > 0:
                physical_area_mm2 = round(
                    area_pixels * sx * sy,
                    4
                )

                physical_size_note = (
                    "Area calculated from supplied pixel spacing."
                )
        except Exception:
            physical_area_mm2 = None

    return {
        "tumor_pixels": area_pixels,
        "image_pixels": total_pixels,
        "tumor_area_percentage": round(
            area_percentage,
            4
        ),
        "equivalent_diameter_pixels": round(
            float(equivalent_diameter_pixels),
            2
        ),
        "largest_dimension_pixels": round(
            max_dimension_pixels,
            2
        ),
        "physical_area_mm2": physical_area_mm2,
        "physical_size_note": physical_size_note,
        "maximum_probability": round(
            max_probability,
            6
        ),
        "mean_probability_in_region": round(
            mean_probability,
            6
        ),
        "component_count": len(components),
        "bounding_box": bbox,
        "centroid": centroid,
        "components": components
    }


# ================================================================
# CREATE BOUNDARY
# ================================================================

def create_boundary_overlay(
    mri_gray,
    mask,
    output_path
):
    output = cv2.cvtColor(
        mri_gray,
        cv2.COLOR_GRAY2BGR
    )

    binary = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cv2.drawContours(
            output,
            contours,
            -1,
            (0, 255, 0),
            2
        )

    components = analyze_components(mask)

    for index, component in enumerate(components):
        x = component["x"]
        y = component["y"]
        w = component["width"]
        h = component["height"]

        thickness = 2 if index == 0 else 1

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            thickness
        )

        cx = int(component["centroid_x"])
        cy = int(component["centroid_y"])

        cv2.circle(
            output,
            (cx, cy),
            4,
            (255, 0, 255),
            -1
        )

    cv2.imwrite(output_path, output)
    return output


# ================================================================
# SEGMENTATION OVERLAY
# ================================================================

def create_segmentation_overlay(
    mri_gray,
    mask,
    output_path
):
    base = cv2.cvtColor(
        mri_gray,
        cv2.COLOR_GRAY2BGR
    )

    binary = (mask > 0).astype(np.uint8)

    color_mask = np.zeros_like(base)
    color_mask[binary == 1] = (0, 255, 0)

    overlay = base.copy()
    region = binary == 1

    if np.any(region):
        overlay[region] = cv2.addWeighted(
            base[region],
            0.45,
            color_mask[region],
            0.55,
            0
        )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 255),
            2
        )

    cv2.imwrite(output_path, overlay)
    return overlay


# ================================================================
# SAVE MASK
# ================================================================

def save_mask(mask, output_path):
    cv2.imwrite(output_path, mask)
    return output_path


# ================================================================
# CREATE TUMOR REGION CROP
# ================================================================

def create_tumor_region_crop(
    mri_gray,
    mask,
    output_path,
    padding=10
):
    components = analyze_components(mask)

    if not components:
        return None

    largest = components[0]

    h, w = mri_gray.shape[:2]

    x1 = max(0, largest["x"] - padding)
    y1 = max(0, largest["y"] - padding)
    x2 = min(
        w,
        largest["x"] + largest["width"] + padding
    )
    y2 = min(
        h,
        largest["y"] + largest["height"] + padding
    )

    crop = mri_gray[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop_bgr = cv2.cvtColor(
        crop,
        cv2.COLOR_GRAY2BGR
    )

    crop_mask = mask[y1:y2, x1:x2]
    crop_region = crop_mask > 0

    if np.any(crop_region):
        green = np.zeros_like(crop_bgr)
        green[:] = (0, 255, 0)

        crop_bgr[crop_region] = cv2.addWeighted(
            crop_bgr[crop_region],
            0.45,
            green[crop_region],
            0.55,
            0
        )

    cv2.rectangle(
        crop_bgr,
        (0, 0),
        (
            crop_bgr.shape[1] - 1,
            crop_bgr.shape[0] - 1
        ),
        (0, 255, 255),
        2
    )

    cv2.imwrite(
        output_path,
        crop_bgr
    )

    return output_path


# ================================================================
# INPUT GRADIENT SALIENCY
# ================================================================

def create_input_gradient_saliency(
    model,
    model_input,
    probability=None,
    threshold=0.50
):
    x = tf.convert_to_tensor(
        model_input,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        tape.watch(x)

        prediction = model(
            x,
            training=False
        )

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        pred = prediction

        if len(pred.shape) == 4:
            if pred.shape[-1] == 1:
                pred2d = pred[..., 0]
            else:
                pred2d = tf.reduce_max(
                    pred,
                    axis=-1
                )
        else:
            pred2d = pred

        flat = tf.reshape(
            pred2d,
            [tf.shape(pred2d)[0], -1]
        )

        pixel_count = tf.shape(flat)[1]

        k = tf.maximum(
            tf.constant(32, dtype=tf.int32),
            tf.cast(
                tf.math.ceil(
                    tf.cast(
                        pixel_count,
                        tf.float32
                    ) * 0.05
                ),
                tf.int32
            )
        )

        k = tf.minimum(
            k,
            pixel_count
        )

        top_values = tf.math.top_k(
            flat,
            k=k,
            sorted=False
        ).values

        target = tf.reduce_mean(
            top_values
        )

        if probability is not None:
            positive_fraction = float(
                np.mean(
                    probability >= float(threshold)
                )
            )

            if positive_fraction > 0.0:
                target = (
                    0.75 * target
                    + 0.25 * tf.reduce_mean(pred2d)
                )

    gradients = tape.gradient(
        target,
        x
    )

    if gradients is None:
        raise RuntimeError(
            "Could not calculate input gradients."
        )

    gradients = gradients.numpy()[0]

    if gradients.ndim == 3:
        saliency = np.max(
            np.abs(gradients),
            axis=-1
        )
    else:
        saliency = np.abs(gradients)

    min_v = float(np.min(saliency))
    max_v = float(np.max(saliency))

    if max_v - min_v > 1e-8:
        saliency = (
            saliency - min_v
        ) / (
            max_v - min_v
        )
    else:
        saliency = np.zeros_like(
            saliency
        )

    return saliency.astype(
        np.float32
    )


# ================================================================
# SALIENCY HEATMAP
# ================================================================

def create_saliency_outputs(
    saliency,
    mri_gray,
    heatmap_path,
    overlay_path
):
    original_h, original_w = mri_gray.shape[:2]

    saliency_resized = cv2.resize(
        saliency,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    saliency_u8 = (
        np.clip(
            saliency_resized * 255,
            0,
            255
        )
        .astype(np.uint8)
    )

    heatmap = cv2.applyColorMap(
        saliency_u8,
        cv2.COLORMAP_TURBO
    )

    cv2.imwrite(
        heatmap_path,
        heatmap
    )

    base = cv2.cvtColor(
        mri_gray,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        base,
        0.50,
        heatmap,
        0.50,
        0
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    return (
        heatmap,
        overlay,
        saliency_resized
    )


# ================================================================
# FEATURE MAP PROCESSING
# ================================================================

def feature_activation_image(
    activation,
    target_size
):
    activation = np.asarray(
        activation
    )

    if activation.ndim == 4:
        activation = activation[0]

    if activation.ndim != 3:
        raise ValueError(
            f"Unexpected activation shape: {activation.shape}"
        )

    channel_maps = []
    channels = activation.shape[-1]

    max_channels = min(
        channels,
        128
    )

    for c in range(max_channels):
        feature = activation[..., c]
        feature = np.maximum(
            feature,
            0
        )

        f_min = float(
            np.min(feature)
        )

        f_max = float(
            np.max(feature)
        )

        if f_max - f_min > 1e-8:
            feature = (
                feature - f_min
            ) / (
                f_max - f_min
            )
        else:
            feature = np.zeros_like(
                feature
            )

        channel_maps.append(
            feature
        )

    if channel_maps:
        combined = np.max(
            np.stack(
                channel_maps,
                axis=0
            ),
            axis=0
        )
    else:
        combined = np.zeros(
            activation.shape[:2],
            dtype=np.float32
        )

    target_w, target_h = target_size

    combined = cv2.resize(
        combined,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )

    return np.clip(
        combined,
        0,
        1
    ).astype(np.float32)


# ================================================================
# SAVE FEATURE MAP
# ================================================================

def save_feature_map(
    feature_map,
    output_path
):
    feature_u8 = (
        np.clip(
            feature_map * 255,
            0,
            255
        )
        .astype(np.uint8)
    )

    heatmap = cv2.applyColorMap(
        feature_u8,
        cv2.COLORMAP_TURBO
    )

    cv2.imwrite(
        output_path,
        heatmap
    )

    return heatmap


# ================================================================
# FEATURE MAP OVERLAY
# ================================================================

def create_feature_overlay(
    mri_gray,
    feature_map,
    output_path
):
    h, w = mri_gray.shape[:2]

    feature_map = cv2.resize(
        feature_map,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    feature_u8 = (
        np.clip(
            feature_map * 255,
            0,
            255
        )
        .astype(np.uint8)
    )

    heatmap = cv2.applyColorMap(
        feature_u8,
        cv2.COLORMAP_TURBO
    )

    base = cv2.cvtColor(
        mri_gray,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        base,
        0.50,
        heatmap,
        0.50,
        0
    )

    cv2.imwrite(
        output_path,
        overlay
    )

    return overlay


# ================================================================
# GENERATE FEATURE MAPS
# ================================================================

def generate_feature_maps(
    model_input,
    mri_gray,
    output_prefix
):
    feature_model = build_feature_model()

    result = {
        "shallow": None,
        "middle": None,
        "deep": None
    }

    if feature_model is None:
        return result

    try:
        activations = feature_model.predict(
            model_input,
            verbose=0
        )

        if not isinstance(
            activations,
            list
        ):
            activations = [activations]

        h, w = mri_gray.shape[:2]

        for i, activation in enumerate(
            activations
        ):
            feature_map = feature_activation_image(
                activation,
                (w, h)
            )

            level = [
                "shallow",
                "middle",
                "deep"
            ][min(i, 2)]

            feature_path = (
                f"{output_prefix}_"
                f"{level}_features.jpg"
            )

            overlay_path = (
                f"{output_prefix}_"
                f"{level}_overlay.jpg"
            )

            save_feature_map(
                feature_map,
                feature_path
            )

            create_feature_overlay(
                mri_gray,
                feature_map,
                overlay_path
            )

            layer_name = None
            channels = None

            if i < len(
                SELECTED_FEATURE_INFO
            ):
                layer_name = (
                    SELECTED_FEATURE_INFO[i]["name"]
                )

                channels = (
                    SELECTED_FEATURE_INFO[i]["channels"]
                )

            result[level] = {
                "feature_map": feature_path,
                "overlay": overlay_path,
                "layer_name": layer_name,
                "channels": channels
            }

    except Exception as e:
        print(
            "Feature map generation warning:",
            str(e)
        )

    return result


# ================================================================
# SAVE ORIGINAL MRI
# ================================================================

def save_original_mri(
    image,
    output_path
):
    cv2.imwrite(
        output_path,
        image
    )

    return output_path


# ================================================================
# SAVE VISUALIZATION WITH TITLE
# ================================================================

def add_text_to_image(
    image,
    text
):
    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (
            min(
                output.shape[1],
                600
            ),
            38
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# ================================================================
# PARSE OPTIONAL PIXEL SPACING
# ================================================================

def _get_pixel_spacing_from_request():
    sx = request.form.get(
        "pixel_spacing_x"
    )

    sy = request.form.get(
        "pixel_spacing_y"
    )

    if sx is None or sy is None:
        return None

    try:
        sx = float(sx)
        sy = float(sy)

        if sx <= 0 or sy <= 0:
            return None

        return (
            sx,
            sy
        )

    except Exception:
        return None


# ================================================================
# COMPLETE U-NET ANALYSIS
# ================================================================

def run_unet_analysis(
    image_path,
    threshold=0.50,
    clean_small_regions=True,
    pixel_spacing=None
):
    _print_header(
        "NEUROSCAN U-NET ANALYSIS"
    )

    probability, data, prediction = (
        predict_probability(
            image_path
        )
    )

    mri = data["normalized_gray"]

    original_w, original_h = (
        data["original_size"]
    )

    analysis_id = uuid.uuid4().hex[:12]

    prefix = os.path.join(
        RESULT_DIR,
        f"unet_{analysis_id}"
    )

    # ------------------------------------------------------------
    # Original MRI
    # ------------------------------------------------------------

    original_path = (
        f"{prefix}_mri.jpg"
    )

    save_original_mri(
        mri,
        original_path
    )

    # ------------------------------------------------------------
    # RAW probability heatmap
    # ------------------------------------------------------------

    probability_heatmap_path = (
        f"{prefix}_probability_heatmap.jpg"
    )

    create_probability_heatmap(
        probability,
        probability_heatmap_path
    )

    # ------------------------------------------------------------
    # RAW probability overlay
    # ------------------------------------------------------------

    probability_overlay_path = (
        f"{prefix}_probability_overlay.jpg"
    )

    create_probability_overlay(
        mri,
        probability,
        probability_overlay_path
    )

    # ------------------------------------------------------------
    # RAW U-NET MASK
    # ------------------------------------------------------------

    raw_mask = make_binary_mask(
        probability,
        threshold
    )

    if clean_small_regions:
        cleaned_mask = clean_mask(
            raw_mask,
            min_component_area=MIN_COMPONENT_AREA
        )
    else:
        cleaned_mask = raw_mask

    raw_selected_mask = (
        select_relevant_components(
            cleaned_mask,
            probability,
            min_component_area=MIN_COMPONENT_AREA,
            max_components=MAX_COMPONENTS_TO_KEEP,
            relative_peak_keep=RELATIVE_PEAK_KEEP
        )
    )

    if (
        np.any(cleaned_mask)
        and not np.any(raw_selected_mask)
    ):
        raw_selected_mask = cleaned_mask

    # ------------------------------------------------------------
    # NEW: brain ROI estimate used for the plausibility guard below.
    # Computed once here and reused, independent of whether the
    # image-guided refinement step runs, is skipped, or is rejected.
    # ------------------------------------------------------------

    brain_mask_for_plausibility = estimate_brain_region(mri)

    # ------------------------------------------------------------
    # NEW IMAGE-GUIDED REFINEMENT
    # ------------------------------------------------------------

    refinement_info = {
        "status": "DISABLED",
        "refined": False
    }

    if REFINEMENT_ENABLED:
        try:
            refined_mask, refinement_info = (
                refine_tumor_mask_with_image(
                    mri,
                    probability,
                    threshold=threshold
                )
            )

            # If refinement fails, keep original U-Net mask.
            if not np.any(refined_mask):
                segmentation_mask = (
                    raw_selected_mask
                )

                refinement_info[
                    "fallback_to_raw_mask"
                ] = True
            else:
                segmentation_mask = (
                    refined_mask
                )

        except Exception as e:
            print(
                "Image-guided refinement warning:",
                str(e)
            )

            segmentation_mask = (
                raw_selected_mask
            )

            refinement_info = {
                "status": "REFINEMENT_ERROR_RAW_MASK_USED",
                "refined": False,
                "error": str(e)
            }

    else:
        segmentation_mask = (
            raw_selected_mask
        )

    # ------------------------------------------------------------
    # Safety: preserve raw U-Net seed.
    # ------------------------------------------------------------

    if not np.any(segmentation_mask):
        segmentation_mask = (
            raw_selected_mask
        )

    # ------------------------------------------------------------
    # NEW: PLAUSIBILITY GUARD
    #
    # This runs regardless of which path produced segmentation_mask
    # (image-guided refinement, its excessive-area fallback, a
    # refinement error fallback, or REFINEMENT_ENABLED == False).
    # It also evaluates the raw U-Net mask on its own, since that is
    # what every fallback ultimately derives from.
    # ------------------------------------------------------------

    raw_implausible, raw_lesion_fraction_of_brain = (
        check_lesion_plausibility(
            raw_selected_mask,
            brain_mask_for_plausibility
        )
    )

    final_implausible, final_lesion_fraction_of_brain = (
        check_lesion_plausibility(
            segmentation_mask,
            brain_mask_for_plausibility
        )
    )

    # ------------------------------------------------------------
    # Save raw mask separately.
    # ------------------------------------------------------------

    raw_mask_path = (
        f"{prefix}_raw_unet_mask.jpg"
    )

    save_mask(
        raw_selected_mask,
        raw_mask_path
    )

    # ------------------------------------------------------------
    # Save final refined mask.
    # ------------------------------------------------------------

    mask_path = (
        f"{prefix}_binary_mask.jpg"
    )

    save_mask(
        segmentation_mask,
        mask_path
    )

    # ------------------------------------------------------------
    # Segmentation overlay
    # ------------------------------------------------------------

    segmentation_overlay_path = (
        f"{prefix}_segmentation_overlay.jpg"
    )

    create_segmentation_overlay(
        mri,
        segmentation_mask,
        segmentation_overlay_path
    )

    # ------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------

    boundary_path = (
        f"{prefix}_boundary.jpg"
    )

    create_boundary_overlay(
        mri,
        segmentation_mask,
        boundary_path
    )

    # ------------------------------------------------------------
    # Tumor region crop
    # ------------------------------------------------------------

    tumor_region_path = (
        f"{prefix}_tumor_region.jpg"
    )

    tumor_region_path = (
        create_tumor_region_crop(
            mri,
            segmentation_mask,
            tumor_region_path,
            padding=10
        )
    )

    # ------------------------------------------------------------
    # Threshold maps
    # ------------------------------------------------------------

    threshold_results = {}

    for t in THRESHOLD_VALUES:
        threshold_mask = make_binary_mask(
            probability,
            t
        )

        if clean_small_regions:
            threshold_mask = clean_mask(
                threshold_mask,
                min_component_area=MIN_COMPONENT_AREA
            )

        threshold_mask = (
            select_relevant_components(
                threshold_mask,
                probability,
                min_component_area=MIN_COMPONENT_AREA,
                max_components=MAX_COMPONENTS_TO_KEEP,
                relative_peak_keep=RELATIVE_PEAK_KEEP
            )
        )

        if not np.any(threshold_mask):
            fallback = make_binary_mask(
                probability,
                t
            )

            if clean_small_regions:
                fallback = clean_mask(
                    fallback,
                    min_component_area=MIN_COMPONENT_AREA
                )

            threshold_mask = fallback

        filename = (
            f"{prefix}_threshold_"
            f"{int(t * 100):02d}.jpg"
        )

        save_mask(
            threshold_mask,
            filename
        )

        threshold_geometry = (
            calculate_tumor_geometry(
                threshold_mask,
                probability,
                pixel_spacing=pixel_spacing
            )
        )

        threshold_results[str(t)] = {
            "mask": filename,
            "tumor_pixels": (
                threshold_geometry[
                    "tumor_pixels"
                ]
            ),
            "tumor_area_percentage": (
                threshold_geometry[
                    "tumor_area_percentage"
                ]
            ),
            "maximum_probability": (
                threshold_geometry[
                    "maximum_probability"
                ]
            ),
            "component_count": (
                threshold_geometry[
                    "component_count"
                ]
            )
        }

    # ------------------------------------------------------------
    # Main geometry
    # ------------------------------------------------------------

    geometry = calculate_tumor_geometry(
        segmentation_mask,
        probability,
        pixel_spacing=pixel_spacing
    )

    # ------------------------------------------------------------
    # Input gradient saliency
    # ------------------------------------------------------------

    saliency_heatmap_path = (
        f"{prefix}_saliency_heatmap.jpg"
    )

    saliency_overlay_path = (
        f"{prefix}_saliency_overlay.jpg"
    )

    saliency_result = None

    try:
        model = load_unet_model()

        saliency = (
            create_input_gradient_saliency(
                model,
                data["model_input"],
                probability=probability,
                threshold=threshold
            )
        )

        _, _, saliency_resized = (
            create_saliency_outputs(
                saliency,
                mri,
                saliency_heatmap_path,
                saliency_overlay_path
            )
        )

        saliency_result = {
            "heatmap": saliency_heatmap_path,
            "overlay": saliency_overlay_path,
            "maximum_intensity": round(
                float(
                    np.max(
                        saliency_resized
                    )
                ),
                6
            ),
            "mean_intensity": round(
                float(
                    np.mean(
                        saliency_resized
                    )
                ),
                6
            )
        }

    except Exception as e:
        print(
            "Saliency warning:",
            str(e)
        )

    # ------------------------------------------------------------
    # Feature maps
    # ------------------------------------------------------------

    feature_results = (
        generate_feature_maps(
            data["model_input"],
            mri,
            prefix
        )
    )

    # ------------------------------------------------------------
    # Probability statistics
    # ------------------------------------------------------------

    max_probability = float(
        np.max(probability)
    )

    mean_probability = float(
        np.mean(probability)
    )

    # ------------------------------------------------------------
    # Response level
    # ------------------------------------------------------------

    if max_probability >= 0.75:
        confidence_level = (
            "HIGH U-NET RESPONSE"
        )
    elif max_probability >= 0.50:
        confidence_level = (
            "MODERATE U-NET RESPONSE"
        )
    elif max_probability >= 0.20:
        confidence_level = (
            "LOW U-NET RESPONSE"
        )
    else:
        confidence_level = (
            "VERY LOW U-NET RESPONSE"
        )

    # ------------------------------------------------------------
    # Tumor response status
    #
    # NEW: a mask covering an implausibly large fraction of the brain
    # is flagged for review instead of being reported as a normal
    # "TUMOR REGION FOUND" result. This applies whether that large
    # mask came from the raw U-Net output, a refinement fallback, or
    # the refinement step itself.
    # ------------------------------------------------------------

    predicted_tumor_pixels = int(
        geometry["tumor_pixels"]
    )

    if predicted_tumor_pixels == 0:
        segmentation_status = (
            "NO TUMOR REGION ABOVE SELECTED THRESHOLD"
        )
    elif final_implausible:
        segmentation_status = (
            "IMPLAUSIBLE_PREDICTION_REVIEW_REQUIRED"
        )
    else:
        segmentation_status = (
            "TUMOR REGION FOUND"
        )

    # ------------------------------------------------------------
    # Domain warning
    # ------------------------------------------------------------

    domain_note = (
        "The U-Net was trained on its training dataset domain. "
        "Performance may decrease on MRI images from different "
        "sources, scanners, acquisition protocols, or tumor types."
    )

    # ------------------------------------------------------------
    # Print analysis
    # ------------------------------------------------------------

    print()
    print("U-NET RESULTS")
    print("-" * 72)

    print(
        f"Maximum probability : "
        f"{max_probability:.6f}"
    )

    print(
        f"Mean probability    : "
        f"{mean_probability:.6f}"
    )

    print(
        f"Mean region prob.   : "
        f"{geometry['mean_probability_in_region']:.6f}"
    )

    print(
        f"RAW U-Net pixels    : "
        f"{int(np.sum(raw_selected_mask > 0))}"
    )

    print(
        f"RAW/brain fraction  : "
        f"{raw_lesion_fraction_of_brain:.4f}"
        f"{'  <-- FLAGGED' if raw_implausible else ''}"
    )

    print(
        f"FINAL tumor pixels  : "
        f"{geometry['tumor_pixels']}"
    )

    print(
        f"FINAL/brain fraction: "
        f"{final_lesion_fraction_of_brain:.4f}"
        f"{'  <-- FLAGGED' if final_implausible else ''}"
    )

    print(
        f"Tumor area          : "
        f"{geometry['tumor_area_percentage']:.4f}%"
    )

    print(
        f"Equivalent diameter : "
        f"{geometry['equivalent_diameter_pixels']:.2f} px"
    )

    print(
        f"Components          : "
        f"{geometry['component_count']}"
    )

    print(
        f"Status              : "
        f"{segmentation_status}"
    )

    print(
        f"Refinement          : "
        f"{refinement_info.get('status', 'UNKNOWN')}"
    )

    print(
        f"Response level      : "
        f"{confidence_level}"
    )

    print()
    print("=" * 72)

    # ------------------------------------------------------------
    # Return complete result
    # ------------------------------------------------------------

    return {
        "success": True,

        "analysis_id": analysis_id,

        "model": {
            "name": "NeuroScan U-Net Improved",
            "path": MODEL_PATH,
            "input_size": [
                int(UNET_MODEL_INPUT_SIZE[0]),
                int(UNET_MODEL_INPUT_SIZE[1])
            ],
            "input_channels": int(
                UNET_INPUT_CHANNELS
            )
        },

        "image": {
            "width": int(original_w),
            "height": int(original_h)
        },

        "probability": {
            "maximum": round(
                max_probability,
                6
            ),
            "mean": round(
                mean_probability,
                6
            ),
            "minimum": round(
                float(np.min(probability)),
                6
            ),
            "heatmap": (
                probability_heatmap_path
            ),
            "overlay": (
                probability_overlay_path
            )
        },

        "segmentation": {
            "threshold": float(threshold),

            # Final image-guided mask.
            "mask": mask_path,

            "overlay": (
                segmentation_overlay_path
            ),

            "boundary": boundary_path,

            "tumor_region": (
                tumor_region_path
            ),

            "raw_unet_mask": raw_mask_path,

            "status": segmentation_status,

            "refinement": refinement_info,

            # NEW: brain-area plausibility diagnostics, independent
            # of MAX_REFINED_IMAGE_FRACTION (which only guards the
            # refinement expansion step).
            "plausibility": {
                "max_plausible_fraction_of_brain": (
                    MAX_PLAUSIBLE_LESION_FRACTION_OF_BRAIN
                ),
                "raw_mask_fraction_of_brain": (
                    raw_lesion_fraction_of_brain
                ),
                "raw_mask_flagged": raw_implausible,
                "final_mask_fraction_of_brain": (
                    final_lesion_fraction_of_brain
                ),
                "final_mask_flagged": final_implausible
            }
        },

        "measurements": geometry,

        "thresholds": threshold_results,

        "saliency": saliency_result,

        "feature_maps": feature_results,

        "interpretation": {
            "response_level": confidence_level,

            "warning": (
                "U-Net segmentation and image-guided refinement "
                "are AI-generated research outputs and are not "
                "a medical diagnosis."
            ),

            "domain_note": domain_note,

            "segmentation_note": (
                "The final mask uses the U-Net prediction as a "
                "seed and may use local MRI intensity/connectivity "
                "to refine the visible lesion region. It is not "
                "ground-truth segmentation."
            ),

            "plausibility_note": (
                "If 'plausibility.final_mask_flagged' is true, the "
                "predicted region covers an unusually large fraction "
                "of the estimated brain area for a single lesion. "
                "This is more consistent with a degenerate model "
                "output (e.g. a saturated or miscalibrated "
                "prediction, or an out-of-domain input) than an "
                "actual tumor, and should be reviewed manually "
                "before being trusted."
            )
        },

        "files": {
            "mri": original_path,

            "probability_heatmap": (
                probability_heatmap_path
            ),

            "probability_overlay": (
                probability_overlay_path
            ),

            "raw_unet_mask": raw_mask_path,

            "binary_mask": mask_path,

            "segmentation_overlay": (
                segmentation_overlay_path
            ),

            "boundary": boundary_path,

            "tumor_region": (
                tumor_region_path
            )
        }
    }


# ================================================================
# CONVERT LOCAL FILE PATH TO STATIC URL
# ================================================================

def local_path_to_url(
    app,
    file_path
):
    try:
        relative = os.path.relpath(
            file_path,
            os.path.join(
                BASE_DIR,
                "static"
            )
        )

        relative = relative.replace(
            os.sep,
            "/"
        )

        return url_for(
            "static",
            filename=relative,
            _external=False
        )

    except Exception:
        return None


# ================================================================
# CONVERT RESULT PATHS TO URLS
# ================================================================

def convert_result_paths_to_urls(
    app,
    data
):
    if isinstance(data, dict):
        return {
            key: convert_result_paths_to_urls(
                app,
                value
            )
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            convert_result_paths_to_urls(
                app,
                item
            )
            for item in data
        ]

    if isinstance(data, str):
        normalized = os.path.normpath(
            data
        )

        result_dir_normalized = (
            os.path.normpath(
                RESULT_DIR
            )
        )

        try:
            is_result_file = (
                os.path.commonpath(
                    [
                        normalized,
                        result_dir_normalized
                    ]
                )
                == result_dir_normalized
            )
        except Exception:
            is_result_file = False

        if is_result_file:
            return local_path_to_url(
                app,
                data
            )

    return data


# ================================================================
# FLASK ENDPOINT
# ================================================================

def unet_analyze_endpoint(app):
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": (
                "No MRI image was provided. "
                "Use form field name 'image'."
            )
        }), 400

    uploaded_file = request.files[
        "image"
    ]

    if (
        uploaded_file is None
        or uploaded_file.filename == ""
    ):
        return jsonify({
            "success": False,
            "error": "No MRI file selected."
        }), 400

    allowed_extensions = {
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff",
        "webp"
    }

    original_filename = secure_filename(
        uploaded_file.filename
    )

    extension = (
        original_filename.rsplit(
            ".",
            1
        )[-1].lower()
        if "." in original_filename
        else ""
    )

    if extension not in allowed_extensions:
        return jsonify({
            "success": False,
            "error": (
                "Unsupported image format. "
                "Use PNG, JPG, JPEG, BMP, TIFF or WEBP."
            )
        }), 400

    unique_name = (
        f"unet_input_"
        f"{uuid.uuid4().hex[:12]}."
        f"{extension}"
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    try:
        uploaded_file.save(
            image_path
        )

        threshold_value = request.form.get(
            "threshold",
            "0.50"
        )

        try:
            threshold_value = float(
                threshold_value
            )
        except Exception:
            threshold_value = 0.50

        threshold_value = float(
            np.clip(
                threshold_value,
                0.01,
                0.99
            )
        )

        pixel_spacing = (
            _get_pixel_spacing_from_request()
        )

        result = run_unet_analysis(
            image_path,
            threshold=threshold_value,
            clean_small_regions=True,
            pixel_spacing=pixel_spacing
        )

        result = (
            convert_result_paths_to_urls(
                app,
                result
            )
        )

        result["input_image"] = url_for(
            "static",
            filename=(
                "uploads/"
                + unique_name
            )
        )

        result["pixel_spacing"] = (
            {
                "x_mm_per_pixel":
                    pixel_spacing[0],
                "y_mm_per_pixel":
                    pixel_spacing[1]
            }
            if pixel_spacing is not None
            else None
        )

        return jsonify(result)

    except Exception as e:
        print()
        print("=" * 72)
        print("U-NET ENDPOINT ERROR")
        print("=" * 72)
        print(str(e))
        print(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(e),
            "type": e.__class__.__name__
        }), 500


# ================================================================
# REGISTER U-NET WITH EXISTING FLASK APP
# ================================================================

def register_unet(app):
    try:
        load_unet_model()

        try:
            build_feature_model()
        except Exception as feature_error:
            print(
                "Feature model initialization warning:",
                feature_error
            )

    except Exception as e:
        print()
        print("=" * 72)
        print("WARNING: U-NET MODEL NOT LOADED AT STARTUP")
        print("=" * 72)
        print(str(e))
        print(
            "The application can still start. "
            "U-Net will attempt to load when used."
        )

    app.add_url_rule(
        "/unet/analyze",
        endpoint="unet_analyze",
        view_func=lambda:
            unet_analyze_endpoint(app),
        methods=["POST"]
    )

    print()
    print("=" * 72)
    print("NEUROSCAN U-NET MODULE REGISTERED")
    print("=" * 72)

    print(
        "Endpoint: POST /unet/analyze"
    )

    print()
    print("Features:")
    print("  ✓ Raw U-Net probability map")
    print("  ✓ Probability heatmap")
    print("  ✓ Probability overlay")
    print("  ✓ Raw U-Net segmentation mask")
    print("  ✓ Image-guided tumor refinement")
    print("  ✓ Brain ROI protection")
    print("  ✓ Seed-connected lesion expansion")
    print("  ✓ Brain-area plausibility guard (raw + final mask)")
    print("  ✓ Binary segmentation")
    print("  ✓ Threshold 0.50")
    print("  ✓ Threshold 0.30")
    print("  ✓ Threshold 0.20")
    print("  ✓ Threshold 0.10")
    print("  ✓ Conservative morphology cleanup")
    print("  ✓ Probability-guided components")
    print("  ✓ Tumor boundary")
    print("  ✓ Tumor region crop")
    print("  ✓ Tumor area")
    print("  ✓ Tumor area percentage")
    print("  ✓ Equivalent diameter in pixels")
    print("  ✓ Largest lesion dimension in pixels")
    print("  ✓ Bounding box")
    print("  ✓ Tumor centroid")
    print("  ✓ Connected components")
    print("  ✓ Optional physical area")
    print("  ✓ Input-gradient saliency")
    print("  ✓ Saliency heatmap")
    print("  ✓ Shallow feature activation")
    print("  ✓ Middle feature activation")
    print("  ✓ Deep feature activation")

    print("=" * 72)

    return app


# ================================================================
# DIRECT TEST MODE
# ================================================================

if __name__ == "__main__":
    print()
    print("=" * 72)
    print("NEUROSCAN U-NET FEATURES MODULE")
    print("=" * 72)

    print()
    print(
        "This file is intended to be imported by app.py."
    )

    print()
    print(
        f"Model expected at:\n{MODEL_PATH}"
    )

    print()
    print(
        "Image-guided refinement:"
        f" {REFINEMENT_ENABLED}"
    )

    print()
    print(
        "Plausibility guard: max "
        f"{MAX_PLAUSIBLE_LESION_FRACTION_OF_BRAIN * 100:.0f}% "
        "of estimated brain area"
    )

    print()
    print(
        "Example integration:"
    )

    print()
    print(
        "from unet_features import register_unet"
    )

    print(
        "register_unet(app)"
    )

    print()
    print("=" * 72)
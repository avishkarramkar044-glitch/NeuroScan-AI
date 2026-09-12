import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# ============================================================
# PATHS & CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# VGG16 Fallback Hierarchy
PRIMARY_VGG_PATH = os.path.join(BASE_DIR, "colab_vgg16_upgraded_best.keras")
FINETUNED_PATH = os.path.join(BASE_DIR, "Backup_Models", "brain_tumor_finetuned_best.keras")
DEFAULT_VGG_PATH = os.path.join(BASE_DIR, "brain_tumor_final_best.keras")

if os.path.exists(PRIMARY_VGG_PATH):
    MODEL_PATH = PRIMARY_VGG_PATH
elif os.path.exists(FINETUNED_PATH):
    MODEL_PATH = FINETUNED_PATH
else:
    MODEL_PATH = DEFAULT_VGG_PATH

CATEGORIES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

IMG_SIZE = 224

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_tumor(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        return

    print(f"Loading model from: {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    print(f"Processing image: {image_path}...")
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocessing strictly matching VGG16 input specifications
    processed_input = vgg16_preprocess(img_batch.astype(np.float32))

    print("Running inference...")
    predictions = model.predict(processed_input, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CATEGORIES[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    print("\n" + "="*40)
    print("      NEUROSCAN AI DIAGNOSTIC RESULTS")
    print("="*40)
    print(f"Predicted Class : {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence      : {confidence:.2f}%\n")
    print("Class Probabilities:")
    print("-" * 40)
    for cat, prob in zip(CATEGORIES, predictions):
        print(f" - {cat.replace('_', ' ').title():<20}: {prob * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = input("Enter path to MRI image file: ").strip('"').strip("'")

    predict_tumor(test_image_path)
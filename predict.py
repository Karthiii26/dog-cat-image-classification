"""
predict.py
==========
Inference and prediction module for Dog-Cat Image Classification.

Features:
- CLI inference on single images or entire folders.
- Strict input validation (detects nonexistent files, zero-byte files, corrupted images, unsupported extensions).
- Preprocesses input image: resizes to (128, 128, 3) and normalizes pixel values to [0, 1].
- Formats output: Predicted Class ('Dog' or 'Cat'), Confidence percentage, Raw Sigmoid output.
- Returns structured prediction dictionary for programmatic integration (e.g. Streamlit app).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union

# Suppress minor TensorFlow backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

DEFAULT_MODEL_PATH = Path("models/dog_cat_cnn.keras")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Cache loaded model globally to optimize multiple calls
_LOADED_MODEL = None
_LOADED_MODEL_PATH = None


def load_classification_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Loads and caches the trained Keras CNN model.
    """
    global _LOADED_MODEL, _LOADED_MODEL_PATH
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Please train the model using 'python train.py' first!"
        )

    if _LOADED_MODEL is None or _LOADED_MODEL_PATH != str(model_path):
        print(f"[INFO] Loading CNN model from: {model_path}...")
        _LOADED_MODEL = tf.keras.models.load_model(str(model_path))
        _LOADED_MODEL_PATH = str(model_path)

    return _LOADED_MODEL


def validate_image_file(image_path: Union[str, Path]) -> Path:
    """
    Performs comprehensive validation of an image file:
    - Path existence and file type
    - Non-zero file size
    - Valid image extension
    - File readability and decoding via PIL
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {path}")

    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Supported formats: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}"
        )

    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"File is corrupted or cannot be decoded as an image: {e}")

    return path


def preprocess_image(image_input: Union[str, Path, Image.Image], target_size=(128, 128)) -> np.ndarray:
    """
    Preprocesses an image for the 3-layer CNN:
    1. Loads or accepts PIL Image.
    2. Converts to 3-channel RGB.
    3. Resizes to target_size (128, 128) using high-quality Lanczos resampling.
    4. Converts to float32 NumPy array with shape (1, 128, 128, 3).
    Note: The model includes Rescaling(1./255), so pixel values in range [0, 255] are accepted.
    """
    if isinstance(image_input, (str, Path)):
        validated_path = validate_image_file(image_input)
        with Image.open(validated_path) as img:
            img_rgb = img.convert("RGB")
    elif isinstance(image_input, Image.Image):
        img_rgb = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    resized_img = img_rgb.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    # Expand batch dimension -> (1, 128, 128, 3)
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array


def predict_dog_or_cat(
    image_input: Union[str, Path, Image.Image],
    model: tf.keras.Model = None,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Executes binary classification on an image.

    Returns:
        dict: {
            "predicted_class": "Dog" or "Cat",
            "confidence": float (percentage, e.g. 97.45),
            "dog_probability": float (0.0 to 1.0),
            "cat_probability": float (0.0 to 1.0),
            "raw_output": float
        }
    """
    if model is None:
        model = load_classification_model(model_path)

    batch_array = preprocess_image(image_input, target_size=(128, 128))

    raw_output = float(model.predict(batch_array, verbose=0)[0][0])
    dog_prob = raw_output
    cat_prob = 1.0 - raw_output

    if dog_prob >= 0.5:
        predicted_class = "Dog"
        confidence = dog_prob * 100.0
    else:
        predicted_class = "Cat"
        confidence = cat_prob * 100.0

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "dog_probability": round(dog_prob, 4),
        "cat_probability": round(cat_prob, 4),
        "raw_output": round(raw_output, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Dog vs Cat CNN Image Classifier (CLI Inference)")
    parser.add_argument("--image", type=str, default=None, help="Path to the image to classify")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing images to classify in batch")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model (.keras)")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found at: {model_path}")
        print("Please train the model first by running:\n  python train.py")
        sys.exit(1)

    try:
        model = load_classification_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    if args.image:
        try:
            print(f"\n[INFO] Classifying image: {args.image}")
            result = predict_dog_or_cat(args.image, model=model)
            print("=" * 50)
            print(f"  PREDICTED CLASS : {result['predicted_class'].upper()}")
            print(f"  CONFIDENCE      : {result['confidence']:.2f}%")
            print(f"  Dog Probability : {result['dog_probability']:.4f}")
            print(f"  Cat Probability : {result['cat_probability']:.4f}")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            sys.exit(1)

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"[ERROR] Specified directory does not exist: {dir_path}")
            sys.exit(1)

        images = [p for p in dir_path.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        print(f"[INFO] Found {len(images)} images in {dir_path}. Running batch inference...")
        print("-" * 65)
        print(f"{'Filename':<35} | {'Class':<6} | {'Confidence':<10} | {'Raw':<6}")
        print("-" * 65)

        for img_path in images:
            try:
                res = predict_dog_or_cat(img_path, model=model)
                print(f"{img_path.name:<35} | {res['predicted_class']:<6} | {res['confidence']:>8.2f}% | {res['raw_output']:>6.4f}")
            except Exception as e:
                print(f"{img_path.name:<35} | ERROR  | {str(e)[:25]}")
        print("-" * 65)

    else:
        # Check if sample_images folder has samples
        sample_dir = Path("sample_images")
        if sample_dir.exists() and any(sample_dir.iterdir()):
            print("[INFO] No image specified. Demonstrating inference on available sample images:")
            for sample in sorted(sample_dir.glob("*.*")):
                if sample.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    try:
                        res = predict_dog_or_cat(sample, model=model)
                        print(f"  [{sample.name}] -> {res['predicted_class']} ({res['confidence']:.2f}%)")
                    except Exception as e:
                        print(f"  [{sample.name}] -> Error: {e}")
        else:
            print("[USAGE] Please specify an image using: python predict.py --image <path_to_image>")


if __name__ == "__main__":
    main()

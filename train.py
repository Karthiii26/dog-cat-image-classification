"""
train.py
========
Model training, evaluation, and visualization pipeline for:
"Development of a Convolutional Neural Network (CNN) for Dog-Cat Image Classification Using Three Convolutional Layers"

Workflow:
1. Environment configuration (GPU auto-detection vs CPU fallback, reproducible seeds).
2. Data loading from dataset/train, dataset/val, and dataset/test with batching and prefetching.
3. Model instantiation using 3-layer CNN from model.py.
4. Model training with callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).
5. Independent evaluation on the unseen test dataset (10% split).
6. Visualization generation:
   - Training vs Validation Accuracy curve
   - Training vs Validation Loss curve
   - Confusion Matrix (Heatmap)
   - Classification Report (Precision, Recall, F1-Score, Support)
7. Exporting best model to models/dog_cat_cnn.keras and history to models/training_history.json.
"""

import os
import sys
import io
import json
import random
import argparse
from pathlib import Path

# Fix Windows console unicode encoding issues with Keras progress bars
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress minor TensorFlow backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from model import build_cnn_model

RANDOM_SEED = 42


def set_reproducible_seeds(seed: int = RANDOM_SEED):
    """
    Sets reproducible random seeds across Python, NumPy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] Reproducible random seed set to: {seed}")


def configure_compute_device():
    """
    Detects and configures hardware acceleration (GPU if available, CPU fallback).
    """
    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 60)
    print("HARDWARE & COMPUTE CONFIGURATION")
    print("=" * 60)
    if gpus:
        print(f"[INFO] GPU(s) detected ({len(gpus)} device(s)):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"    Memory growth setting note: {e}")
        device_name = "GPU"
    else:
        print("[INFO] No GPU found. Training will proceed on CPU using multithreading.")
        device_name = "CPU"
    print(f"[INFO] Active execution device: {device_name}")
    print("=" * 60 + "\n")
    return device_name


def load_datasets(dataset_dir: Path, image_size=(128, 128), batch_size=32):
    """
    Loads train, validation, and test datasets using image_dataset_from_directory.
    Label mapping: 0 -> Cat, 1 -> Dog (alphabetical: 'cats'=0, 'dogs'=1).
    """
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(
                f"Directory '{d}' not found. Please run 'python setup_dataset.py' first!"
            )

    print(f"[INFO] Loading datasets from '{dataset_dir}' (Image Size: {image_size}, Batch Size: {batch_size})...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["cats", "dogs"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["cats", "dogs"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["cats", "dogs"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    # Performance optimization: prefetching
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def plot_training_history(history_dict: dict, save_path: Path):
    """
    Generates and saves professional graphs for:
    1. Training and Validation Accuracy
    2. Training and Validation Loss
    """
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.suptitle(
        "Dog-Cat CNN Training & Validation Performance (3-Layer Architecture)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # 1. Accuracy Plot
    ax1.plot(epochs, history_dict["accuracy"], "o-", color="#2563EB", linewidth=2, label="Training Accuracy")
    ax1.plot(epochs, history_dict["val_accuracy"], "s--", color="#10B981", linewidth=2, label="Validation Accuracy")
    ax1.set_title("Model Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0.45, 1.02])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Annotate Best Val Accuracy
    best_val_acc = max(history_dict["val_accuracy"])
    best_val_epoch = history_dict["val_accuracy"].index(best_val_acc) + 1
    ax1.annotate(
        f"Peak: {best_val_acc:.2%}\n(Epoch {best_val_epoch})",
        xy=(best_val_epoch, best_val_acc),
        xytext=(best_val_epoch, best_val_acc - 0.08),
        arrowprops=dict(facecolor="#10B981", shrink=0.08, width=1.5, headwidth=6),
        ha="center",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=1),
    )

    # 2. Loss Plot
    ax2.plot(epochs, history_dict["loss"], "o-", color="#DC2626", linewidth=2, label="Training Loss")
    ax2.plot(epochs, history_dict["val_loss"], "s--", color="#F59E0B", linewidth=2, label="Validation Loss")
    ax2.set_title("Model Loss (Binary Cross-Entropy) vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Training history plot saved to: {save_path}")


def evaluate_and_generate_metrics(model: tf.keras.Model, test_ds, plots_dir: Path):
    """
    Evaluates model on the test dataset, generates confusion matrix heatmap,
    and produces classification report.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("TEST DATASET EVALUATION & METRICS GENERATION")
    print("=" * 60)

    # Accumulate true labels and model predictions
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy().flatten())
        y_pred_probs_list.extend(preds.flatten())

    y_true = np.array(y_true_list, dtype=int)
    y_probs = np.array(y_pred_probs_list)
    y_pred = (y_probs >= 0.5).astype(int)

    # Test Accuracy & Loss
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n[RESULTS] Final Test Evaluation:")
    print(f"  - Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  - Test Loss     : {test_loss:.4f}")

    # Classification Report
    target_names = ["Cat (0)", "Dog (1)"]
    report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nClassification Report:\n" + report_str)

    report_path = plots_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("CNN Dog-Cat Image Classification - Test Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n")
        f.write(f"Test Loss    : {test_loss:.4f}\n\n")
        f.write(report_str)
    print(f"[SUCCESS] Classification report written to: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Confusion Matrix - Test Dataset", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Confusion matrix plot saved to: {cm_path}")

    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Train 3-Layer CNN for Dog-Cat Classification")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing train/val/test splits")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam optimizer learning rate")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained model")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to save evaluation plots")

    args = parser.parse_args()

    set_reproducible_seeds(RANDOM_SEED)
    configure_compute_device()

    dataset_dir = Path(args.dataset_dir)
    models_dir = Path(args.models_dir)
    plots_dir = Path(args.plots_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_ds, val_ds, test_ds = load_datasets(
        dataset_dir=dataset_dir,
        image_size=(128, 128),
        batch_size=args.batch_size,
    )

    # 2. Build Model
    print("[INFO] Constructing CNN model with exactly 3 Convolutional layers...")
    model = build_cnn_model(input_shape=(128, 128, 3), learning_rate=args.lr)
    model.summary()

    # 3. Callbacks
    best_model_path = models_dir / "dog_cat_cnn.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 4. Train Model
    print("\n" + "=" * 60)
    print(f"TRAINING CNN MODEL (Epochs: {args.epochs}, Batch Size: {args.batch_size})")
    print("=" * 60)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # 5. Save History Data
    history_dict = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    history_json_path = models_dir / "training_history.json"
    with open(history_json_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[SUCCESS] Training history logged to: {history_json_path}")

    # 6. Plot Training Curves
    plot_training_history(history_dict, plots_dir / "training_history.png")

    # 7. Evaluate on Test Dataset
    # Load the best saved checkpoint to evaluate optimal weights
    if best_model_path.exists():
        print(f"[INFO] Reloading best saved model from {best_model_path} for evaluation...")
        eval_model = tf.keras.models.load_model(str(best_model_path))
    else:
        eval_model = model

    test_metrics = evaluate_and_generate_metrics(eval_model, test_ds, plots_dir)
    test_metrics_path = models_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  - Model file: {best_model_path}")
    print(f"  - History plot: {plots_dir / 'training_history.png'}")
    print(f"  - Confusion matrix: {plots_dir / 'confusion_matrix.png'}")
    print(f"  - Classification report: {plots_dir / 'classification_report.txt'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

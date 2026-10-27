# Dog vs Cat Image Classification using CNN

A deep learning project that classifies images as **Dog** or **Cat** using a custom Convolutional Neural Network built with exactly three convolutional layers. Includes a training pipeline, evaluation metrics, and a Streamlit web app for live predictions.

---

## Model Architecture

The CNN is built with TensorFlow/Keras and follows this exact structure:

| Layer | Configuration |
|---|---|
| Input | 128 x 128 x 3 (RGB) |
| Rescaling | Normalizes pixels to [0, 1] |
| Conv2D + MaxPooling | 32 filters, 3x3 kernel, ReLU |
| Conv2D + MaxPooling | 64 filters, 3x3 kernel, ReLU |
| Conv2D + MaxPooling | 128 filters, 3x3 kernel, ReLU |
| Flatten | - |
| Dense | 128 neurons, ReLU |
| Dropout | 0.5 |
| Output | 1 neuron, Sigmoid (0 = Cat, 1 = Dog) |

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr = 0.001)
- **Metric:** Accuracy

---

## Project Structure

```
coe-project2/
├── model.py            # CNN architecture definition
├── train.py            # Training pipeline with callbacks and evaluation
├── predict.py          # CLI inference on single images or folders
├── setup_dataset.py    # Downloads and splits the dataset (80/10/10)
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
├── dataset/            # Created by setup_dataset.py
│   ├── train/
│   ├── val/
│   └── test/
├── models/             # Saved model and training history
├── plots/              # Training curves and confusion matrix
└── sample_images/      # Sample images for quick testing
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and prepare the dataset

```bash
python setup_dataset.py --download
```

For a quicker run with fewer images:
```bash
python setup_dataset.py --download --max-images-per-class 1000
```

### 3. Train the model

```bash
python train.py --epochs 15 --batch-size 32
```

Training automatically saves the best model to models/dog_cat_cnn.keras, generates accuracy/loss plots, and produces a confusion matrix and classification report.

### 4. Run the web app

```bash
streamlit run app.py
```

Upload any dog or cat image to get a prediction with confidence score.

### 5. CLI prediction

```bash
python predict.py --image path/to/image.jpg
python predict.py --dir sample_images/
```

---

## Training Features

- **Callbacks:** ModelCheckpoint, EarlyStopping (patience=6), ReduceLROnPlateau (factor=0.5)
- **Reproducibility:** Fixed seed (42) across Python, NumPy, and TensorFlow
- **GPU support:** Automatically uses GPU if available, falls back to CPU
- **Outputs:** plots/training_history.png, plots/confusion_matrix.png, plots/classification_report.txt

---

## Dataset

Uses the public Dogs vs. Cats dataset (~25,000 images). The setup script handles downloading, integrity checks, and an 80/10/10 stratified split into train/val/test sets.

---

## Running on Google Colab

1. Open colab.research.google.com and switch to a **T4 GPU** runtime
2. Clone the repo and install dependencies:
```bash
!git clone https://github.com/Karthiii26/dog-cat-image-classification.git
%cd dog-cat-image-classification
!pip install -r requirements.txt
```
3. Run setup and training:
```bash
!python setup_dataset.py --download
!python train.py --epochs 20 --batch-size 32
```

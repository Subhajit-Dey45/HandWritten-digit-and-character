# Handwritten Digit and Character Recognition System

A deep learning project built with TensorFlow and Keras to recognize handwritten digits (`0`–`9`) from grayscale image data. Developed and tested on the Kaggle Digit Recognizer (MNIST) dataset, this project demonstrates a progressive neural network workflow—starting from a baseline single-layer perceptron and improving accuracy through a multi-layer neural network—with confusion matrix visualizations for error analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Workflow & Model Architecture](#workflow--model-architecture)
- [Results & Performance Comparison](#results--performance-comparison)
- [Getting Started](#getting-started)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

Optical character and digit recognition is a foundational task in computer vision with applications in automated form processing, postal code sorting, and bank check digitization. This repository implements an end-to-end classification pipeline that:

1. Loads and inspects flattened pixel data from CSV files.
2. Normalizes pixel intensities and visualizes sample handwritten digits.
3. Splits the dataset using stratified sampling to preserve class balance.
4. Trains and compares two Keras Sequential neural network architectures.
5. Evaluates model performance using accuracy, loss, and confusion matrix heatmaps.

> **Note:** The current notebook implements handwritten **digit** recognition (`0`–`9`). The architecture is structured so it can be easily extended to full alphanumeric **character** recognition (e.g., via the EMNIST dataset).

---

## Dataset

This project uses the **Kaggle Digit Recognizer** dataset (derived from MNIST):

- **`train.csv`**: `42,000` labeled samples with `785` columns (`label` + `pixel0` through `pixel783`).
- **`test.csv`**: `28,000` unlabeled samples with `784` pixel columns.
- **Image Format**: Each row represents a flattened `$28 \times 28$` grayscale image (`784` features) with pixel values ranging from `0` to `255`.

You can download the dataset directly from Kaggle: [Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer/data).

---

## Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow, Keras
- **Data Manipulation:** NumPy, Pandas
- **Machine Learning Utilities:** Scikit-learn (`train_test_split`)
- **Visualization:** Matplotlib, Seaborn

---

## Workflow & Model Architecture

### 1. Data Preprocessing
- Separated target labels (`y`) and pixel features (`x`) into NumPy arrays.
- Normalized pixel values to the `[0, 1]` range by dividing by `255`.
- Performed a stratified 70/30 train-test split (`test_size=0.3`, `random_state=20`, `stratify=y`), yielding `29,400` training samples and `12,600` test samples.

### 2. Model 1 — Baseline Neural Network
A single-layer dense neural network mapping the 784 input pixels directly to 10 output classes:
- **Layer 1 (Output):** `Dense(10, input_shape=(784,), activation='sigmoid')`
- **Optimizer:** `adam`
- **Loss Function:** `sparse_categorical_crossentropy`
- **Training:** `5` epochs

### 3. Model 2 — Neural Network with Hidden Layer
An enhanced architecture adding a ReLU hidden layer to capture non-linear spatial patterns:
- **Layer 1 (Hidden):** `Dense(100, input_shape=(784,), activation='relu')`
- **Layer 2 (Output):** `Dense(10, activation='sigmoid')`
- **Optimizer:** `adam`
- **Loss Function:** `sparse_categorical_crossentropy`
- **Training:** `5` epochs

### 4. Evaluation
- Predictions are converted to class indices using `np.argmax`.
- Confusion matrices are computed with `tf.math.confusion_matrix` and plotted as annotated Seaborn heatmaps to inspect per-digit classification accuracy and misclassifications.

---

## Results & Performance Comparison

Adding a single 100-unit ReLU hidden layer significantly reduced test loss and boosted test accuracy by **+4.47%** over the baseline model within just 5 epochs:

| Model | Architecture | Epochs | Training Accuracy (Final Epoch) | Test Loss | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline)** | `Dense(10, sigmoid)` | 5 | 92.18% | 0.2792 | **92.23%** |
| **Model 2 (Hidden Layer)** | `Dense(100, relu)` -> `Dense(10, sigmoid)` | 5 | 98.01% | 0.1079 | **96.70%** |

---

## Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn jupyter
```

### Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Download the dataset:**
   Place `train.csv` and `test.csv` from the Kaggle Digit Recognizer competition into a local `data/` directory, and update the file paths in the notebook from `/kaggle/input/digit-recognizer/` to your local path:
   ```python
   data = pd.read_csv("data/train.csv")
   test_data = pd.read_csv("data/test.csv")
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open the notebook file and run all cells sequentially.

### Running on Kaggle
Upload the notebook directly to [Kaggle](https://www.kaggle.com/), attach the **Digit Recognizer** competition dataset, and click **Run All**. No path changes are required.

---

## Future Improvements

- **Expand to Full Character Recognition:** Train on the [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset to classify uppercase and lowercase letters (`A`–`Z`, `a`–`z`) alongside digits (`0`–`9`).
- **Convolutional Neural Networks (CNNs):** Reshape inputs to `($28 \times 28 \times 1$)` and introduce `Conv2D`, `MaxPooling2D`, and `Dropout` layers to push accuracy above 99%.
- **Output Activation Optimization:** Replace `sigmoid` in the final layer with `softmax` for normalized multi-class probability distributions.
- **Test Set Submission:** Generate predictions on `test_data` and export a formatted `submission.csv` for Kaggle leaderboard evaluation.
- **Interactive Drawing Interface:** Build a lightweight web app (e.g., with Streamlit or Gradio) allowing users to draw digits/characters in real time for live inference.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

# Multimodal Parkinson's Disease Detection (Voice + Image)

## Overview

This project was built to detect Parkinson's disease using two completely different types of input data — voice signal features and hand-drawn spiral/wave sketches. It's not a single unified model; it's two separate pipelines that run one after another in the same script, each trained and evaluated independently.

The voice side uses classical machine learning models on numerical features extracted from speech recordings. The image side uses MobileNetV2 (a pretrained CNN) for feature extraction, both as a fine-tuned classifier and as an embedding generator for classical ML models. The goal was to compare how well traditional ML and transfer learning perform on this kind of medical classification problem when the dataset available is fairly small.

This was done as part of my B.Tech coursework/project work, so the dataset sizes and scope are limited compared to what you'd see in an actual research paper — but the pipeline itself is complete and functional end-to-end.

---

## Features

- Two independent detection pipelines (voice-based and image-based) in a single script
- Automatic skipping of a pipeline if its input data isn't found, instead of the script crashing
- Multiple models trained and compared for each modality, with the best one selected automatically based on F1-score
- MobileNetV2 used both for a fine-tuned CNN classifier and as a frozen feature extractor for classical ML
- Confusion matrix, ROC curve, and Precision-Recall curve generated and saved for every model trained
- All trained models saved to disk using `joblib` (sklearn models) and native Keras format (CNNs)
- A small random-sampling check at the end that verifies the best model on a few real test examples

---

## Technologies Used

| Category | Tools/Libraries |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras (MobileNetV2) |
| Classical ML | scikit-learn (SVM, kNN, Logistic Regression, MLP) |
| Data Handling | pandas, NumPy |
| Visualization | Matplotlib |
| Model Persistence | joblib, Keras `.keras` format |

---

## Project Structure

```
├── parkinsons.csv              # voice dataset (input)
├── PD_Dataset.zip               # zipped image dataset (input)
├── PD_Dataset/                  # extracted image dataset
│   ├── spiral/
│   │   ├── training/
│   │   │   ├── healthy/
│   │   │   └── parkinson/
│   │   └── testing/
│   │       ├── healthy/
│   │       └── parkinson/
│   └── wave/
│       ├── training/
│       │   ├── healthy/
│       │   └── parkinson/
│       └── testing/
│           ├── healthy/
│           └── parkinson/
│
├── models/                      # all trained models get saved here
│   ├── voice_scaler.joblib
│   ├── voice_SVM.joblib
│   ├── voice_kNN.joblib
│   ├── voice_Logistic.joblib
│   ├── voice_MLP.joblib
│   ├── voice_best.joblib
│   ├── spiral_cnn.keras
│   ├── wave_cnn.keras
│   ├── spiral_emb_scaler.joblib
│   ├── wave_emb_scaler.joblib
│   ├── spiral_svc_emb.joblib
│   ├── spiral_knn_emb.joblib
│   ├── wave_svc_emb.joblib
│   └── wave_knn_emb.joblib
│
└── outputs/                     # all metrics + plots get saved here
    ├── voice_test_metrics.csv
    ├── image_test_metrics.csv
    └── confusion matrix / ROC / PR curve PNGs for every model
```

> Note: the `results` folder included alongside this repo is just a small set of sample screenshots and plots taken from a run of the script — it isn't the full `outputs/` folder that gets generated when you actually run the project.

---

## Dataset Description

**Voice dataset (`parkinsons.csv`)**
- 195 rows, 24 columns total
- 22 numeric feature columns are used for training (things like jitter, shimmer, HNR — standard voice signal measures)
- Label column is `status` (0 = healthy, 1 = Parkinson's). The code also checks for alternate label names like `label`, `target`, `diagnosis`, or `class` in case the column is named differently

**Image dataset (`PD_Dataset.zip`)**
- Contains two modalities: `spiral` and `wave` drawings
- Each modality has its own `training/` and `testing/` folder, with `healthy/` and `parkinson/` subfolders inside
- Around 72 training images and 30 testing images per modality

If `PD_Dataset/` doesn't already exist, the script extracts it automatically from `PD_Dataset.zip`.

---

## Voice Detection Pipeline

1. **Load the CSV** and rename the label column to `status` if needed.
2. **Keep only numeric columns** as features — this avoids errors from any non-numeric columns that might be present in the raw file.
3. **Handle missing values** by filling any NaNs with the column mean.
4. **Scale the features** using `StandardScaler` (fitted on the training set, saved as `voice_scaler.joblib`).
5. **Split the data**: 80% train+val / 20% test, then the train+val portion is split again 80/20 into train and validation. This resulted in 124 training samples, 32 validation samples, and 39 test samples.
6. **Train four different classifiers**, each tuned with `GridSearchCV` and wrapped in `CalibratedClassifierCV` for better probability outputs:
   - Support Vector Machine (RBF kernel)
   - k-Nearest Neighbors
   - Logistic Regression
   - Multi-Layer Perceptron (MLP)
7. **Evaluate all four on the test set** and pick the best one based on F1-score.

---

## Image Detection Pipeline

The same steps are repeated separately for both the `spiral` and `wave` modalities.

1. **Load images** at 224×224 resolution using `image_dataset_from_directory`.
2. **Build a CNN using MobileNetV2** as the base:
   - MobileNetV2 is loaded with `imagenet` weights, `include_top=False`, and kept **frozen** (no fine-tuning of the base layers)
   - On top of it: `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(128, activation='relu')` → `Dense(1, activation='sigmoid')`
   - Trained for 6 epochs using the Adam optimizer (learning rate 1e-4) and binary cross-entropy loss
3. **Extract embeddings** from the frozen MobileNetV2 base separately (before the added dense layers) for both training and testing images.
4. **Scale the embeddings** using `StandardScaler`, then train two more classifiers directly on these embeddings:
   - SVM (RBF kernel)
   - kNN (`n_neighbors=5`)

So for each modality, three models end up being compared: the CNN itself, an SVM trained on MobileNetV2 embeddings, and a kNN trained on MobileNetV2 embeddings.

**Note on MobileNetV2:** this project does not use a custom-built CNN architecture anywhere. MobileNetV2 (pretrained on ImageNet) is the only convolutional backbone used, and it is kept frozen throughout — it's used purely as a feature extractor rather than being fine-tuned end-to-end.

---

## Model Architectures and Training Process

**Voice models** — all trained on scaled numeric features, tuned with small `GridSearchCV` grids (since the dataset is small):

| Model | Grid Searched |
|---|---|
| SVM (RBF) | `C = [1, 5]` |
| kNN | `n_neighbors = [3, 5, 7]` |
| Logistic Regression | `C = [0.5, 1, 5]` |
| MLP | `hidden_layer_sizes = [(50,), (100,)]`, `alpha = [1e-4, 1e-3]` |

**Image CNN architecture:**

```
Input(224, 224, 3)
  → Rescaling(1/127.5, offset=-1)
  → MobileNetV2 (frozen, imagenet weights, no top)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(128, activation='relu')
  → Dense(1, activation='sigmoid')
```

Trained for 6 epochs, Adam optimizer at 1e-4 learning rate, binary cross-entropy loss.

**Embedding-based models:** SVM (RBF, tuned over `C = [1, 5]`) and kNN (`n_neighbors=5`), both trained on the pooled MobileNetV2 output vector instead of raw pixels.

---

## Evaluation Metrics

The following metrics were calculated for **every model** in both pipelines:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

Precision-Recall curves are also plotted for each model (with AUPR shown on the plot itself), along with confusion matrices.

---

## Results

**Voice pipeline — test set results (sorted by F1-score):**

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| MLP | 0.9487 | 0.9655 | 0.9655 | 0.9655 | 0.8655 | 0.9862 |
| kNN | 0.9231 | 0.9333 | 0.9655 | 0.9492 | 0.7934 | 0.9569 |
| SVM | 0.8974 | 0.9310 | 0.9310 | 0.9310 | 0.7310 | 0.9379 |
| Logistic Regression | 0.8718 | 0.9286 | 0.8966 | 0.9123 | 0.6759 | 0.9448 |

**Best voice model:** MLP, based on highest F1-score on the test set.

**Image pipeline — test set results (sorted by F1-score):**

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| wave SVM (embeddings) | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.8667 | 0.9467 |
| wave CNN | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 0.7333 | 0.9333 |
| spiral kNN (embeddings) | 0.8333 | 0.8571 | 0.8000 | 0.8276 | 0.6682 | 0.8956 |
| spiral SVM (embeddings) | 0.8000 | 0.7368 | 0.9333 | 0.8235 | 0.6225 | 0.8800 |
| wave kNN (embeddings) | 0.8000 | 0.8462 | 0.7333 | 0.7857 | 0.6054 | 0.8600 |
| spiral CNN | 0.7000 | 0.7143 | 0.6667 | 0.6897 | 0.4009 | 0.7911 |

**Best model — spiral:** kNN on embeddings (F1 = 0.8276)
**Best model — wave:** SVM on embeddings (F1 = 0.9333)

One thing worth pointing out: for both spiral and wave images, the fine-tuned CNN performed worse than the classical models trained on MobileNetV2 embeddings. This is most likely because the CNN only had 6 epochs and around 72 training images to learn from, which isn't a lot for training the added dense layers, while SVM/kNN can work reasonably well on strong pretrained features even with fewer samples.

All of the numbers above are taken directly from the metrics generated by running the script — nothing here is estimated.

---

## Output Files Generated

**Metrics (CSV):**
- `outputs/voice_test_metrics.csv`
- `outputs/image_test_metrics.csv`

**Plots (PNG, generated for every model):**
- Confusion matrix
- ROC curve
- Precision-Recall curve

**Saved Models:**
- 4 voice classifiers + scaler + a copy of the best voice model
- 2 CNNs (one per image modality, `.keras` format)
- 4 embedding-based classifiers (SVM + kNN for each modality) + 2 embedding scalers

---

## Installation Steps

Clone the repository and install the required libraries:

```bash
git clone <repository-url>
cd <repository-folder>
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

---

## How to Run the Project

1. Place `parkinsons.csv` and `PD_Dataset.zip` in the project's root folder.
2. Run the script (either as a `.py` file or inside a Jupyter/Colab notebook).
3. The script will:
   - Extract `PD_Dataset.zip` automatically if `PD_Dataset/` isn't already present
   - Run the voice pipeline if `parkinsons.csv` is found
   - Run the image pipeline if the image folders are found
   - Save all models to `models/` and all metrics/plots to `outputs/`

If either input is missing, that part of the pipeline is simply skipped with a printed message instead of throwing an error.

---

## Future Improvements

- The dataset used here is quite small (39 test samples for voice, 30 test images per modality), so the results can vary a fair bit between different runs or splits. Testing on a larger dataset would give a better idea of real performance.
- MobileNetV2 is completely frozen in this project. Fine-tuning at least the last few layers might help the CNN specifically, since it currently performs worse than the embedding-based models in both image modalities.
- The voice and image pipelines are currently independent — they don't share information. A proper multimodal model that combines both voice and image features together could be explored as a next step.
- Cross-validation instead of a single train/test split would give more reliable performance estimates.

---

## License

This project is for academic and research purposes.

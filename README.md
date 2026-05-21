# Human Activity Recognition using Hybrid Deep Learning Architectures

## Project Overview

This project investigates Human Activity Recognition (HAR) using deep learning architectures that combine Convolutional Neural Networks (CNNs), Gated Recurrent Units (GRUs), and Autoencoder (AE) components. The primary objective is to classify human activities from multichannel smartphone sensor signals while analyzing the contribution of different architectural blocks through ablation experiments.

The proposed study evaluates multiple architectures with increasing complexity, beginning from a CNN-only baseline and progressing toward hybrid models integrating recurrent and representation-learning components. The experiments focus not only on classification accuracy, but also on understanding the effect of temporal modeling, feature compression, and regularization strategies on HAR performance.

The project was developed for the Deep Learning with Python course.

---

# Dataset

## UCI Human Activity Recognition (HAR) Dataset

The experiments utilize the UCI Human Activity Recognition Using Smartphones Dataset, originally introduced by Anguita et al. The dataset contains motion and orientation sensor signals collected from smartphones carried by participants performing daily activities.

### Dataset Characteristics

- 6 activity classes
- 30 participants
- Fixed-length time windows
- Multichannel inertial sensor measurements

### Activity Classes

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### Sensor Channels

The following inertial sensor channels were used:

- body_acc_x
- body_acc_y
- body_acc_z
- body_gyro_x
- body_gyro_y
- body_gyro_z
- total_acc_x
- total_acc_y
- total_acc_z

### Final Input Shape

Each sample was represented as:

```text
(128 time steps, 9 channels)
```

---

# Preprocessing Pipeline

The preprocessing pipeline was designed to ensure consistency across all experiments.

## Step 1 — Dataset Loading

Raw inertial sensor signals were loaded from the UCI HAR dataset.

## Step 2 — Sensor Feature Selection

Nine sensor channels containing accelerometer and gyroscope measurements were selected.

## Step 3 — Normalization

All sensor channels were normalized using statistics computed only from the training split.

## Step 4 — Dataset Splitting

The dataset was divided into:

- Training set
- Validation set
- Test set

A stratified validation split was applied to preserve class distribution consistency.

---

# Evaluation Metrics

The following metrics were used to evaluate all experiments. Each metric provides a different perspective on model performance.

## Test Accuracy

The proportion of correctly classified samples over the entire test set. This is the primary metric for overall comparison across experiments.

## Macro F1-Score

The unweighted average of per-class F1-scores. This metric treats all classes equally regardless of sample count, making it particularly important for datasets with class imbalance. A high macro F1-score indicates that the model performs well across all activity classes, not just the majority ones.

## Per-Class F1-Score

The harmonic mean of precision and recall computed separately for each activity class. This metric reveals which specific activities are well-classified and which are confused with others.

### Why SITTING and STANDING F1-Scores Matter

SITTING and STANDING are the most difficult activity classes to distinguish because they share similar motion characteristics — both involve minimal movement and produce highly similar sensor patterns. Models that achieve high accuracy on these two classes demonstrate stronger temporal modeling capabilities.

### Why LAYING F1-Score is Expected to Be High

LAYING produces a distinctly different sensor signal compared to all other activities due to the gravitational acceleration distribution across sensor axes. A near-perfect F1-score for LAYING is expected across all architectures.

---

# Proposed Experimental Architecture

The project follows an incremental experimental design in order to analyze the contribution of each architectural component.

## Experiment 1 — CNN-only Baseline

### Architecture

```text
Input
→ Conv1D
→ MaxPooling1D
→ Conv1D
→ MaxPooling1D
→ GlobalAveragePooling1D
→ Dense
→ Output
```

### Purpose

This experiment establishes the baseline architecture for HAR classification. The objective is to evaluate whether local temporal patterns extracted through 1D convolutional layers are sufficient for activity recognition.

### Architectural Roles

#### CNN Layers

The convolutional layers extract local temporal motion patterns from multichannel sensor signals.

#### MaxPooling Layers

Pooling layers reduce temporal dimensionality while preserving dominant motion features.

### Experimental Result

| Metric | Value |
|---|---|
| Test Accuracy | 92.30% |
| Test Loss | 0.5091 |
| Macro F1-Score | 0.9235 |
| SITTING F1-Score | 0.8277 |
| STANDING F1-Score | 0.8401 |
| LAYING F1-Score | 0.9981 |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| WALKING | 0.9958 | 0.9657 | 0.9806 | 496 |
| WALKING_UPSTAIRS | 0.9798 | 0.9257 | 0.9520 | 471 |
| WALKING_DOWNSTAIRS | 0.8917 | 1.0000 | 0.9428 | 420 |
| SITTING | 0.8146 | 0.8411 | 0.8277 | 491 |
| STANDING | 0.8617 | 0.8195 | 0.8401 | 532 |
| LAYING | 0.9981 | 0.9981 | 0.9981 | 537 |
| **Accuracy** | | | **0.9230** | 2947 |
| Macro Avg | 0.9236 | 0.9250 | 0.9235 | 2947 |
| Weighted Avg | 0.9244 | 0.9230 | 0.9230 | 2947 |

### Observations

The CNN-only baseline achieved strong overall classification performance. However, confusion was observed between low-motion activities such as SITTING and STANDING, indicating limitations in modeling long-range temporal dependencies using convolutional layers alone.

---

# Experiment 2 — CNN + GRU Model

## Architecture

```text
Input
→ Conv1D
→ MaxPooling1D
→ Conv1D
→ MaxPooling1D
→ GRU
→ Dense
→ Output
```

## Purpose

This experiment extends the CNN baseline by introducing recurrent temporal modeling through a GRU layer.

## Architectural Roles

### CNN Layers

The convolutional layers extract local sensor motion patterns.

### GRU Layer

The GRU layer models temporal dependencies and sequential relationships between learned feature representations.

## Experimental Result

| Metric | Value |
|---|---|
| Test Accuracy | 93.38% |
| Test Loss | 0.3447 |
| Macro F1-Score | 0.9333 |
| SITTING F1-Score | 0.8695 |
| STANDING F1-Score | 0.8963 |
| LAYING F1-Score | 0.9944 |

## Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| WALKING | 0.9934 | 0.9113 | 0.9506 | 496 |
| WALKING_UPSTAIRS | 0.9591 | 0.9448 | 0.9519 | 471 |
| WALKING_DOWNSTAIRS | 0.8856 | 0.9952 | 0.9372 | 420 |
| SITTING | 0.9243 | 0.8208 | 0.8695 | 491 |
| STANDING | 0.8614 | 0.9342 | 0.8963 | 532 |
| LAYING | 0.9890 | 1.0000 | 0.9944 | 537 |
| **Accuracy** | | | **0.9338** | 2947 |
| Macro Avg | 0.9354 | 0.9344 | 0.9333 | 2947 |
| Weighted Avg | 0.9364 | 0.9338 | 0.9336 | 2947 |

## Observations

The addition of the GRU layer improved overall classification performance compared to the CNN-only baseline. The most significant improvements were observed for low-motion activities such as SITTING and STANDING, with SITTING F1 improving from 0.8277 to 0.8695 and STANDING F1 improving from 0.8401 to 0.8963. This confirms that recurrent temporal modeling is effective for discriminating activities with similar local sensor characteristics.

---

# Experiment 3 — Autoencoder + CNN

## Architecture

```text
Input
→ Autoencoder Encoder (frozen)
  → Conv1D (64)
  → MaxPooling1D
  → Conv1D (128)
  → MaxPooling1D
→ Conv1D (64)
→ MaxPooling1D
→ Conv1D (128)
→ GlobalAveragePooling1D
→ Dense
→ Output
```

## Purpose

This experiment investigates the effect of feature compression and denoising through an Autoencoder encoder. The encoder is pretrained in a self-supervised manner to reconstruct the input signals, and its weights are subsequently frozen. A CNN classifier is then trained on top of the learned representations.

## Architectural Roles

### Autoencoder Encoder

The encoder is pretrained to reconstruct raw sensor signals, learning a compressed and denoised latent representation of the input without requiring activity labels.

### CNN Layers

The convolutional layers perform local feature extraction on top of the encoder's compressed representations.

### GlobalAveragePooling

Reduces the temporal feature maps into a fixed-size vector for classification.

## Experimental Result

| Metric | Value |
|---|---|
| Test Accuracy | 92.06% |
| Test Loss | 0.4559 |
| Macro F1-Score | 0.9213 |
| SITTING F1-Score | 0.8202 |
| STANDING F1-Score | 0.8444 |
| LAYING F1-Score | 0.9877 |

## Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| WALKING | 1.0000 | 0.9718 | 0.9857 | 496 |
| WALKING_UPSTAIRS | 0.9177 | 0.9469 | 0.9321 | 471 |
| WALKING_DOWNSTAIRS | 0.9190 | 1.0000 | 0.9578 | 420 |
| SITTING | 0.7965 | 0.8452 | 0.8202 | 491 |
| STANDING | 0.8931 | 0.8008 | 0.8444 | 532 |
| LAYING | 1.0000 | 0.9758 | 0.9877 | 537 |
| **Accuracy** | | | **0.9206** | 2947 |
| Macro Avg | 0.9211 | 0.9234 | 0.9213 | 2947 |
| Weighted Avg | 0.9221 | 0.9206 | 0.9204 | 2947 |

## Observations

The Autoencoder encoder provided a strong compressed representation of the sensor signals. However, the absence of recurrent temporal modeling resulted in slightly lower performance compared to the CNN + GRU baseline. SITTING and STANDING F1-scores remained similar to the CNN baseline, confirming that feature compression alone is insufficient for discriminating low-motion activities without sequential modeling.

---

# Experiment 4 — Autoencoder + CNN + GRU (Main Hybrid Model)

## Architecture

```text
Input
→ Autoencoder Encoder (frozen)
  → Conv1D (64)
  → MaxPooling1D
  → Conv1D (128)
  → MaxPooling1D
→ Conv1D (64)
→ MaxPooling1D
→ Conv1D (128)
→ GRU (64)
→ Dense
→ Output
```

## Purpose

This experiment constructs the main hybrid architecture by combining the pretrained Autoencoder encoder with convolutional feature extraction and recurrent temporal modeling. This architecture integrates representation learning, local feature extraction, and sequential dependency modeling.

## Architectural Roles

### Autoencoder Encoder

Provides pretrained compressed representations of the raw sensor signals.

### CNN Layers

Extract local temporal motion patterns from the encoder representations.

### GRU Layer

Models sequential dependencies and long-range temporal relationships between extracted feature representations.

## Experimental Result

| Metric | Value |
|---|---|
| Test Accuracy | 91.96% |
| Test Loss | 0.3791 |
| Macro F1-Score | 0.9198 |
| SITTING F1-Score | 0.8156 |
| STANDING F1-Score | 0.8420 |
| LAYING F1-Score | 1.0000 |

## Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| WALKING | 0.9979 | 0.9476 | 0.9721 | 496 |
| WALKING_UPSTAIRS | 0.9371 | 0.9490 | 0.9430 | 471 |
| WALKING_DOWNSTAIRS | 0.8974 | 1.0000 | 0.9459 | 420 |
| SITTING | 0.8028 | 0.8289 | 0.8156 | 491 |
| STANDING | 0.8809 | 0.8064 | 0.8420 | 532 |
| LAYING | 1.0000 | 1.0000 | 1.0000 | 537 |
| **Accuracy** | | | **0.9196** | 2947 |
| Macro Avg | 0.9193 | 0.9220 | 0.9198 | 2947 |
| Weighted Avg | 0.9206 | 0.9196 | 0.9193 | 2947 |

## Observations

The hybrid architecture combining the Autoencoder encoder, CNN layers, and GRU temporal modeling demonstrated competitive performance. LAYING achieved a perfect F1-score of 1.0000. However, SITTING and STANDING F1-scores remained below the CNN + GRU baseline, suggesting that the frozen encoder representations may limit the model's ability to fine-tune features specifically for low-motion activity discrimination. This limitation motivated the introduction of regularization in Experiment 5.

---

# Experiment 5 — Regularized Hybrid Model

## Architecture

```text
Input
→ Autoencoder Encoder (frozen)
  → Conv1D (64)
  → MaxPooling1D
  → Conv1D (128)
  → MaxPooling1D
→ Conv1D (64)
→ BatchNormalization
→ Conv1D (128)
→ BatchNormalization
→ GRU (tuned units)
→ Dropout
→ Dense (64)
→ Dropout
→ Output
```

## Purpose

This experiment extends the main hybrid architecture with regularization techniques to improve generalization performance. Batch Normalization is applied after convolutional layers to stabilize training, and Dropout is applied after the GRU and Dense layers to reduce overfitting. Hyperparameter tuning is performed across multiple configurations to identify the optimal dropout rate, GRU units, learning rate, and batch size.

## Architectural Roles

### Autoencoder Encoder

Provides pretrained compressed representations of the raw sensor signals.

### CNN Layers with Batch Normalization

Convolutional layers extract local temporal motion patterns. Batch Normalization stabilizes activations and accelerates convergence.

### GRU Layer

Models sequential temporal dependencies between feature representations.

### Dropout

Reduces overfitting by randomly deactivating units during training.

## Hyperparameter Tuning

The following configurations were evaluated during tuning:

| Config | Dropout Rate | GRU Units | Learning Rate | Batch Size |
|---|---|---|---|---|
| 1 | 0.2 | 64 | 0.001 | 64 |
| 2 | 0.3 | 64 | 0.001 | 64 |
| 3 | 0.3 | 128 | 0.0005 | 32 |
| 4 | 0.4 | 128 | 0.001 | 64 |

## Experimental Result

| Metric | Value |
|---|---|
| Test Accuracy | 92.23% |
| Test Loss | 0.3446 |
| Macro F1-Score | 0.9229 |
| SITTING F1-Score | 0.8122 |
| STANDING F1-Score | 0.8330 |
| LAYING F1-Score | 0.9954 |

## Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| WALKING | 0.9743 | 0.9940 | 0.9840 | 496 |
| WALKING_UPSTAIRS | 0.9740 | 0.9533 | 0.9635 | 471 |
| WALKING_DOWNSTAIRS | 0.9415 | 0.9571 | 0.9492 | 420 |
| SITTING | 0.8139 | 0.8106 | 0.8122 | 491 |
| STANDING | 0.8410 | 0.8252 | 0.8330 | 532 |
| LAYING | 0.9908 | 1.0000 | 0.9954 | 537 |
| **Accuracy** | | | **0.9223** | 2947 |
| Macro Avg | 0.9226 | 0.9234 | 0.9229 | 2947 |
| Weighted Avg | 0.9218 | 0.9223 | 0.9220 | 2947 |

## Observations

The regularized hybrid model achieved the best test accuracy among all Autoencoder-based experiments. Batch Normalization improved training stability and Dropout reduced overfitting. However, SITTING and STANDING remain the most challenging classes across all experiments, reflecting the inherent difficulty of distinguishing these activities from sensor signals alone.

---

# Hyperparameter Configuration

The following baseline hyperparameters were used throughout the experiments unless otherwise specified.

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Maximum Epochs | 50 |
| Early Stopping Patience | 8 |
| GRU Units | 64 |

---

# Ablation Study

This project follows an ablation-study-oriented experimental design. Each experiment incrementally introduces additional architectural components in order to analyze their contribution to HAR performance.

| Experiment | Architecture | Test Accuracy | Test Loss | Macro F1 | SITTING F1 | STANDING F1 |
|---|---|---|---|---|---|---|
| Experiment 1 | CNN | 92.30% | 0.5091 | 0.9235 | 0.8277 | 0.8401 |
| Experiment 2 | CNN + GRU | 93.38% | 0.3447 | 0.9333 | 0.8695 | 0.8963 |
| Experiment 3 | AE + CNN | 92.06% | 0.4559 | 0.9213 | 0.8202 | 0.8444 |
| Experiment 4 | AE + CNN + GRU | 91.96% | 0.3791 | 0.9198 | 0.8156 | 0.8420 |
| Experiment 5 | Regularized Hybrid | 92.23% | 0.3446 | 0.9229 | 0.8122 | 0.8330 |

### Key Findings

- **Experiment 2 (CNN + GRU)** achieved the highest overall test accuracy at 93.38%, demonstrating that recurrent temporal modeling provides the most significant performance gain.
- **SITTING and STANDING** remain the most challenging classes across all experiments, with F1-scores consistently below 0.90, highlighting the inherent difficulty of low-motion activity discrimination.
- **LAYING** achieved near-perfect or perfect F1-scores in all experiments due to its distinctive gravitational sensor signature.
- **Experiment 5** achieved the lowest test loss (0.3446) among all experiments, indicating that regularization improved model generalization even when absolute accuracy gains were modest.

---

# Repository Structure

```text
notebooks/
├── 01_preprocessing.ipynb
├── 02_cnn_baseline.ipynb
├── 03_cnn_gru.ipynb
├── 04_ae_cnn.ipynb
├── 05_ae_cnn_gru.ipynb
├── 06_ae_cnn_gru_regularized.ipynb

```

---


# References

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," ESANN 2013.

[2] UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

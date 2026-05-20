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

## Observations

The addition of the GRU layer improved overall classification performance compared to the CNN-only baseline. The most significant improvements were observed for low-motion activities such as SITTING and STANDING, suggesting that recurrent temporal modeling improves discrimination between activities with similar local sensor characteristics.

---

# Planned Experiments

The following experiments are currently in progress:

## Experiment 3 — Autoencoder + CNN

Purpose:
- investigate feature compression and denoising effects using Autoencoder representations prior to CNN classification.

## Experiment 4 — Autoencoder + CNN + GRU

Purpose:
- construct the main hybrid architecture integrating representation learning, local feature extraction, and temporal sequence modeling.

## Experiment 5 — Regularized Hybrid Model

Purpose:
- improve generalization performance using:
  - Batch Normalization
  - Dropout
  - Hyperparameter tuning
  - Early stopping

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

| Experiment | Architecture | Purpose |
|---|---|---|
| Experiment 1 | CNN | Baseline local feature extraction |
| Experiment 2 | CNN + GRU | Temporal dependency modeling |
| Experiment 3 | AE + CNN | Feature compression and denoising |
| Experiment 4 | AE + CNN + GRU | Hybrid architecture |
| Experiment 5 | Regularized Hybrid | Generalization improvement |

---

# Repository Structure

```text
notebooks/
├── 01_preprocessing.ipynb
├── 02_cnn_baseline.ipynb
├── 03_cnn_gru.ipynb

```

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

---

# References

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," ESANN 2013.

[2] UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
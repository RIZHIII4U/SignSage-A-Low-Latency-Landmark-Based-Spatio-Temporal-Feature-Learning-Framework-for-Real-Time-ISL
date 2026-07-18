# SignSage: A Lightweight Hybrid Framework for Real-Time Indian Sign Language Recognition

> Official implementation of the **SignSage** framework for real-time Indian Sign Language (ISL) recognition.

---

## Overview

This repository contains the implementation of **SignSage**, a lightweight hybrid framework for real-time Indian Sign Language (ISL) recognition using MediaPipe hand landmarks. The framework combines wrist-relative landmark normalization, motion-aware routing, Random Forest classification for static gestures, LSTM-based temporal modeling for dynamic gestures, and majority voting for stable real-time predictions.

---

## Highlights

- ✅ Real-time Indian Sign Language recognition
- ✅ MediaPipe-based hand landmark extraction
- ✅ Wrist-relative landmark normalization
- ✅ Motion-aware routing using Motion Energy
- ✅ Motion Energy Threshold (**τ = 0.02**)
- ✅ Random Forest classifier for static gestures
- ✅ Single-layer LSTM classifier for dynamic gestures
- ✅ Majority-voting temporal smoothing
- ✅ CPU-friendly implementation
- ✅ Pre-trained models included

---

# Framework Configuration

| Component | Configuration |
|-----------|---------------|
| Input | RGB Video |
| Landmark Detector | MediaPipe Hands |
| Hand Landmarks | 21 |
| Landmark Features | Wrist-relative normalized (x, y) coordinates |
| Motion Metric | Motion Energy |
| Motion Threshold (τ) | **0.02** |
| Static Gesture Model | Random Forest |
| Dynamic Gesture Model | Single-layer LSTM |
| Prediction Stabilization | Majority Voting |
| Deployment | Real-time CPU Inference |

---

# Repository Structure

```
.
├── Main.ipynb
├── final_code.py
├── feature_extractor.py
├── extract_all.py
├── augment_dataset.py
│
├── realtime_static_voice_ui.py
├── realtime_dynamic_lstm_inference.py
│
├── alphabets_model.h5
├── numbers_model.h5
├── dynamic_model.h5
├── isl_dynamic_lstm.h5
├── lstm_signsage.keras
│
├── dynamic_classes.npy
├── label1.txt
├── label2.txt
├── label3.txt
│
├── hand_landmarker.task
│
├── Fig_Motion_Energy_Routing.png
├── Fig_RF_Confusion_Matrix.png
├── Fig_LSTM_Confusion_Matrix.png
├── Fig_LSTM_Training_Curves.png
├── Fig_Learning_Rate_Study.png
├── Fig_Final_Results_Summary.png
│
└── README.md
```

---

# Repository Contents

## Source Code

| File | Description |
|------|-------------|
| `Main.ipynb` | Main notebook containing training and evaluation pipeline |
| `final_code.py` | Complete SignSage implementation |
| `feature_extractor.py` | MediaPipe hand landmark extraction |
| `extract_all.py` | Feature extraction from the dataset |
| `augment_dataset.py` | Dataset augmentation |
| `realtime_static_voice_ui.py` | Real-time static gesture recognition with voice output |
| `realtime_dynamic_lstm_inference.py` | Real-time dynamic gesture recognition |

---

## Pre-trained Models

| Model | Purpose |
|--------|---------|
| `alphabets_model.h5` | Static alphabet recognition |
| `numbers_model.h5` | Static number recognition |
| `dynamic_model.h5` | Dynamic gesture recognition |
| `isl_dynamic_lstm.h5` | LSTM dynamic gesture model |
| `lstm_signsage.keras` | Final SignSage LSTM model |

---

## Supporting Files

| File | Description |
|------|-------------|
| `dynamic_classes.npy` | Dynamic gesture labels |
| `label1.txt` | Alphabet labels |
| `label2.txt` | Number labels |
| `label3.txt` | Dynamic gesture labels |
| `hand_landmarker.task` | MediaPipe hand landmark model |

---

## Figures

| Figure | Description |
|--------|-------------|
| `Fig_Motion_Energy_Routing.png` | Motion-aware routing mechanism |
| `Fig_RF_Confusion_Matrix.png` | Confusion matrix for Random Forest |
| `Fig_LSTM_Confusion_Matrix.png` | Confusion matrix for LSTM |
| `Fig_LSTM_Training_Curves.png` | LSTM training and validation curves |
| `Fig_Learning_Rate_Study.png` | Learning rate analysis |
| `Fig_Final_Results_Summary.png` | Overall experimental results |

---

# Requirements

- Python 3.10+
- TensorFlow
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- Pandas
- Matplotlib
- joblib
- pyttsx3

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Usage

### Extract Features

```bash
python extract_all.py
```

### Augment Dataset

```bash
python augment_dataset.py
```

### Run Static Gesture Recognition

```bash
python realtime_static_voice_ui.py
```

### Run Dynamic Gesture Recognition

```bash
python realtime_dynamic_lstm_inference.py
```

---

# Reproducibility

This repository accompanies the manuscript submitted to **Scientific Reports**.

It provides the implementation of the proposed SignSage framework, including:

- MediaPipe hand landmark extraction
- Wrist-relative landmark normalization
- Motion-aware routing
- Static gesture recognition using Random Forest
- Dynamic gesture recognition using LSTM
- Real-time inference pipeline

The repository has been made **publicly available** to support transparency and reproducibility of the reported research.

---
# License

This repository is intended for academic and research purposes.

---


## Note

This repository corresponds to the implementation of the revised manuscript submitted to **Scientific Reports**. Documentation may be updated over time to improve clarity and reproducibility without changing the reported methodology.

# SignSage: A Lightweight Hybrid Framework for Real-Time Indian Sign Language Recognition

> **Official implementation of the SignSage framework described in our Scientific Reports manuscript.**

---

## Reproducibility

This repository accompanies the manuscript submitted to **Scientific Reports**.

It contains the complete implementation of the proposed **SignSage** framework, including:

- MediaPipe hand landmark extraction
- Wrist-relative landmark normalization
- Motion-aware routing
- Random Forest classifier for static gesture recognition
- LSTM-based temporal modeling for dynamic gesture recognition
- Majority-voting temporal smoothing
- Real-time inference pipeline

The repository is made publicly available to support transparency and reproducibility of the reported results.

---

## Paper

**Title**

> **<Replace with your final revised manuscript title exactly>**

**Authors**

Deepika J., *et al.*

**Journal**

Scientific Reports *(Under Review)*

---

## Overview

SignSage is a lightweight hybrid framework for real-time Indian Sign Language (ISL) recognition designed for CPU-based deployment. The framework combines MediaPipe-based hand landmark extraction with wrist-relative normalization, motion-aware routing, Random Forest classification for static gestures, LSTM-based temporal modeling for dynamic gestures, and majority-voting based temporal smoothing to achieve efficient and accurate recognition.

---

## Framework

The overall pipeline consists of the following stages:

1. Video Capture
2. MediaPipe Hand Landmark Extraction
3. Wrist-relative Landmark Normalization
4. Motion Energy Computation
5. Motion-aware Routing
6. Random Forest Classification (Static Gestures)
7. LSTM Classification (Dynamic Gestures)
8. Majority Voting
9. Final Gesture Prediction

---

## Repository Structure

```text
SignSage/
│
├── dataset/
├── models/
├── preprocessing/
├── training/
├── inference/
├── utils/
├── results/
├── figures/
├── requirements.txt
└── README.md
```

---

## Dataset

The experiments use publicly available Indian Sign Language datasets described in the manuscript.

Please download the corresponding datasets from their original sources and organize them using the following directory structure.

```text
dataset/
    static/
    dynamic/
```

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Feature Representation | Wrist-relative normalized landmarks |
| Static Classifier | Random Forest |
| Dynamic Classifier | Single-layer LSTM |
| Motion Routing | Motion Energy Threshold |
| Temporal Smoothing | Majority Voting |

---

## Requirements

- Python 3.10+
- TensorFlow
- Scikit-learn
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Framework

### Train

```bash
python train.py
```

### Test

```bash
python test.py
```

### Real-Time Inference

```bash
python inference.py
```

---

## Results

The manuscript reports:

- Static Gesture Recognition
- Dynamic Gesture Recognition
- Overall Framework Performance
- CPU Inference Time
- Frames Per Second (FPS)

Detailed experimental settings and evaluation metrics are described in the accompanying manuscript.

---

## Reproducibility Notes

The implementation follows the methodology described in the manuscript.

For successful reproduction:

- Use the same preprocessing pipeline.
- Maintain the dataset directory structure.
- Install all required dependencies.
- Follow the experimental configuration reported in the paper.

---

## License

This project is released for academic and research purposes.

---


## Note

This repository contains the implementation corresponding to the revised manuscript submitted to **Scientific Reports**. Minor updates may be incorporated to improve documentation and reproducibility while preserving the methodology and experimental settings described in the manuscript.

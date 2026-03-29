# Brain Signal Classification using EEG (CSP + SVM)

This project implements a Brain-Computer Interface (BCI) pipeline to classify motor imagery EEG signals using machine learning.

## Overview

The system processes EEG signals from the PhysioNet EEG Motor Movement dataset and predicts whether a subject is imagining left or right hand movement.

Pipeline:

EEG Data → Bandpass Filter → Epoch Extraction → CSP Feature Extraction → SVM Classifier → Cross-Validation

## Dataset

PhysioNet EEG Motor Movement/Imagery Dataset

Subjects used: 1–4  
Runs used: 6, 10, 14

## Methods

- EEG preprocessing with MNE
- Bandpass filtering (7–30 Hz)
- Epoch segmentation (2 seconds)
- Common Spatial Patterns (CSP) for feature extraction
- Support Vector Machine (SVM) classifier
- Stratified 5-fold cross-validation

## Results

Mean classification accuracy:

~56%

This performance is consistent with classical baseline methods for motor imagery EEG classification.

## Technologies Used

- Python
- MNE
- Scikit-learn
- NumPy

## Author

Alex Thomas

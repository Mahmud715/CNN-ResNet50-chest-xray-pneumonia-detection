# Chest X-Ray Pneumonia Detection Pipeline

**Repository Name:** CNN-ResNet50-chest-xray-pneumonia-detection  
**Description:** A CPU-friendly pipeline for automated pneumonia detection on chest X-ray images, combining classical preprocessing (OpenCV) with a transfer-learned ResNet-50 model (TensorFlow/Keras).

---

## 🚀 Project Overview
This repository provides all code and model artifacts needed to reproduce and extend the pneumonia detection workflow described in Mahmud Huseynov’s May 28, 2025 project report. It includes:

- **Training script (`train_full.py`)**: a self-contained Python script for preprocessing, model definition, training, and evaluation.  
- **Inference script (`inference_on_folder.py`)**: loads the pretrained model to classify a folder of unseen chest X-ray images.  
- **Data folders**:
  - `chest_xray/`: original dataset split into `train/`, `val/`, and `test/` subdirectories.  
  - `chest_unseen_images/`: external samples used for demonstration of inference.

---

## 📁 Repository Structure
```
CNN-ResNet50-chest-xray-pneumonia-detection/
├── chest_xray/               # Kaggle dataset (train/val/test)
├── chest_unseen_images/      # Unseen X-ray samples for inference
├── train_full.py             # One-file training & evaluation script
├── inference_on_folder.py    # Script to classify unseen images
└── README.md                 # Project documentation (this file)
```

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Mahmud715/CNN-ResNet50-chest-xray-pneumonia-detection.git
   cd CNN-ResNet50-chest-xray-pneumonia-detection
   ```
2. Create a Python virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate.bat    # Windows
   pip install --upgrade pip
   pip install tensorflow opencv-python
   ```
3. Download dataset from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
4. Place your dataset folders (`chest_xray/`) and unseen images (`chest_unseen_images/`) in the repo root.

---

## ▶️ Usage

### 1. Training & Evaluation
```bash
python train_full.py
```
- Trains the ResNet-50 model on the CPU.  
- Saves the final model as `pneumonia_model_full.h5`.  
- Prints training/validation metrics and final test accuracy.

### 2. Inference on New Images
```bash
python inference_on_folder.py
```
- Loads `pneumonia_model_full.h5`.  
- Processes all images in `chest_unseen_images/`.  
- Displays each image with predicted label and confidence.

---

## 📊 Results Summary
Visuals and quantitative metrics demonstrate the model’s high training accuracy (98.1%) and test accuracy (67.8%), along with discussion of overfitting and generalization limitations.

---

## 🔖 References
1. He K. et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.  
2. Simonyan K. & Zisserman A., "Very Deep Convolutional Networks for Large-Scale Image Recognition," *ICLR*, 2015.  
3. Wang X. et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," *CVPR*, 2017.  
4. Wang L. et al., "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images," *arXiv*, 2020.  
5. Chollet F., "Xception: Deep Learning with Depthwise Separable Convolutions," *CVPR*, 2017.  
6. Bradski G., "The OpenCV Library," *Dr. Dobb’s J. Softw. Tools*, 2000.  
7. Timothymooney P., "Chest X-ray Pneumonia," *Kaggle*, 2018. Available: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 📜 License
This project is released under the Apache License 2.0.

---

## 🤝 Contribution
Contributions are welcome! Please open issues or submit pull requests for bug fixes, performance improvements, or additional features.

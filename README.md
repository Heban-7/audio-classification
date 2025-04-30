# 🎵 Music Genre Classification

This repository presents a comprehensive solution for classifying music genres using the **GTZAN dataset**, developed during the **48-Hour Audio ML Challenge**. The project encompasses the entire machine learning pipeline—from data preprocessing to model evaluation—leveraging **Mel spectrograms** and two distinct models:

- A **Custom Convolutional Neural Network (CNN)** achieving **89.49%** test accuracy.
- A **Modified ResNet18** model achieving **96.80%** test accuracy.

---

## 📚 Table of Contents

- [About the Challenge](#-about-the-challenge)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)
- [Contact](#-contact)

---

## 🏁 About the Challenge

The **48-Hour Audio ML Challenge** tasks participants with developing an end-to-end audio classification solution within a 48-hour timeframe. The focus is on delivering a functional, high-performing model with clean code and clear documentation. This project aims to classify music genres effectively, providing robust and interpretable results.

---

## 🎼 Dataset

The project utilizes the **GTZAN dataset**, which comprises:

- **1,000** audio clips, each **30 seconds** long.
- **10** genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.
- Sample rate: **22,050 Hz**.

👉 **Download Instructions:** Please refer to `data.examples/README.md` for detailed instructions on downloading and preparing the dataset.

---

## 📁 Project Structure

```bash
music-genre-classification/
├── data.examples/
│   └── README.md                  # Instructions to download GTZAN dataset
├── scripts/
│   ├── data_preprocessing.py      # Data preprocessing and feature extraction
│   ├── cnn_model.py               # Custom CNN model definition
│   ├── model_evaluation.py        # Model evaluation script
│   └── resnet18_model.py          # ResNet18 model definition
├── src/
│   └── mel_spectogram_feature_extraction.py  # Extracts Mel spectrogram features
├── models/                        # Directory to save trained models
├── notebooks/
│   ├── data_preprocessing.ipynb   # Data exploration and preprocessing
│   └── cnn_model_training.ipynb   # CNN model training notebook
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore file
├── README.md                      # Project overview and instructions
└── report.pdf                     # 2-page project report
```

---


## 🚀 Usage
### 1. Clone the Repository

`git clone https://github.com/Heban-7/audio-classification.git
cd audio-classification
`

---

### 2. Installation
Requires Python 3.6+

`pip install -r requirements.txt`

---

### 3. Download and Prepare the GTZAN Dataset
Follow the instructions provided in `data.examples/README.md` to download and set up the dataset.

### 4. Extract Mel Spectrogram Features
Run the following script to extract Mel spectrogram features and split the data into training, validation, and test sets:
`python src/mel_spectogram_feature_extraction.py
`

---

### 5. Data Preprocessing 
Process input raw data using `scripts/data_preprocessing.py` scripts on `notebook/data_preprocessing.ipynb`

---

### 6. Train the Models
* Custom CNN Model:
  Train the cnn model using `scripts/cnn_model.py` scripts on `notebook/cnn_model_training.ipynb`
* ResNet18 Model:
  Train the Resnet18 model using `scripts/resnet_model.py` scripts on `notebook/cnn_model_training.ipynb`

---

### 7. Evaluate the Models
After training, evaluate the performance of the models using:
python scripts/model_evaluation.py

---

# 📊 Results

Model	Test Accuracy
Custom CNN	89.49%
ResNet18	96.80%

`**🔍 Note**: These results significantly outperform the human benchmark of ~70% accuracy on 3-second clips.`

---

# 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions, improvements, or bug fixes

# 📬 Contact

For questions, feel free to reach out to Liul Jima at liuljima1896@gmail.com

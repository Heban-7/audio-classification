# 🎵 Music Genre Classification:

This repository contains the code and documentation for a music genre classification project using the **GTZAN dataset**, developed as part of the **48-Hour Audio ML Challenge**. The project implements a complete machine learning pipeline, from data preprocessing to model evaluation, using **Mel spectrograms** and two models:

- A **custom CNN**, achieving **89.49%** test accuracy
- A **modified ResNet18**, achieving **96.80%** test accuracy

---

## 📚 Table of Contents

- [About the Challenge](#about-the-challenge)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Quick Start](#quick-start)
- [Results](#results)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🏁 About the Challenge

The **48-Hour Audio ML Challenge** requires participants to build an end-to-end audio classification solution within 48 hours. The emphasis is on functionality, performance, code quality, and clear communication. This project classifies music genres efficiently with robust and interpretable results.

---

## 🎼 Dataset

The project uses the **GTZAN dataset**, which contains:

- 1,000 audio clips (30 seconds each)
- 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- Sample rate: 22,050 Hz

👉 **Download Instructions:** See `data.examples/README.md` for dataset preparation.

---

## 📁 Project Structure

```bash
data.examples/                  # Dataset download & usage instructions
scripts/
├── data_preprocess.py         # Preprocess raw audio files
├── cnn_model.py               # Custom CNN training
├── resnet18_model.py          # ResNet18 training
└── model_evaluation.py        # Evaluate model performance
src/
└── mel_spectogram_feature_extraction.py  # Mel spectrogram extraction
models/                        # Saved trained models
notebooks/
├── data_preprocessing.ipynb   # Data exploration & preprocessing
└── cnn_model_training.ipynb   # CNN training notebook
requirements.txt               # Python dependencies
report.pdf                     # Project summary report
.gitignore
README.md

```
---
## 💻 Installation
Requires Python 3.6+

`pip install -r requirements.txt`

---
## 🚀 Running the Project

### Download and prepare the GTZAN dataset

Follow instructions in `data.examples/README.md`

---

### Extract Mel spectrogram features

`python src/mel_spectogram_feature_extraction.py
`

---

# ⚡ Quick Start

`git clone https://github.com/Heban-7/audio-classification.git

cd music-genre-classification

pip install -r requirements.txt`

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
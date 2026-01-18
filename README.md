# CodeAlpha_Emotion_Recognition_from_Speech
🎧 Speech Emotion Recognition using Deep Learning to classify human emotions (happy, sad, angry, neutral,..) from audio signals using MFCC features and neural networks.
# Speech Emotion Recognition using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Overview

A deep learning-based system for recognizing emotions from speech audio using **Convolutional Neural Networks (CNN)** and **Mel-Frequency Cepstral Coefficients (MFCC)** feature extraction. This project classifies audio recordings into 8 different emotion categories with ~73% accuracy.

## 🎯 Objective

To build an intelligent emotion recognition system that can accurately identify human emotions from voice recordings, useful for applications in customer service, mental health monitoring, virtual assistants, and human-computer interaction.

## 🔬 Methodology

### Deep Learning Architecture:
- **CNN (Convolutional Neural Network)** - Primary model
- **MFCC Feature Extraction** - Audio preprocessing
- **Dropout Regularization** - Overfitting prevention

### Emotions Recognized:
| Emotion | Label | Emoji |
|---------|-------|-------|
| Neutral | 0 | 😐 |
| Calm | 1 | 😌 |
| Happy | 2 | 😊 |
| Sad | 3 | 😢 |
| Angry | 4 | 😠 |
| Fearful | 5 | 😨 |
| Disgust | 6 | 🤢 |
| Surprised | 7 | 😲 |

## 📊 Model Performance

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 95.2% | 72.9% |
| **Loss** | 0.15 | 1.37 |

### Per-Emotion Performance:

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.84 | 0.75 | 0.79 | 36 |
| Calm | 0.78 | 0.85 | 0.81 | 33 |
| Disgust | 0.75 | 0.64 | 0.69 | 47 |
| Fearful | 0.73 | 0.77 | 0.75 | 39 |
| Happy | 0.65 | 0.67 | 0.66 | 36 |
| Neutral | 0.61 | 0.79 | 0.69 | 14 |
| Sad | 0.72 | 0.65 | 0.68 | 43 |
| Surprised | 0.71 | 0.80 | 0.75 | 40 |
| **Overall** | **0.73** | **0.73** | **0.73** | **288** |

## 🛠️ Technologies Used

- **Python 3.7+**
- **Deep Learning Frameworks**:
  - TensorFlow 2.x - Model building
  - Keras - High-level API
- **Audio Processing**:
  - librosa - Audio feature extraction
  - soundfile - Audio I/O operations
- **Data Science**:
  - numpy - Numerical computations
  - pandas - Data manipulation
  - scikit-learn - Model evaluation
- **Visualization**:
  - matplotlib - Plotting
  - seaborn - Statistical visualization

## 📁 Project Structure
```
Speech-Emotion-Recognition/
│
├── Emotion_Recognition_from_Speech.ipynb  # Main notebook
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
│
├── models/
│   ├── emotion_recognition_model.h5       # Trained CNN model
│   └── label_encoder.pkl                  # Label encoder
│
├── data/
│   └── Audio_Speech_Actors_01-24/         # RAVDESS dataset
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
│
└── results/
    ├── confusion_matrix.png               # Confusion matrix plot
    ├── training_curves.png                # Training visualization
    └── classification_report.txt          # Detailed metrics
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7 or higher
pip package manager
Jupyter Notebook (for .ipynb file)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download RAVDESS dataset:
```bash
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip
```

4. Run the Jupyter notebook:
```bash
jupyter notebook Emotion_Recognition_from_Speech.ipynb
```

## 📝 Model Workflow

### 1. Data Loading & Preprocessing
- Load RAVDESS dataset (1,440 audio samples)
- Extract emotion labels from filenames
- Handle missing data

### 2. Feature Extraction
- Extract 40 MFCC coefficients per audio
- Duration: 3 seconds, Offset: 0.5 seconds
- Sample rate: 22,050 Hz

### 3. Model Architecture
```
Input: (40, 1, 1)
├── Conv2D (64 filters, 3×1) + ReLU
├── MaxPooling2D (2×1)
├── Dropout (0.3)
├── Conv2D (128 filters, 3×1) + ReLU
├── MaxPooling2D (2×1)
├── Dropout (0.3)
├── Flatten
├── Dense (128 units) + ReLU
├── Dropout (0.3)
└── Dense (8 units) + Softmax

Total Parameters: 157,192
```

### 4. Training Configuration
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 32
- Epochs: 30
- Train/Test Split: 80/20

### 5. Evaluation Metrics
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Training/Validation Accuracy & Loss Curves

## 💻 Usage

### Predict Emotion from Audio File
```python
# Import necessary libraries
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Load trained model
model = load_model('emotion_recognition_model.h5')

# Function to extract MFCC
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Predict emotion
def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc.reshape(1, 40, 1, 1)
    prediction = model.predict(mfcc)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emotion = emotions[np.argmax(prediction)]
    return emotion

# Example usage
result = predict_emotion('test_audio.wav')
print(f"Predicted Emotion: {result}")
```

### Upload and Test in Google Colab
```python
from google.colab import files

# Upload audio file
uploaded = files.upload()

# Predict emotion
for file_name in uploaded:
    emotion = predict_emotion(file_name)
    print(f"File: {file_name}")
    print(f"Predicted Emotion: {emotion}")
```

## 📈 Key Visualizations

### 1. Confusion Matrix
Shows prediction accuracy across all emotion classes.

### 2. Training & Validation Curves
Displays model learning progress over 30 epochs.

### 3. Classification Report
Detailed precision, recall, and F1-scores per emotion.

## 💡 Key Insights

1. **Training vs Validation**: Gap indicates overfitting (95% train vs 73% validation)
2. **Best Emotions**: Calm (81% F1) and Angry (79% F1) perform best
3. **Challenging Emotions**: Happy (66% F1) and Neutral (69% F1) need improvement
4. **Dataset Size**: Limited to 1,440 samples affects generalization
5. **Feature Quality**: MFCC effectively captures emotional patterns

## 🔮 Future Improvements

- [ ] Implement data augmentation (pitch shifting, time stretching, noise addition)
- [ ] Add LSTM/GRU layers for temporal sequence modeling
- [ ] Use transfer learning with pre-trained audio models
- [ ] Implement ensemble methods (CNN + RNN)
- [ ] Add attention mechanisms for feature focus
- [ ] Increase dataset size with TESS, CREMA-D datasets
- [ ] Deploy as web application using Flask/Streamlit
- [ ] Real-time emotion detection from microphone
- [ ] Multi-language support
- [ ] Add SHAP values for model explainability

## 📊 Dataset Information

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Total Samples**: 1,440 audio files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Format**: WAV (16-bit, 48kHz)
- **Duration**: ~3 seconds per recording
- **Emotions**: 8 categories (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Source**: [Zenodo Repository](https://zenodo.org/record/1188976)

#### Filename Convention:
```
03-01-06-01-02-01-12.wav
│  │  │  │  │  │  └─ Actor ID (01-24)
│  │  │  │  │  └──── Repetition (01 or 02)
│  │  │  │  └─────── Statement (01 or 02)
│  │  │  └────────── Intensity (01: normal, 02: strong)
│  │  └───────────── Emotion (01-08)
│  └──────────────── Vocal channel (01: speech)
└─────────────────── Modality (03: Audio-only)
```

## 🎓 Learning Outcomes

Through this project, I gained expertise in:
- Deep learning for audio classification
- MFCC feature extraction and preprocessing
- CNN architecture design for 1D audio data
- Handling imbalanced datasets
- Model evaluation and performance metrics
- TensorFlow/Keras implementation
- Audio signal processing with librosa
- Overfitting prevention techniques

## ⚠️ Limitations

1. **Overfitting**: High training (95%) vs validation (73%) accuracy gap
2. **Dataset Size**: Only 1,440 samples limits generalization
3. **Language**: English-only dataset
4. **Recording Quality**: Studio-quality audio may not reflect real-world conditions
5. **Browser Constraints**: Google Colab doesn't support real-time microphone recording
6. **Speaker Dependency**: Limited to 24 actors' voices

## 👨‍💻 Author

**Your Name**
- GitHub: [PONNADIAN ](https://github.com/PONNADIAN)
- LinkedIn: [PONNADIAN SA](https://linkedin.com/in/ponnadian-sa-5649a5328)
- Email: upgrademyskill@gmail.com
## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/PONNADIAN/Speech-Emotion-Recognition/issues).

### Contribution Guidelines:
- Follow PEP 8 coding standards
- Add unit tests for new features
- Update documentation
- Write clear commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RAVDESS dataset creators
- TensorFlow and Keras teams
- librosa library developers
- The open-source community

## 📞 Contact

For any queries regarding this project:
- Create an issue in this repository
- Connect with me on LinkedIn
- Send an email

---

<div align="center">
  
### ⭐ If you found this project helpful, please give it a star!

**Made with ❤️ for Speech Emotion Recognition**

[🔝 Back to Top](#speech-emotion-recognition-using-deep-learning)

</div>

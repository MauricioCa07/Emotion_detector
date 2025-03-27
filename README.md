# Emotion Detection System

A real-time facial emotion recognition system that detects and classifies seven human emotions using deep learning and computer vision.

![Emotion Detection System](https://github.com/yourusername/emotion-detection/raw/main/screenshots/demo.png)

## Overview

This application uses a Convolutional Neural Network (CNN) trained on facial expressions to detect and classify emotions in real-time through a webcam feed. The system can identify seven different emotions: anger, disgust, fear, happiness, neutrality, sadness, and surprise.

## Features

- **Real-time emotion detection** through webcam
- **User-friendly GUI** built with CustomTkinter
- **Visual feedback** with progress bars for each emotion probability
- **Easy-to-use controls** for starting/stopping detection
- **Responsive design** that works on various screen sizes

## Dataset

The model was trained using the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) from Kaggle. This dataset contains approximately 35,000 grayscale images of faces displaying various emotions.

The dataset includes:
- 7 emotion categories (angry, disgust, fear, happy, neutral, sad, surprise)
- 48x48 pixel grayscale images
- Images already aligned and centered on the face

## Model Architecture

The CNN model architecture includes:
- Multiple convolutional layers with batch normalization
- Max pooling layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for classification
- L2 regularization for better generalization


## Performance

The trained model achieved **79% accuracy** on the validation set, which is competitive with state-of-the-art models on this challenging dataset.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- CustomTkinter
- PIL (Pillow)
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. Make sure you have a webcam connected to your system.

## Usage

1. Run the application:
```bash
python emotion_detector.py
```

2. Click "Start Detection" to begin emotion recognition.

3. The application will display:
   - Live webcam feed
   - Currently detected emotion
   - Probability bars for each emotion

4. Click "Stop Detection" to pause or "Quit" to exit the application.

## Training Your Own Model

If you want to train your own model:

1. Download the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) from Kaggle.

2. Organize the dataset in the following structure:
```
carpetas/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

3. Run the training script:
```bash
python train_model.py
```

4. The trained model will be saved as `modelo_entrenado.keras`.


## Acknowledgments

- [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) by Jonathan Oheix
- TensorFlow team for the deep learning framework
- CustomTkinter for the modern UI components

## Contact

If you have any questions or suggestions, please open an issue or contact [carrillomauricio007@gmail.com](mailto:carrillomauricio007@gmail.com).

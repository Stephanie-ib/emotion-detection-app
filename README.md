# 🎭 Face Emotion Detection Web App

An AI-powered web application that detects emotions from facial images using deep learning.

## Features
- Upload facial images
- AI detects 7 emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Stores user data in SQLite database
- Beautiful, responsive web interface

## Technologies Used
- **Backend**: Flask (Python)
- **AI/ML**: TensorFlow/Keras, CNN Model
- **Database**: SQLite
- **Frontend**: HTML, CSS (inline)
- **Deployment**: Render

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/Stephanie-ib/FACE_DETECTION.git
cd FACE_DETECTION
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - if model file not included):
```bash
python model_training.py
```

4. Run the app:
```bash
python app.py
```

5. Open browser and go to: `http://127.0.0.1:5001`

## Dataset
This project uses the FER2013 dataset for training.

## Author
Stephanie Ibrahim

## License
MIT License
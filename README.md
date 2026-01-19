Rock-Paper-Scissors AI (MediaPipe + TensorFlow)

An interactive Rock–Paper–Scissors (Suit Digital) game that uses computer vision and AI to recognize real-time hand gestures via webcam.  
The system detects the player’s hand using MediaPipe Hands, classifies gestures with a TensorFlow (Keras) model, and plays against a computer opponent in a Best of 5 (BO5) match format.
This project demonstrates the integration of computer vision and machine learning for real-time human–computer interaction.

---

Features

- Real-time hand tracking using MediaPipe Hands
- Gesture classification using a trained TensorFlow/Keras model
- Supported gestures:
  - ✊ Rock (Batu)
  - ✌️ Scissors (Gunting)
  - ✋ Paper (Kertas)
- Countdown & freeze-frame gesture analysis
- Computer randomly selects gestures
- Automatic win/lose evaluation
- Best of 5 (BO5) scoring system
- Game UI built with Pygame
- Sound effects and visual feedback

---

How It Works

1. Webcam captures the player’s hand.
2. MediaPipe detects and tracks hand landmarks.
3. The frame is frozen for gesture analysis.
4. The trained ML model predicts the gesture.
5. The computer randomly selects its move.
6. The system compares results and updates the score.
7. First side to reach 3 wins (BO5) wins the match.
8. The game automatically resets after a match ends.

> ⚠️ Note:  
> A match result screen was planned, but due to a known bug, the game currently resets scores immediately after BO5 completion.

---

Project Structure

```txt
ROCK-PAPER-SCISSORS-AI-MEDIAPIPE/
│
├── assets/                 # Images, audio, and visual assets
│
├── dataset/                # Collected hand gesture datasets
│
├── models/
│   ├── keras_model.h5      # Trained Keras model
│   └── labels.txt          # Gesture labels
│
├── src/
│   ├── main.py             # Main game loop
│   ├── collect_dataset.py # Dataset collection script
│   └── train_mediapipe_model.py # Model training script
│
├── requirements.txt
└── README.md

Environment & Dependencies

Environment
- Python: 3.10

Libraries
- TensorFlow: 2.15.0
- NumPy: 1.26.4
- MediaPipe: 0.10.9
- OpenCV: 4.8.0.76
- Pygame

Installation

1. Clone Installation
```bash
git clone https://github.com/rionugrahafalasca/rock-paper-scissors-ai-mediapipe.git
cd rock-paper-scissors-ai-mediapipe
2. Install dependencies:
```bash
pip install -r requirements.txt

Running the Game
python src/main.py
Make sure:
- Webcam is connected and accessible
- You are running the command from the project root directory

Model Training (Optional)

Originally, the model was trained using Google Teachable Machine.
Later, the training pipeline was migrated to custom Python-based data collection and training, following recommendations for better flexibility and performance.

To collect dataset:
```bash
python src/collect_dataset.py

To train the model:
```bash
python src/train_mediapipe_model.py

Future Improvements
- Add final match result screen (Win/Lose)

Credits & Notes

- Hand tracking powered by MediaPipe
- Machine learning model built with TensorFlow / Keras
- Background assets sourced from public image resources
- This project was developed as an independent AI & computer vision experiment and learning project

License

This project is intended for educational and portfolio purposes.
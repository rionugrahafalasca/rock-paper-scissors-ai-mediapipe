import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame
import time
import random
from collections import deque
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BACKGROUNDS_DIR = os.path.join(ASSETS_DIR, "backgrounds")

# =========================
# CONFIG
# =========================
CANVAS_W, CANVAS_H = 1280, 720
CAM_W, CAM_H = 360, 270

WIN_SCORE = 3
CONF_THRESHOLD = 0.7
BUFFER_LEN = 20
STABLE_RATIO = 0.75
COUNTDOWN_TIME = 3
ANIM_DELAY = 0.15

# =========================
# AUDIO
# =========================
pygame.mixer.init()
spin_sound = pygame.mixer.Sound(
    os.path.join(AUDIO_DIR, "video-game-text-330163.mp3")
)
result_sound = pygame.mixer.Sound(
    os.path.join(AUDIO_DIR, "game-start-6104.mp3")
)
spin_sound.set_volume(0.4)
result_sound.set_volume(0.7)

# =========================
# LOAD ASSETS
# =========================
bg = cv2.resize(
    cv2.imread(
        os.path.join(BACKGROUNDS_DIR, "game-background-4956017_1280.webp")
    ),
    (CANVAS_W, CANVAS_H)
)

robot_imgs = {
    0: cv2.resize(
        cv2.imread(os.path.join(IMAGES_DIR, "batu.png")),
        (220, 220)
    ),
    1: cv2.resize(
        cv2.imread(os.path.join(IMAGES_DIR, "gunting.png")),
        (220, 220)
    ),
    2: cv2.resize(
        cv2.imread(os.path.join(IMAGES_DIR, "kertas.png")),
        (220, 220)
    )
}

classes = ["Batu", "Gunting", "Kertas"]
model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "keras_model.h5"),
    compile=False
)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# GAME STATE
# =========================
state = "PREVIEW"  # PREVIEW, LOCKED, COUNTDOWN, ROUND_RESULT, MATCH_RESULT

pred_buffer = deque(maxlen=BUFFER_LEN)
stable_choice = None
stable_conf = 0.0

locked_choice = None
robot_choice = None

score_player = 0
score_robot = 0

countdown_start = 0
robot_anim_index = 0
robot_anim_timer = time.time()

spin_playing = False

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# FUNCTIONS
# =========================
def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def winner(p, r):
    if p == r:
        return "SERI"
    if (p==0 and r==1) or (p==1 and r==2) or (p==2 and r==0):
        return "MENANG"
    return "KALAH"

def reset_round():
    global state, pred_buffer, locked_choice, robot_choice
    pred_buffer.clear()
    locked_choice = None
    robot_choice = None
    state = "PREVIEW"

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cam = cv2.resize(frame, (CAM_W, CAM_H))
    canvas = bg.copy()

    # =========================
    # HAND TRACKING (ALWAYS ON)
    # =========================
    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    gesture_text = "Tidak Jelas"

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(cam, hand, mp_hands.HAND_CONNECTIONS)

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        x1, y1 = int(min(xs)*CAM_W), int(min(ys)*CAM_H)
        x2, y2 = int(max(xs)*CAM_W), int(max(ys)*CAM_H)
        pad = 30

        crop = cam[max(0,y1-pad):min(CAM_H,y2+pad),
                   max(0,x1-pad):min(CAM_W,x2+pad)]

        if crop.size > 0:
            pred = model.predict(preprocess(crop), verbose=0)[0]
            if max(pred) > CONF_THRESHOLD:
                pred_buffer.append(np.argmax(pred))

    # =========================
    # STABLE GESTURE
    # =========================
    if len(pred_buffer) == BUFFER_LEN:
        stable_choice = max(set(pred_buffer), key=pred_buffer.count)
        stable_conf = pred_buffer.count(stable_choice) / BUFFER_LEN
        gesture_text = classes[stable_choice]

        if stable_conf >= STABLE_RATIO and state == "PREVIEW":
            state = "LOCKED"

    # =========================
    # GAME LOGIC
    # =========================
    if state == "COUNTDOWN":
        if not spin_playing:
            spin_sound.play(-1)
            spin_playing = True

        if time.time() - robot_anim_timer > ANIM_DELAY:
            robot_anim_index = (robot_anim_index + 1) % 3
            robot_anim_timer = time.time()

        if time.time() - countdown_start >= COUNTDOWN_TIME:
            spin_sound.stop()
            spin_playing = False

            robot_choice = random.randint(0,2)
            result = winner(locked_choice, robot_choice)
            result_sound.play()

            if result == "MENANG":
                score_player += 1
            elif result == "KALAH":
                score_robot += 1

            state = "MATCH_RESULT" if max(score_player, score_robot) == WIN_SCORE else "ROUND_RESULT"

    # =========================
    # DRAW ROBOT
    # =========================
    rx, ry = 530, 210
    if state == "COUNTDOWN":
        canvas[ry:ry+220, rx:rx+220] = robot_imgs[robot_anim_index]
    elif robot_choice is not None:
        canvas[ry:ry+220, rx:rx+220] = robot_imgs[robot_choice]

    # =========================
    # UI
    # =========================
    canvas[CANVAS_H-CAM_H-30:CANVAS_H-30, 30:30+CAM_W] = cam

    cv2.putText(canvas, f"PLAYER {score_player}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(canvas, f"ROBOT {score_robot}", (30,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.putText(canvas, f"Gesture: {gesture_text.upper()}",
                (30,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

    if state == "LOCKED":
        cv2.putText(canvas, "TEKAN S",
                    (560,170),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,255,255),2)

    if state in ["ROUND_RESULT", "MATCH_RESULT"]:
        cv2.putText(canvas, "TEKAN P UNTUK LANJUT",
                    (500,460),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,
                    (255,255,255),2)

    cv2.imshow("SUIT DIGITAL AI - BO5", canvas)

    # =========================
    # INPUT
    # =========================
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s") and state == "LOCKED":
        locked_choice = stable_choice
        countdown_start = time.time()
        state = "COUNTDOWN"

    if key == ord("p") and state in ["ROUND_RESULT", "MATCH_RESULT"]:
        if state == "MATCH_RESULT":
            score_player = 0
            score_robot = 0
        reset_round()

cap.release()
cv2.destroyAllWindows()

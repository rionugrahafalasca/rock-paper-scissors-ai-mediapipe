import os
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf

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

STATE_LABELS = {
    "PREVIEW": "SIAPKAN GESTUR",
    "LOCKED": "GESTUR TERKUNCI",
    "COUNTDOWN": "PERTANDINGAN DIMULAI",
    "ROUND_RESULT": "HASIL RONDE",
    "MATCH_RESULT": "HASIL PERTANDINGAN",
}

RESULT_STYLES = {
    "MENANG": ((72, 214, 120), "KAMU MENANG RONDE INI"),
    "KALAH": ((96, 96, 255), "ROBOT MENANG RONDE INI"),
    "SERI": ((0, 215, 255), "RONDE BERAKHIR SERI"),
}

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
round_result = None
match_winner = None

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
# DRAW HELPERS
# =========================
def blend_rect(img, top_left, bottom_right, color, alpha=0.6, radius=24):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)


def draw_label_value(img, label, value, pos, value_color=(255, 255, 255)):
    x, y = pos
    cv2.putText(
        img,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (190, 205, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        value,
        (x, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        value_color,
        2,
        cv2.LINE_AA,
    )


def draw_progress_bar(img, top_left, size, progress, color):
    x, y = top_left
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 105, 135), 2, cv2.LINE_AA)
    inner_w = int((w - 4) * max(0.0, min(1.0, progress)))
    if inner_w > 0:
        cv2.rectangle(
            img,
            (x + 2, y + 2),
            (x + 2 + inner_w, y + h - 2),
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_center_text(img, text, y, scale, color, thickness):
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (CANVAS_W - text_w) // 2
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# =========================
# GAME HELPERS
# =========================
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)



def winner(player_choice, enemy_choice):
    if player_choice == enemy_choice:
        return "SERI"
    if (
        (player_choice == 0 and enemy_choice == 1)
        or (player_choice == 1 and enemy_choice == 2)
        or (player_choice == 2 and enemy_choice == 0)
    ):
        return "MENANG"
    return "KALAH"



def reset_round():
    global state, pred_buffer, locked_choice, robot_choice, round_result
    global stable_choice, stable_conf
    pred_buffer.clear()
    locked_choice = None
    robot_choice = None
    round_result = None
    stable_choice = None
    stable_conf = 0.0
    state = "PREVIEW"



def reset_match():
    global score_player, score_robot, match_winner
    score_player = 0
    score_robot = 0
    match_winner = None
    reset_round()


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

        x1, y1 = int(min(xs) * CAM_W), int(min(ys) * CAM_H)
        x2, y2 = int(max(xs) * CAM_W), int(max(ys) * CAM_H)
        pad = 30

        crop = cam[
            max(0, y1 - pad):min(CAM_H, y2 + pad),
            max(0, x1 - pad):min(CAM_W, x2 + pad),
        ]

        if crop.size > 0:
            pred = model.predict(preprocess(crop), verbose=0)[0]
            if max(pred) > CONF_THRESHOLD and state in ["PREVIEW", "LOCKED"]:
                pred_buffer.append(int(np.argmax(pred)))

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

            robot_choice = random.randint(0, 2)
            round_result = winner(locked_choice, robot_choice)
            result_sound.play()

            if round_result == "MENANG":
                score_player += 1
            elif round_result == "KALAH":
                score_robot += 1

            if max(score_player, score_robot) == WIN_SCORE:
                match_winner = "PLAYER" if score_player > score_robot else "ROBOT"
                state = "MATCH_RESULT"
            else:
                state = "ROUND_RESULT"

    # =========================
    # MAIN PANELS
    # =========================
    blend_rect(canvas, (24, 24), (396, 204), (14, 24, 44), 0.78)
    blend_rect(canvas, (24, 420), (412, 690), (14, 24, 44), 0.72)
    blend_rect(canvas, (440, 110), (840, 506), (14, 24, 44), 0.66)
    blend_rect(canvas, (874, 24), (1248, 204), (14, 24, 44), 0.78)
    blend_rect(canvas, (874, 228), (1248, 690), (14, 24, 44), 0.72)

    # =========================
    # DRAW CAMERA PANEL
    # =========================
    canvas[432:432 + CAM_H, 38:38 + CAM_W] = cam
    cv2.rectangle(canvas, (38, 432), (38 + CAM_W, 432 + CAM_H), (105, 224, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "LIVE CAMERA", (44, 464), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (105, 224, 255), 2, cv2.LINE_AA)

    # =========================
    # DRAW ROBOT
    # =========================
    rx, ry = 530, 190
    if state == "COUNTDOWN":
        canvas[ry:ry + 220, rx:rx + 220] = robot_imgs[robot_anim_index]
    elif robot_choice is not None:
        canvas[ry:ry + 220, rx:rx + 220] = robot_imgs[robot_choice]

    draw_center_text(canvas, "SUIT DIGITAL AI", 76, 1.25, (255, 255, 255), 3)
    draw_center_text(canvas, "Best of 5 • Kamera aktif • AI lawan siap", 108, 0.62, (190, 205, 230), 1)

    # Scoreboard
    draw_label_value(canvas, "PLAYER", str(score_player), (52, 70), (72, 214, 120))
    draw_label_value(canvas, "ROBOT", str(score_robot), (210, 70), (96, 96, 255))
    draw_label_value(canvas, "TARGET", f"{WIN_SCORE} WIN", (52, 138), (255, 215, 0))
    draw_label_value(canvas, "STATUS", STATE_LABELS[state], (874, 70), (105, 224, 255))
    draw_label_value(canvas, "GESTUR", gesture_text.upper(), (874, 138), (255, 255, 255))

    progress_color = (72, 214, 120) if stable_conf >= STABLE_RATIO else (0, 215, 255)
    cv2.putText(canvas, "Kestabilan gesture", (52, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (190, 205, 230), 1, cv2.LINE_AA)
    draw_progress_bar(canvas, (52, 252), (300, 24), stable_conf, progress_color)
    cv2.putText(canvas, f"{int(stable_conf * 100)}%", (304, 273), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Versus info
    cv2.putText(canvas, "ARENA PERTANDINGAN", (518, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (190, 205, 230), 1, cv2.LINE_AA)
    draw_center_text(canvas, "PLAYER", 452, 0.8, (72, 214, 120), 2)
    draw_center_text(canvas, "VS", 484, 1.0, (255, 255, 255), 2)
    draw_center_text(canvas, "ROBOT", 516, 0.8, (96, 96, 255), 2)

    # Instruction card
    cv2.putText(canvas, "KONTROL", (900, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (190, 205, 230), 1, cv2.LINE_AA)
    instructions = [
        "1. Tunjukkan batu / gunting / kertas ke kamera.",
        "2. Tunggu meter kestabilan penuh.",
        "3. Tekan S untuk mulai duel.",
        "4. Tekan P untuk lanjut / main lagi.",
        "5. Tekan Q untuk keluar.",
    ]
    for index, text in enumerate(instructions):
        cv2.putText(
            canvas,
            text,
            (900, 312 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    locked_text = classes[locked_choice] if locked_choice is not None else "Belum dikunci"
    enemy_text = classes[robot_choice] if robot_choice is not None else "Menunggu hasil"
    cv2.putText(canvas, "Pilihan Player", (900, 556), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 205, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, locked_text, (900, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (72, 214, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Pilihan Robot", (900, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 205, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, enemy_text, (900, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (96, 96, 255), 2, cv2.LINE_AA)

    # Contextual prompts
    if state == "LOCKED":
        blend_rect(canvas, (464, 532), (816, 610), (0, 143, 255), 0.75)
        draw_center_text(canvas, "GESTUR SIAP • TEKAN S", 582, 0.95, (255, 255, 255), 2)

    if state == "COUNTDOWN":
        remaining = max(0.0, COUNTDOWN_TIME - (time.time() - countdown_start))
        blend_rect(canvas, (474, 530), (806, 620), (27, 38, 59), 0.86)
        draw_center_text(canvas, f"DUEL DIMULAI DALAM {remaining:.1f}", 584, 0.95, (255, 215, 0), 2)

    if state == "ROUND_RESULT" and round_result is not None:
        result_color, result_message = RESULT_STYLES[round_result]
        blend_rect(canvas, (430, 534), (850, 644), (18, 30, 54), 0.88)
        draw_center_text(canvas, result_message, 585, 0.86, result_color, 2)
        draw_center_text(canvas, "Tekan P untuk ronde berikutnya", 621, 0.62, (255, 255, 255), 1)

    if state == "MATCH_RESULT" and round_result is not None and match_winner is not None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (CANVAS_W, CANVAS_H), (6, 10, 24), -1)
        cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)
        blend_rect(canvas, (248, 126), (1032, 622), (12, 22, 44), 0.9)

        title_color = (72, 214, 120) if match_winner == "PLAYER" else (96, 96, 255)
        subtitle = "Kamu berhasil menjadi juara pertandingan!" if match_winner == "PLAYER" else "Robot memenangkan pertandingan kali ini."
        final_score = f"Skor akhir {score_player} - {score_robot}"
        round_color, round_message = RESULT_STYLES[round_result]

        draw_center_text(canvas, "PERTANDINGAN SELESAI", 220, 1.15, (255, 255, 255), 3)
        draw_center_text(canvas, f"PEMENANG: {match_winner}", 300, 1.05, title_color, 3)
        draw_center_text(canvas, subtitle, 350, 0.72, (220, 230, 255), 2)
        draw_center_text(canvas, final_score, 406, 0.9, (255, 215, 0), 2)
        draw_center_text(canvas, round_message, 466, 0.78, round_color, 2)
        draw_center_text(canvas, "Tekan P untuk bermain ulang", 552, 0.84, (255, 255, 255), 2)
        draw_center_text(canvas, "Tekan Q untuk keluar dari game", 592, 0.66, (190, 205, 230), 1)

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
        robot_anim_timer = time.time()
        state = "COUNTDOWN"

    if key == ord("p") and state in ["ROUND_RESULT", "MATCH_RESULT"]:
        if state == "MATCH_RESULT":
            reset_match()
        else:
            reset_round()

cap.release()
cv2.destroyAllWindows()

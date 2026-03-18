import os
import random
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR      = os.path.join(BASE_DIR, "assets")
AUDIO_DIR       = os.path.join(ASSETS_DIR, "audio")
IMAGES_DIR      = os.path.join(ASSETS_DIR, "images")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
BACKGROUNDS_DIR = os.path.join(ASSETS_DIR, "backgrounds")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1280, 720
CAM_W, CAM_H       = 300, 225   

WIN_SCORE      = 3
CONF_THRESHOLD = 0.7
BUFFER_LEN     = 20
STABLE_RATIO   = 0.75
COUNTDOWN_TIME = 3
ANIM_DELAY     = 0.15

STATE_LABELS = {
    "PREVIEW":      "SIAPKAN GESTUR",
    "LOCKED":       "GESTUR TERKUNCI",
    "COUNTDOWN":    "DIMULAI",
    "ROUND_RESULT": "HASIL RONDE",
    "MATCH_RESULT": "HASIL AKHIR",
}

RESULT_STYLES = {
    "MENANG": ((72, 214, 120), "KAMU MENANG RONDE INI"),
    "KALAH":  ((96,  96, 255), "ROBOT MENANG RONDE INI"),
    "SERI":   ((0,  215, 255), "RONDE BERAKHIR SERI"),
}

L_X1, L_X2 =   8, 399      
C_X1, C_X2 = 407, 873      
R_X1, R_X2 = 881, 1272     

T_Y1, T_Y2 =   8, 235      
B_Y1, B_Y2 = 243, 712      

C_W      = C_X2 - C_X1         
L_W      = L_X2 - L_X1          
R_W      = R_X2 - R_X1          
CENTER_X = (C_X1 + C_X2) // 2   

# Warna
COL_PANEL = (14, 24, 44)
COL_MED   = (21, 35, 64)
COL_LABEL = (190, 205, 230)
COL_WHITE = (255, 255, 255)
COL_CYAN  = (105, 224, 255)
COL_GREEN = (72,  214, 120)
COL_PURP  = (96,   96, 255)
COL_GOLD  = (255, 215,   0)
COL_HINT  = (160, 175, 200)

ROBOT_SIZE = 155
ROBOT_X    = CENTER_X - ROBOT_SIZE // 2
ROBOT_Y    = T_Y1 + 68

CAM_PAD = 10
CAM_X   = L_X1 + (L_W - CAM_W) // 2
CAM_Y   = B_Y1 + 52


# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────
pygame.mixer.init()
spin_sound   = pygame.mixer.Sound(os.path.join(AUDIO_DIR, "video-game-text-330163.mp3"))
result_sound = pygame.mixer.Sound(os.path.join(AUDIO_DIR, "game-start-6104.mp3"))
spin_sound.set_volume(0.4)
result_sound.set_volume(0.7)

# ─────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────
bg = cv2.resize(
    cv2.imread(os.path.join(BACKGROUNDS_DIR, "game-background-4956017_1280.webp")),
    (CANVAS_W, CANVAS_H),
)

robot_imgs = {
    k: cv2.resize(cv2.imread(os.path.join(IMAGES_DIR, f)), (ROBOT_SIZE, ROBOT_SIZE))
    for k, f in {0: "batu.png", 1: "gunting.png", 2: "kertas.png"}.items()
}

classes = ["Batu", "Gunting", "Kertas"]
model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "keras_model.h5"), compile=False
)

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ─────────────────────────────────────────────
# GAME STATE
# ─────────────────────────────────────────────
state         = "PREVIEW"
pred_buffer   = deque(maxlen=BUFFER_LEN)
stable_choice = None
stable_conf   = 0.0
locked_choice = None
robot_choice  = None
round_result  = None
match_winner  = None
score_player  = 0
score_robot   = 0
countdown_start  = 0
robot_anim_index = 0
robot_anim_timer = time.time()
spin_playing     = False

cap = cv2.VideoCapture(0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────
def blend_rect(img, x1, y1, x2, y2, color, alpha=0.75):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)


def fit_scale(text, max_w, s0, thick, smin=0.38):
    s = s0
    while s > smin:
        (w, _), _ = cv2.getTextSize(text, FONT, s, thick)
        if w <= max_w:
            return s
        s -= 0.02
    return smin


def put(img, text, x, y, scale, color, thick=1, max_w=None):
    if max_w:
        scale = fit_scale(text, max_w, scale, thick)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def put_c(img, text, y, scale, color, thick=1, cx=None, max_w=None):
    if cx is None:
        cx = CENTER_X
    if max_w:
        scale = fit_scale(text, max_w, scale, thick)
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (cx - tw // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


def prog_bar(img, x, y, w, h, ratio, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 85, 110), 1, cv2.LINE_AA)
    fw = int((w - 2) * max(0.0, min(1.0, ratio)))
    if fw > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fw, y + h - 1),
                      color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# GAME HELPERS
# ─────────────────────────────────────────────
def preprocess(img):
    return np.expand_dims(
        cv2.resize(img, (224, 224)).astype("float32") / 255.0, axis=0
    )


def winner(p, r):
    if p == r:
        return "SERI"
    return "MENANG" if (p, r) in {(0, 1), (1, 2), (2, 0)} else "KALAH"


def reset_round():
    global state, pred_buffer, locked_choice, robot_choice
    global round_result, stable_choice, stable_conf
    pred_buffer.clear()
    locked_choice = robot_choice = round_result = stable_choice = None
    stable_conf = 0.0
    state = "PREVIEW"


def reset_match():
    global score_player, score_robot, match_winner
    score_player = score_robot = 0
    match_winner = None
    reset_round()


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    cam    = cv2.resize(frame, (CAM_W, CAM_H))
    canvas = bg.copy()

    # ── Hand Detection ──────────────────────────────────────────────
    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    gesture_text = "Tidak Jelas"

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(cam, hand, mp_hands.HAND_CONNECTIONS)
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        pad  = 20
        x1   = int(min(xs) * CAM_W);  y1 = int(min(ys) * CAM_H)
        x2   = int(max(xs) * CAM_W);  y2 = int(max(ys) * CAM_H)
        crop = cam[max(0, y1-pad):min(CAM_H, y2+pad),
                   max(0, x1-pad):min(CAM_W, x2+pad)]
        if crop.size > 0:
            pred = model.predict(preprocess(crop), verbose=0)[0]
            if max(pred) > CONF_THRESHOLD and state in ("PREVIEW", "LOCKED"):
                pred_buffer.append(int(np.argmax(pred)))

    # ── Stable Gesture ──────────────────────────────────────────────
    if len(pred_buffer) == BUFFER_LEN:
        stable_choice = max(set(pred_buffer), key=pred_buffer.count)
        stable_conf   = pred_buffer.count(stable_choice) / BUFFER_LEN
        gesture_text  = classes[stable_choice]
        if stable_conf >= STABLE_RATIO and state == "PREVIEW":
            state = "LOCKED"

    # ── Countdown Logic ─────────────────────────────────────────────
    if state == "COUNTDOWN":
        if not spin_playing:
            spin_sound.play(-1)
            spin_playing = True
        if time.time() - robot_anim_timer > ANIM_DELAY:
            robot_anim_index = (robot_anim_index + 1) % 3
            robot_anim_timer = time.time()
        if time.time() - countdown_start >= COUNTDOWN_TIME:
            spin_sound.stop()
            spin_playing  = False
            robot_choice  = random.randint(0, 2)
            round_result  = winner(locked_choice, robot_choice)
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

    # ══════════════════════════════════════════════════════════════════
    # GAMBAR PANEL (6 panel, tidak ada overlap)
    # ══════════════════════════════════════════════════════════════════
    blend_rect(canvas, L_X1, T_Y1, L_X2, T_Y2, COL_PANEL, 0.82)  
    blend_rect(canvas, L_X1, B_Y1, L_X2, B_Y2, COL_PANEL, 0.76)  
    blend_rect(canvas, C_X1, T_Y1, C_X2, T_Y2, COL_PANEL, 0.60)  
    blend_rect(canvas, C_X1, B_Y1, C_X2, B_Y2, COL_PANEL, 0.60)  
    blend_rect(canvas, R_X1, T_Y1, R_X2, T_Y2, COL_PANEL, 0.82)  
    blend_rect(canvas, R_X1, B_Y1, R_X2, B_Y2, COL_PANEL, 0.76)  

    p  = 10                              
    bw = (L_W - p*2 - 6) // 3           
    bx = L_X1 + p
    by = T_Y1 + 10

    for label, val, col in [
        ("PLAYER", str(score_player), COL_GREEN),
        ("ROBOT",  str(score_robot),  COL_PURP),
        ("TARGET", f"{WIN_SCORE} WIN", COL_GOLD),
    ]:
        blend_rect(canvas, bx, by, bx + bw, by + 70, COL_MED, 0.88)
        put(canvas, label, bx + 5, by + 20, 0.46, COL_LABEL, 1, bw - 10)
        put(canvas, val,   bx + 5, by + 58, 0.92, col,       2, bw - 10)
        bx += bw + 3

    bar_x = L_X1 + p
    bar_y = by + 80       
    bar_w = L_W - p*2
    put(canvas, "Kestabilan Gesture", bar_x, bar_y, 0.46, COL_LABEL, 1)
    p_col = COL_GREEN if stable_conf >= STABLE_RATIO else COL_CYAN
    prog_bar(canvas, bar_x, bar_y + 10, bar_w - 44, 13, stable_conf, p_col)
    put(canvas, f"{int(stable_conf*100)}%",
        bar_x + bar_w - 38, bar_y + 21, 0.48, COL_WHITE, 1)

    # Gesture label
    g_y = bar_y + 40
    put(canvas, "Gesture:", bar_x, g_y, 0.48, COL_LABEL, 1, 68)
    put(canvas, gesture_text.upper(), bar_x + 72, g_y, 0.66, COL_CYAN, 2,
        bar_w - 72)

    lbx = L_X1 + p
    put(canvas, "LIVE CAMERA", lbx, B_Y1 + 22, 0.58, COL_LABEL, 1, L_W - p*2)
    put(canvas, "Pastikan tangan di tengah frame.",
        lbx, B_Y1 + 42, 0.42, COL_HINT, 1, L_W - p*2)

    cx2 = CAM_X + CAM_W
    cy2 = CAM_Y + CAM_H
    if cy2 <= CANVAS_H and cx2 <= CANVAS_W:
        canvas[CAM_Y:cy2, CAM_X:cx2] = cam
        cv2.rectangle(canvas, (CAM_X, CAM_Y), (cx2, cy2), COL_CYAN, 2, cv2.LINE_AA)

    iy = cy2 + 14
    if iy + 30 < B_Y2:
        put(canvas, "Pilihan Anda:", lbx, iy, 0.48, COL_LABEL, 1, L_W - p*2)
        lk_txt = classes[locked_choice] if locked_choice is not None else "Belum dikunci"
        put(canvas, lk_txt, lbx, iy + 26, 0.70, COL_GREEN, 2, L_W - p*2)

    put_c(canvas, "SUIT DIGITAL AI",
          T_Y1 + 30, 0.96, COL_WHITE, 3, max_w=C_W - 20)
    put_c(canvas, "Best of 5  |  Kamera Aktif  |  AI Siap",
          T_Y1 + 52, 0.44, COL_HINT, 1, max_w=C_W - 24)

    if state == "COUNTDOWN":
        rimg = robot_imgs[robot_anim_index]
    elif robot_choice is not None:
        rimg = robot_imgs[robot_choice]
    else:
        rimg = None

    if rimg is not None:
        ry2 = ROBOT_Y + ROBOT_SIZE
        rx2 = ROBOT_X + ROBOT_SIZE
        if ry2 <= CANVAS_H and rx2 <= CANVAS_W:
            canvas[ROBOT_Y:ry2, ROBOT_X:rx2] = rimg

    mw = C_W - 40    

    put_c(canvas, "ARENA PERTANDINGAN",  B_Y1 + 24, 0.56, COL_LABEL, 1, max_w=mw)

    vs_y = B_Y1 + 58
    put_c(canvas, "PLAYER", vs_y, 0.58, COL_GREEN, 2, cx=CENTER_X - 110, max_w=120)
    put_c(canvas, "VS",     vs_y, 0.76, COL_WHITE, 2, cx=CENTER_X,       max_w=52)
    put_c(canvas, "ROBOT",  vs_y, 0.58, COL_PURP,  2, cx=CENTER_X + 110, max_w=120)

    rb_lbl = classes[robot_choice] if robot_choice is not None else "Menunggu..."
    put_c(canvas, f"Robot memilih: {rb_lbl}", B_Y1 + 86, 0.50, COL_PURP, 1, max_w=mw)

    cv2.line(canvas,
             (C_X1 + 20, B_Y1 + 100), (C_X2 - 20, B_Y1 + 100),
             (70, 85, 110), 1, cv2.LINE_AA)

    pz_top = B_Y1 + 110    
    pz_mw  = C_W - 50

    if state == "PREVIEW":
        put_c(canvas, "Tunjukkan gestur ke kamera,",
              pz_top + 36, 0.52, COL_HINT, 1, max_w=pz_mw)
        put_c(canvas, "tunggu indikator kestabilan penuh.",
              pz_top + 64, 0.52, COL_HINT, 1, max_w=pz_mw)

    elif state == "LOCKED":
        blend_rect(canvas,
                   C_X1 + 22, pz_top + 8,
                   C_X2 - 22, pz_top + 80,
                   (0, 80, 170), 0.84)
        put_c(canvas, "GESTUR TERKUNCI",
              pz_top + 42, 0.78, COL_WHITE, 2, max_w=pz_mw)
        put_c(canvas, "Tekan  S  untuk mulai duel",
              pz_top + 68, 0.56, COL_CYAN,  1, max_w=pz_mw)

    elif state == "COUNTDOWN":
        remaining = max(0.0, COUNTDOWN_TIME - (time.time() - countdown_start))
        blend_rect(canvas,
                   C_X1 + 22, pz_top + 8,
                   C_X2 - 22, pz_top + 80,
                   (18, 28, 48), 0.90)
        put_c(canvas, f"DUEL DALAM  {remaining:.1f}s",
              pz_top + 54, 0.82, COL_GOLD, 2, max_w=pz_mw)

    elif state == "ROUND_RESULT" and round_result is not None:
        r_col, r_msg = RESULT_STYLES[round_result]
        blend_rect(canvas,
                   C_X1 + 22, pz_top + 8,
                   C_X2 - 22, pz_top + 96,
                   (14, 22, 46), 0.92)
        put_c(canvas, r_msg,
              pz_top + 50, 0.78, r_col,    2, max_w=pz_mw)
        put_c(canvas, "Tekan  P  untuk ronde berikutnya",
              pz_top + 80, 0.52, COL_WHITE, 1, max_w=pz_mw)

    # Hint AI — selalu di paling bawah panel tengah-bawah
    put_c(canvas, "AI mengacak pilihan saat countdown selesai.",
          B_Y2 - 18, 0.40, COL_HINT, 1, max_w=mw)

    # ══════════════════════════════════════════════════════════════════
    # [5] KANAN-ATAS  — Status Pertandingan
    #     Y zone: T_Y1 ... T_Y2   (8..235)
    # ══════════════════════════════════════════════════════════════════
    rp  = R_X1 + p
    rw  = R_W - p*2

    put(canvas, "STATUS PERTANDINGAN",  rp, T_Y1 + 22, 0.56, COL_LABEL, 1, rw)

    put(canvas, "Mode:",          rp,      T_Y1 + 56, 0.46, COL_LABEL, 1, 52)
    put(canvas, STATE_LABELS[state],
        rp + 58, T_Y1 + 56, 0.60, COL_CYAN, 2, rw - 58)

    put(canvas, "Gesture:",       rp,      T_Y1 + 90, 0.46, COL_LABEL, 1, 66)
    put(canvas, gesture_text.upper(),
        rp + 72, T_Y1 + 90, 0.62, COL_WHITE, 2, rw - 72)

    # Skor — pakai ASCII biasa agar tidak muncul "???"
    skor_str = f"Skor:  P {score_player}  vs  R {score_robot}"
    put(canvas, skor_str,          rp, T_Y1 + 124, 0.52, COL_GOLD, 1, rw)

    put(canvas, f"Target: {WIN_SCORE} kemenangan",
        rp, T_Y1 + 154, 0.46, COL_LABEL, 1, rw)

    put(canvas, "KONTROL PERMAINAN", rp, B_Y1 + 24, 0.56, COL_LABEL, 1, rw)

    for i, txt in enumerate([
        "1. Tunjukkan gesture ke kamera.",
        "2. Tunggu indikator kestabilan penuh.",
        "3. Tekan S untuk mulai duel.",
        "4. Tekan P untuk lanjut / replay.",
        "5. Tekan Q untuk keluar game.",
    ]):
        put(canvas, txt, rp, B_Y1 + 54 + i * 32, 0.46, (215, 225, 250), 1, rw)

    cv2.line(canvas,
             (R_X1 + p, B_Y1 + 228), (R_X2 - p, B_Y1 + 228),
             (70, 85, 110), 1, cv2.LINE_AA)

    put(canvas, "PILIHAN SAAT INI", rp, B_Y1 + 250, 0.56, COL_LABEL, 1, rw)

    lk_disp = classes[locked_choice] if locked_choice is not None else "Belum dikunci"
    rb_disp = classes[robot_choice]  if robot_choice  is not None else "Menunggu hasil"

    put(canvas, "Player :", rp,      B_Y1 + 282, 0.46, COL_LABEL, 1, 70)
    put(canvas, lk_disp,   rp + 74, B_Y1 + 282, 0.64, COL_GREEN,  2, rw - 74)

    put(canvas, "Robot  :", rp,      B_Y1 + 314, 0.46, COL_LABEL, 1, 70)
    put(canvas, rb_disp,   rp + 74, B_Y1 + 314, 0.64, COL_PURP,   2, rw - 74)

    # ══════════════════════════════════════════════════════════════════
    # MATCH RESULT — Overlay di atas semua panel
    # ══════════════════════════════════════════════════════════════════
    if state == "MATCH_RESULT" and match_winner is not None:
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (CANVAS_W, CANVAS_H), (5, 8, 20), -1)
        cv2.addWeighted(ov, 0.65, canvas, 0.35, 0, canvas)

        bx1, by1, bx2, by2_ = 210, 90, 1070, 640
        blend_rect(canvas, bx1, by1, bx2, by2_, (10, 20, 42), 0.94)
        bcx  = (bx1 + bx2) // 2
        bmw  = bx2 - bx1 - 60

        t_col = COL_GREEN if match_winner == "PLAYER" else COL_PURP
        sub   = ("Kamu berhasil jadi juara!" if match_winner == "PLAYER"
                 else "Robot memenangkan pertandingan.")
        # Gunakan tanda minus ASCII biasa
        sc_str = f"Skor Akhir: {score_player} - {score_robot}"
        r_col, r_msg = RESULT_STYLES[round_result]

        put_c(canvas, "PERTANDINGAN SELESAI",
              by1 + 76,  1.00, COL_WHITE, 3, cx=bcx, max_w=bmw)
        put_c(canvas, f"PEMENANG: {match_winner}",
              by1 + 148, 0.94, t_col,     3, cx=bcx, max_w=bmw)
        put_c(canvas, sub,
              by1 + 204, 0.60, (215, 225, 250), 1, cx=bcx, max_w=bmw)
        put_c(canvas, sc_str,
              by1 + 262, 0.80, COL_GOLD,  2, cx=bcx, max_w=bmw)
        put_c(canvas, r_msg,
              by1 + 318, 0.68, r_col,     2, cx=bcx, max_w=bmw)

        cv2.line(canvas, (bx1+60, by1+358), (bx2-60, by1+358),
                 (70, 85, 110), 1, cv2.LINE_AA)

        put_c(canvas, "Tekan P untuk bermain ulang",
              by1 + 408, 0.74, COL_WHITE, 2, cx=bcx, max_w=bmw)
        put_c(canvas, "Tekan Q untuk keluar",
              by1 + 454, 0.58, COL_HINT,  1, cx=bcx, max_w=bmw)

    # ──────────────────────────────────────────────────────────────────
    cv2.imshow("SUIT DIGITAL AI - BO5", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and state == "LOCKED":
        locked_choice    = stable_choice
        countdown_start  = time.time()
        robot_anim_timer = time.time()
        robot_anim_index = 0
        state = "COUNTDOWN"
    if key == ord("p") and state in ("ROUND_RESULT", "MATCH_RESULT"):
        reset_match() if state == "MATCH_RESULT" else reset_round()

cap.release()
cv2.destroyAllWindows()

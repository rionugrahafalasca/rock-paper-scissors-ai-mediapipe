import cv2
import os
import time
import mediapipe as mp

# =====================
# CONFIG
# =====================
DATASET_DIR = "dataset"
IMG_SIZE = 224
SAVE_DELAY = 0.4  # detik (biar ga spam)
PADDING = 80

LABELS = {
    "1": "Batu",
    "2": "Gunting",
    "3": "Kertas"
}

# =====================
# FOLDER SETUP
# =====================
for label in LABELS.values():
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)
last_save_time = 0

print("=== DATASET COLLECTOR ===")
print("1 = BATU | 2 = GUNTING | 3 = KERTAS | Q = QUIT")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    display = frame.copy()
    hand_crop = None
    hand_detected = False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_detected = True
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        x1, y1 = int(min(xs)*w), int(min(ys)*h)
        x2, y2 = int(max(xs)*w), int(max(ys)*h)

        x1 = max(0, x1-PADDING)
        y1 = max(0, y1-PADDING)
        x2 = min(w, x2+PADDING)
        y2 = min(h, y2+PADDING)

        cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
        hand_crop = frame[y1:y2, x1:x2]

    # UI
    cv2.putText(display,"1:Batu  2:Gunting  3:Kertas",
                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.putText(display,
                "Tangan terdeteksi" if hand_detected else "Arahkan tangan",
                (20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                (0,255,0) if hand_detected else (0,0,255),2)

    cv2.imshow("Collect Dataset", display)

    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    if chr(key) in LABELS and hand_crop is not None:
        if now - last_save_time > SAVE_DELAY:
            label = LABELS[chr(key)]
            count = len(os.listdir(os.path.join(DATASET_DIR, label)))
            img_path = os.path.join(DATASET_DIR, label, f"{label}_{count}.jpg")

            crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(img_path, crop)

            print(f"[SAVED] {img_path}")
            last_save_time = now

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
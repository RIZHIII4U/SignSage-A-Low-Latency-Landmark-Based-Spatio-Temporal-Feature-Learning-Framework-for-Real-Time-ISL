import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import pyttsx3
import mediapipe as mp
from feature_extractor import extract_features

# --- CONFIG ---
MODEL_PATH = "isl_dynamic_lstm.h5"
CLASSES = ["Bye", "Good", "Hello", "Morning", "Nice", "no", "Thankyou", "Welcome", "work", "yes"]
SEQUENCE_LENGTH = 20
CONF_THRESHOLD = 0.70   # Slightly lower for real-time stability

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
last_spoken = ""

engine = pyttsx3.init()
engine.setProperty('rate', 150)

print("✅ Ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    h1 = np.zeros(68, dtype=np.float32)
    h2 = np.zeros(68, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, (lm_list, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            if i > 1:
                break

            mp_draw.draw_landmarks(frame, lm_list, mp.solutions.hands.HAND_CONNECTIONS)

            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in lm_list.landmark],
                dtype=np.float32
            )
            side = handedness.classification[0].label
            feats = extract_features(coords, side).astype(np.float32)

            if i == 0:
                h1 = feats
            else:
                h2 = feats
    else:
        # Slowly clear if no hands
        if len(frame_buffer) > 0:
            frame_buffer.popleft()

    combined = np.concatenate([h1, h2])
    frame_buffer.append(combined)

    text_overlay = "Waiting..."
    conf_overlay = ""

    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
        preds = model.predict(input_data, verbose=0)[0]
        max_idx = np.argmax(preds)
        confidence = preds[max_idx]

        conf_overlay = f"Conf: {confidence*100:.1f}%"

        if confidence >= CONF_THRESHOLD:
            sign = CLASSES[max_idx]
            text_overlay = f"{sign} ({confidence*100:.1f}%)"

            if sign != last_spoken:
                print(f"👉 Detected: {sign} ({confidence*100:.1f}%)")
                engine.say(sign)
                engine.runAndWait()
                last_spoken = sign
                frame_buffer.clear()
        else:
            text_overlay = "Low confidence"

    # Overlay text
    cv2.putText(frame, text_overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    if conf_overlay:
        cv2.putText(frame, conf_overlay, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("SignSage Dynamic ISL (CLI Mode)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mp_hands.close()
print("👋 Exiting.")
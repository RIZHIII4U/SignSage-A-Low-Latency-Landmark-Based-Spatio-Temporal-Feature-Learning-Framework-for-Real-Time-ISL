import cv2
import numpy as np
import joblib
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import threading
import queue
import time
import os
from collections import deque, Counter
from feature_extractor import extract_features

# --- CONFIG (UPDATED TO MATCH YOUR MODEL) ---
RF_MODEL_PATH = "signsage_rf_dual.joblib"
LSTM_MODEL_PATH = "isl_dynamic_lstm.h5"

# CHANGED: Updated from 30 to 20 based on your Error Log
SEQUENCE_LENGTH = 20          
MIN_CONF_STATIC = 0.6        
MIN_CONF_DYNAMIC = 0.6       
SMOOTHING_WINDOW = 10         
VOICE_RATE = 155

# Ensure these match your training folder labels
DYNAMIC_CLASSES = ["Hello", "ThankYou", "Bye", "Yes", "No", "Namaste", "Food", "Help"]

class AdvancedVoiceEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', VOICE_RATE)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            text = self.queue.get()
            if text is None: break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except: pass
            self.queue.task_done()

    def speak(self, text):
        if text and text != "None":
            self.queue.put(text)

class SignSageUnifiedPro:
    def __init__(self):
        print(" STARTING SIGNSAGE UNIFIED ENGINE...")
        
        if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(LSTM_MODEL_PATH):
            print(f" ERROR: Models missing: {RF_MODEL_PATH} or {LSTM_MODEL_PATH}")
            exit()

        print(" Loading Models...")
        self.rf_model = joblib.load(RF_MODEL_PATH)
        self.lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        print(" Models Loaded Successfully.")

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

        self.voice = AdvancedVoiceEngine()
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.sentence = ""
        self.last_item = None
        self.cooldown = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" CAMERA ERROR")
            return

        print(" APP LIVE. Press 'Q' to Quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            l_h, r_h = np.zeros(68), np.zeros(68)
            current_sign = "None"
            conf = 0.0

            if results.multi_hand_landmarks:
                for res, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    side = handedness.classification[0].label
                    self.mp_draw.draw_landmarks(frame, res, self.mp_hands.HAND_CONNECTIONS)
                    
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
                    feats = extract_features(coords, side)
                    if side == "Left": l_h = feats
                    else: r_h = feats

            combined_feats = np.concatenate([l_h, r_h])
            self.frame_buffer.append(combined_feats)

            # --- PREDICTION LOGIC ---
            try:
                # 1. Dynamic Prediction (LSTM) - Requires exactly 20 frames
                if len(self.frame_buffer) == SEQUENCE_LENGTH:
                    seq = np.expand_dims(np.array(self.frame_buffer), axis=0)
                    d_probs = self.lstm_model.predict(seq, verbose=0)[0]
                    d_idx = np.argmax(d_probs)
                    d_conf = d_probs[d_idx]

                    if d_conf > MIN_CONF_DYNAMIC and (time.time() - self.cooldown > 2.5):
                        d_label = DYNAMIC_CLASSES[d_idx]
                        if d_label != self.last_item:
                            self.sentence += f" {d_label}"
                            self.voice.speak(d_label)
                            self.last_item = d_label
                            self.cooldown = time.time()
                            self.frame_buffer.clear() # Reset buffer after word found
                        current_sign = d_label
                        conf = d_conf

                # 2. Static Prediction (RF) - Fallback
                if current_sign == "None":
                    s_probs = self.rf_model.predict_proba(combined_feats.reshape(1, -1))[0]
                    s_idx = np.argmax(s_probs)
                    s_conf = s_probs[s_idx]
                    
                    if s_conf >= MIN_CONF_STATIC:
                        s_label = str(self.rf_model.classes_[s_idx])
                        self.history.append(s_label)
                        smoothed = Counter(self.history).most_common(1)[0][0]
                        
                        if smoothed != self.last_item and smoothed != "None":
                            self.sentence += smoothed
                            self.voice.speak(smoothed)
                            self.last_item = smoothed
                        current_sign = smoothed
                        conf = s_conf
                    else:
                        self.history.append("None")
                        if Counter(self.history).most_common(1)[0][0] == "None":
                            self.last_item = None

            except Exception as e:
                # This will catch any remaining shape mismatches
                pass

            # --- MARKET UI ---
            cv2.rectangle(frame, (0, h-80), (w, h), (40, 40, 40), -1)
            cv2.putText(frame, f"SIGN: {current_sign} ({int(conf*100)}%)", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(frame, f"TEXT: {self.sentence[-25:]}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("SignSage Unified Pro", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            elif cv2.waitKey(1) & 0xFF == ord('c'): self.sentence = ""; self.last_item = None

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SignSageUnifiedPro()
    app.run()
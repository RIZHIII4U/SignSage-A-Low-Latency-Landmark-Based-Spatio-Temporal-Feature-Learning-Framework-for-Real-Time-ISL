import cv2
import numpy as np
import joblib
import mediapipe as mp
import pyttsx3
import threading
import queue
from collections import deque, Counter
from feature_extractor import extract_features

# --- CONFIG ---
MODEL_FILE = "signsage_rf_dual.joblib"
MIN_CONFIDENCE = 0.60         # Minimum probability to accept a prediction
SMOOTHING_WINDOW = 12          # Number of frames to average for stability
VOICE_RATE = 150               # Speech speed

def tts_worker(voice_queue):
    """Background worker to handle speech without freezing the video."""
    engine = pyttsx3.init()
    engine.setProperty('rate', VOICE_RATE)
    while True:
        text = voice_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()

class SignSageISLPro:
    def __init__(self):
        print(" Initializing SignSage ISL Pro Engine...")
        self.rf = joblib.load(MODEL_FILE)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Background Voice Thread
        self.voice_queue = queue.Queue()
        self.voice_thread = threading.Thread(target=tts_worker, args=(self.voice_queue,), daemon=True)
        self.voice_thread.start()

        self.sentence = ""
        self.last_char = None
        self.history = deque(maxlen=SMOOTHING_WINDOW)

    def run(self):
        cap = cv2.VideoCapture(0)
        print(" System Live. [Q]: Exit | [C]: Clear | [Space]: Space | [B]: Backspace")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            # MediaPipe requires RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Initialize empty features for both hands (68 each = 136 total)
            l_h, r_h = np.zeros(68), np.zeros(68)
            pred = "None"
            conf = 0.0

            if results.multi_hand_landmarks:
                for res, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    side = handedness.classification[0].label # 'Left' or 'Right'
                    
                    # Draw landmarks on frame
                    self.mp_draw.draw_landmarks(frame, res, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract coordinates (21 landmarks * 3 coords = 63 + 5 relative distances = 68)
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32)
                    feats = extract_features(coords, side)
                    
                    if side == "Left": l_h = feats
                    else: r_h = feats

                # Combined input for the model
                input_data = np.concatenate([l_h, r_h]).reshape(1, -1)
                
                # Prediction Logic
                probs = self.rf.predict_proba(input_data)[0]
                max_idx = np.argmax(probs)
                conf = probs[max_idx]
                
                if conf >= MIN_CONFIDENCE:
                    pred = str(self.rf.classes_[max_idx])

            # --- Smoothing & Voice Trigger ---
            self.history.append(pred)
            smoothed = Counter(self.history).most_common(1)[0][0]

            if smoothed != "None" and smoothed != self.last_char:
                self.sentence += smoothed
                self.voice_queue.put(smoothed) # Send to background voice thread
                self.last_char = smoothed
            elif smoothed == "None":
                self.last_char = None # Reset when hands are lowered

            # --- Professional UI Overlay ---
            # Text Bar
            cv2.rectangle(frame, (0, 400), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
            cv2.putText(frame, f"SIGN: {smoothed} ({int(conf*100)}%)", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(frame, f"TEXT: {self.sentence}", (20, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.imshow("SignSage ISL Pro", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): self.sentence = ""
            elif key == ord(' '): 
                self.sentence += " "
                self.voice_queue.put("space")
            elif key == ord('b') and self.sentence:
                self.sentence = self.sentence[:-1]

        cap.release()
        cv2.destroyAllWindows()
        self.voice_queue.put(None) # Stop voice thread
        self.voice_thread.join()

if __name__ == "__main__":
    SignSageISLPro().run()
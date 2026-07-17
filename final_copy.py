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

# CONFIGURATION

RF_MODEL_PATH = "signsage_rf_dual.joblib"
LSTM_MODEL_PATH = "isl_dynamic_lstm.h5"

DYNAMIC_CLASSES = np.load(
    "dynamic_classes.npy",
    allow_pickle=True
)

SEQUENCE_LENGTH = 20

# Adaptive motion threshold
MOTION_THRESHOLD = None

MOTION_CALIBRATION_FRAMES = 100

LOW_MOTION_CONFIDENCE = 0.60
HIGH_MOTION_CONFIDENCE = 0.60

SMOOTHING_WINDOW = 10
VOICE_RATE = 155

# VOICE ENGINE

class AdvancedVoiceEngine:

    def __init__(self):

        self.queue = queue.Queue()

        self.engine = pyttsx3.init()

        self.engine.setProperty(
            "rate",
            VOICE_RATE
        )

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    def _run(self):

        while True:

            text = self.queue.get()

            if text is None:
                break

            try:
                self.engine.say(text)
                self.engine.runAndWait()

            except Exception:
                pass

            self.queue.task_done()

    def speak(self,text):

        if text and text != "None":

            self.queue.put(text)

# SIGNSAGE ENGINE

class SignSageUnifiedPro:

    def __init__(self):

        print(
            "STARTING SIGNSAGE UNIFIED ENGINE..."
        )

        if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(LSTM_MODEL_PATH):

            print(
                "ERROR: MODEL FILES MISSING"
            )

            exit()

        print(
            "Loading Models..."
        )


        self.rf_model = joblib.load(
            RF_MODEL_PATH
        )


        self.lstm_model = tf.keras.models.load_model(
            LSTM_MODEL_PATH
        )


        print(
            "Models Loaded Successfully"
        )

        # MediaPipe

        self.mp_hands = mp.solutions.hands

        self.mp_draw = mp.solutions.drawing_utils


        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7
        )

        # Runtime

        self.voice = AdvancedVoiceEngine()


        self.frame_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.history = deque(
            maxlen=SMOOTHING_WINDOW
        )

        self.sentence = ""

        self.last_item = None

        self.cooldown = 0
		# Motion threshold calibration
		self.motion_samples = deque(
		maxlen=MOTION_CALIBRATION_FRAMES
		)

    self.motion_threshold_ready = False
    
	# ADAPTIVE MOTION THRESHOLD CALIBRATION

def calibrate_motion_threshold(self, motion_energy):

    global MOTION_THRESHOLD

    if self.motion_threshold_ready:

        return


    self.motion_samples.append(
        motion_energy
    )


    if len(self.motion_samples) == MOTION_CALIBRATION_FRAMES:


        values = np.array(
            self.motion_samples
        )

        MOTION_THRESHOLD = (
            np.mean(values)
            +
            0.5*np.std(values)
        )

        self.motion_threshold_ready = True

        print(
            f"Adaptive Motion Threshold Selected: {MOTION_THRESHOLD:.4f}"
        )
    def compute_motion_energy(self):

        if len(self.frame_buffer) < 2:

            return 0.0

        energy = 0.0

        for i in range(
            1,
            len(self.frame_buffer)
        ):

            previous = self.frame_buffer[i-1]
            current = self.frame_buffer[i]

            energy += np.linalg.norm(
                current - previous
            )

        return energy / (
            len(self.frame_buffer)-1
        )
   
    # LOW MOTION RECOGNITION
    # Random Forest Expert
  
    def predict_low_motion(self,features):

        probs = self.rf_model.predict_proba(
            features.reshape(1,-1)
        )[0]

        index = np.argmax(probs)

        confidence = probs[index]

        if confidence >= LOW_MOTION_CONFIDENCE:

            return (
                str(self.rf_model.classes_[index]),
                confidence
            )

        return "None", confidence
    
    # HIGH MOTION RECOGNITION
    # LSTM Expert
   
    def predict_high_motion(self):

        sequence = np.expand_dims(
            np.array(self.frame_buffer),
            axis=0
        )

        probs = self.lstm_model.predict(
            sequence,
            verbose=0
        )[0]

        index = np.argmax(probs)

        confidence = probs[index]

        if confidence >= HIGH_MOTION_CONFIDENCE:

            return (
                str(DYNAMIC_CLASSES[index]),
                confidence
            )

        return "None", confidence
		  
    # MAIN RECOGNITION PIPELINE

    def run(self):

        cap = cv2.VideoCapture(0)


        if not cap.isOpened():

            print(
                "CAMERA ERROR"
            )

            return

        print(
            "APP LIVE. Press Q to Quit."
        )

        prev_time = time.time()

        while cap.isOpened():

            start_time = time.time()

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(
                frame,
                1
            )

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(
                rgb
            )
          # Hand Feature Extraction
           
            left_hand = np.zeros(68)

            right_hand = np.zeros(68)


            if results.multi_hand_landmarks:

                for hand, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    side = handedness.classification[0].label

                    self.mp_draw.draw_landmarks(
                        frame,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    coords = np.array(
                        [
                            [
                                lm.x,
                                lm.y,
                                lm.z
                            ]

                            for lm in hand.landmark
                        ],
                        dtype=np.float32
                    )

                   features = extract_features(
                        coords,
                        side
                    )

                    if side == "Left":

                        left_hand = features

                    else:

                        right_hand = features

            combined_features = np.concatenate(
                [
                    left_hand,
                    right_hand
                ]
            )

            self.frame_buffer.append(
                combined_features
            )
            current_sign = "None"

            confidence = 0.0

            motion_state = "WAITING"

            selected_model = "NONE"
          
        # Motion-Aware Routing
          
            try:

                if len(self.frame_buffer) == SEQUENCE_LENGTH:

                    motion_energy = (
                        self.compute_motion_energy()
                    )
					self.calibrate_motion_threshold(
    motion_energy
)

if not self.motion_threshold_ready:

    current_sign = "Calibrating..."

    continue
                 
                    # LOW MOTION
                    # RF Expert
                    
                    if motion_energy < MOTION_THRESHOLD:

                        motion_state = "LOW MOTION"

                        selected_model = "RF"

                        current_sign, confidence = (
                            self.predict_low_motion(
                                combined_features
                            )
                        )

                    # HIGH MOTION
                    # LSTM Expert
                
                    else:

                       motion_state = "HIGH MOTION"

                        selected_model = "LSTM"


                        current_sign, confidence = (
                            self.predict_high_motion()
                        )
                else:

                    motion_energy = 0.0

            except Exception:

                motion_energy = 0.0

                current_sign = "None"

           # Temporal Smoothing
        
            if current_sign != "None":

                self.history.append(
                    current_sign
                )

                smoothed_sign = Counter(
                    self.history
                ).most_common(1)[0][0]

                current_sign = smoothed_sign

                if (
                    current_sign != self.last_item
                    and current_sign != "None"
                ):


                    self.sentence += (
                        " " + current_sign
                    )

                    self.voice.speak(
                        current_sign
                    )

                    self.last_item = current_sign

            else:

                self.history.append(
                    "None"
                )

          # Performance Monitoring
     
            inference_time = (
                time.time() - start_time
            )

            fps = 1 / (
                time.time()-prev_time
            )

            prev_time = time.time()

         
            # UI
        
            cv2.rectangle(
                frame,
                (0,h-120),
                (w,h),
                (40,40,40),
                -1
            )

			cv2.putText(
    frame,
    f"Threshold:{MOTION_THRESHOLD if MOTION_THRESHOLD else 0:.4f}",
    (20,140),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,255,255),
    2
)
            cv2.putText(
                frame,
                f"SIGN: {current_sign} ({confidence*100:.1f}%)",
                (20,40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0,255,100),
                2
            )


            cv2.putText(
                frame,
                f"STATE: {motion_state}",
                (20,75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2
            )

            cv2.putText(
                frame,
                f"MODEL: {selected_model}",
                (20,105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"ME:{motion_energy:.4f} FPS:{fps:.1f} Latency:{inference_time*1000:.1f}ms",
                (20,h-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255,255,255),
                2
            )
            cv2.imshow(
                "SignSage Unified Pro",
                frame
            )
            key = cv2.waitKey(1) & 0xff

            if key == ord('q'):

                break

            elif key == ord('c'):

                self.sentence = ""

                self.last_item = None

        cap.release()

        cv2.destroyAllWindows()

# APPLICATION ENTRY POINT

if __name__ == "__main__":

    app = SignSageUnifiedPro()

    app.run()

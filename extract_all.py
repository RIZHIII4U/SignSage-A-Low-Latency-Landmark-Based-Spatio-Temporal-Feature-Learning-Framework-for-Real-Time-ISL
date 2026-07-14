import os
import cv2
import mediapipe as mp
import numpy as np
from feature_extractor import extract_features

DATA_DIR = "dataset_static_aug"
OUTPUT_X = "X_isl_features.npy"
OUTPUT_Y = "y_isl_labels.npy"

def main():
    X, y = [], []
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    detector = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

    for label in classes:
        label_path = os.path.join(DATA_DIR, label)
        images = [f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f" Processing Class '{label}': {len(images)} images")

        for idx, img_name in enumerate(images):
            img = cv2.imread(os.path.join(label_path, img_name))
            if img is None: continue

            # Progress feedback every 500 images
            if idx % 500 == 0:
                print(f"   → [{label}] Progress: {idx}/{len(images)}")

            results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            left_feats = np.zeros(68, dtype=np.float32)
            right_feats = np.zeros(68, dtype=np.float32)

            if results.multi_hand_landmarks:
                for res, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    side = handedness.classification[0].label
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark])
                    feats = extract_features(coords, side)
                    if side == "Left": left_feats = feats
                    else: right_feats = feats

            if np.any(left_feats) or np.any(right_feats):
                X.append(np.concatenate([left_feats, right_feats]))
                y.append(label)

    detector.close()
    np.save(OUTPUT_X, np.array(X))
    np.save(OUTPUT_Y, np.array(y))
    print(f" Extraction Complete. Dataset shape: {np.array(X).shape}")

if __name__ == "__main__":
    main()
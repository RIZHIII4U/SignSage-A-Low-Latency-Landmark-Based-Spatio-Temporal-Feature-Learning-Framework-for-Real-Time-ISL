import numpy as np
import math

def extract_features(landmarks: np.ndarray, handedness: str) -> np.ndarray:
    """
    Robust 68-dim feature vector for ISL (A-Z, 1-9).
    """
    pts = landmarks.astype(np.float32).copy()

    # 1. Mirror left hand for consistency
    if handedness == "Left":
        pts[:, 0] *= -1

    # 2. Translate: wrist (0) is origin
    wrist = pts[0].copy()
    pts -= wrist

    # 3. Scale normalize
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6: scale = 1.0
    pts /= scale

    # 4. Tip-to-Wrist distances (Essential for numbers 1-9)
    tips = [4, 8, 12, 16, 20]
    tip_dists = [np.linalg.norm(pts[t]) for t in tips]

    # 5. Point-to-Point distances
    p2p = [
        np.linalg.norm(pts[8] - pts[12]),   # index–middle
        np.linalg.norm(pts[12] - pts[16]),  # middle–ring
        np.linalg.norm(pts[16] - pts[20]),  # ring–pinky
        np.linalg.norm(pts[4] - pts[8]),    # thumb–index
        np.linalg.norm(pts[4] - pts[5]),    # thumb–palm
        np.linalg.norm(pts[0] - pts[5])     # palm width
    ]

    # 6. XY coordinates (42 features)
    xy_coords = pts[:, :2].flatten()

    # 7. Basic angles (10 features)
    angles = [math.atan2(pts[i][1], pts[i][0]) for i in range(5, 15)]

    # 8. Padding to reach exactly 68
    combined = np.concatenate([tip_dists, p2p, xy_coords, angles])
    padding = np.zeros(68 - len(combined), dtype=np.float32)

    return np.concatenate([combined, padding]).astype(np.float32)
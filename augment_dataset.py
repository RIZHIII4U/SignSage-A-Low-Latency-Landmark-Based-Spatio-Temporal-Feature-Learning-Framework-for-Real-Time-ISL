import os
import cv2
import numpy as np
from tqdm import tqdm

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
INPUT_DIR = "dataset_static"
OUTPUT_DIR = "dataset_static_aug"
AUGMENT_RATIO = 6  # Number of augmented images per 1 original

os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_image(img):
    augmentations = []
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    for _ in range(AUGMENT_RATIO):
        aug = img.copy()

        # 1. ROTATION: Subtle ±15° to keep sign orientation valid
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use BORDER_REPLICATE to avoid black bars at corners
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. ZOOM: 0.8x to 1.2x to handle different hand distances
        scale = np.random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 3. LIGHTING: Adjust brightness and contrast for indoor diversity
        brightness = np.random.uniform(-50, 50)
        contrast = np.random.uniform(0.7, 1.3)
        aug = cv2.convertScaleAbs(aug, alpha=contrast, beta=brightness)

        # 4. TRANSLATION: Shift X and Y to simulate off-center hands
        shift_x = np.random.randint(-35, 35)
        shift_y = np.random.randint(-35, 35)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 5. BLUR: Randomly apply 3x3 blur to simulate low-res webcams
        if np.random.random() > 0.5:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)

        augmentations.append(aug)

    return augmentations

# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
print(f"🚀 Starting Production-Grade Augmentation...")
total_original = 0

# Get sorted list of class folders
labels = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])

for label in labels:
    in_folder = os.path.join(INPUT_DIR, label)
    out_folder = os.path.join(OUTPUT_DIR, label)
    
    os.makedirs(out_folder, exist_ok=True)
    
    images = [f for f in os.listdir(in_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_original += len(images)

    print(f"Processing Class [{label}]: {len(images)} images")

    for img_name in tqdm(images, desc=f"Augmenting {label}", leave=False):
        img_path = os.path.join(in_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # A. Save the original image first
        cv2.imwrite(os.path.join(out_folder, img_name), img)

        # B. Generate and save augmented versions
        base_name = os.path.splitext(img_name)[0]
        augs = augment_image(img)
        
        for i, aug_img in enumerate(augs):
            aug_filename = f"{base_name}_aug{i}.jpg"
            cv2.imwrite(os.path.join(out_folder, aug_filename), aug_img)

print("\n" + "="*40)
print(f"✅ SUCCESS: Augmentation complete!")
print(f"📦 Total Originals: {total_original}")
print(f"🔥 Total Dataset Size: ~{total_original * (AUGMENT_RATIO + 1)}")
print(f"📂 Location: '{OUTPUT_DIR}'")
print("="*40)
print("👉 NEXT STEP: Run your training script on 'dataset_static_aug'")
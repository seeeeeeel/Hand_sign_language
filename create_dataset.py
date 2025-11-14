import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow warnings

import pickle
import cv2
import mediapipe as mp

# ----------------------
# Setup MediaPipe Hands
# ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ----------------------
# Data directory
# ----------------------
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

data = []
labels = []

# Loop over all folders (A-Z)
for folder_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(class_dir):
        continue  # skip non-folder files

    print(f'Processing images for "{folder_name}"...')

    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                data.append(data_aux)
                labels.append(ord(folder_name.upper()) - 65)  # Convert A-Z → 0-25
        else:
            # Skip image if no hand detected
            continue

# ----------------------
# Save processed data
# ----------------------
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset creation finished for all folders!")

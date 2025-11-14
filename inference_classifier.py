import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# --------------------------
# Camera setup
# --------------------------
cap = cv2.VideoCapture(0)

# --------------------------
# Load trained model
# --------------------------
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# --------------------------
# Mediapipe hands
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --------------------------
# Labels 0-25 → A-Z
# --------------------------
labels_dict = {i: chr(65+i) for i in range(26)}

# --------------------------
# History for smoothing
# --------------------------
hand_histories = [
    deque(maxlen=5),   # Left
    deque(maxlen=5)    # Right
]

prev_time = 0

# --------------------------
# Main loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            # Left or Right
            label = handedness.classification[0].label
            hand_index = 0 if label == 'Left' else 1

            # Landmark coordinates
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            # Skip tiny detections
            if (max(x_) - min(x_)) < 0.05 or (max(y_) - min(y_)) < 0.05:
                continue

            # Prepare features
            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Predict
            pred = model.predict([np.array(data_aux)])
            letter = labels_dict[int(pred[0])]

            # Smooth
            hand_histories[hand_index].append(letter)
            predicted_char = max(
                set(hand_histories[hand_index]),
                key=hand_histories[hand_index].count
            )

            # Bounding box
            x1 = max(int(min(x_) * W) - 10, 0)
            y1 = max(int(min(y_) * H) - 10, 0)
            x2 = min(int(max(x_) * W) + 10, W)
            y2 = min(int(max(y_) * H) + 10, H)

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, f"{label}: {predicted_char}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 0), 3)

            # Landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

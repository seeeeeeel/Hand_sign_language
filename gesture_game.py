import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import random
from collections import deque
import tkinter as tk
from tkinter import messagebox

# --------------------------
# Load Trained Model
# --------------------------
MODEL_PATH = './model.p'
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found! Run train_classifier.py first.")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# --------------------------
# Mediapipe Setup
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
# Labels A-Z
# --------------------------
labels_dict = {i: chr(65 + i) for i in range(26)}  # 0->'A', 1->'B', ..., 25->'Z'

# --------------------------
# Camera Setup
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No working camera found.")
    sys.exit(1)

# --------------------------
# Game Settings
# --------------------------
score = 0
round_time = 30  # start 30 seconds timer
font = cv2.FONT_HERSHEY_SIMPLEX
target_letter = random.choice(list(labels_dict.values()))
start_time = time.time()
round_active = True

# Hand history smoothing
history_length = 5
hand_histories = [deque(maxlen=history_length), deque(maxlen=history_length)]  # Left, Right

print("🎮 Hand Sign Game Started! Press 'q' to quit.")

# --------------------------
# Main Game Loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(0, int(round_time - elapsed_time))

    # --------------------------
    # Detect hands and predict
    # --------------------------
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # Left or Right
            hand_index = 0 if label == 'Left' else 1

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            box_width = max(x_) - min(x_)
            box_height = max(y_) - min(y_)
            if box_width < 0.05 or box_height < 0.05:
                continue  # ignore small/false hands

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Predict
            prediction = model.predict([np.asarray(data_aux)])
            hand_histories[hand_index].append(prediction[0])
            predicted_number = max(set(hand_histories[hand_index]), key=hand_histories[hand_index].count)
            predicted_letter = labels_dict[int(predicted_number)]

            # Draw rectangle with predicted letter
            x1 = max(int(min(x_) * W) - 10, 0)
            y1 = max(int(min(y_) * H) - 10, 0)
            x2 = min(int(max(x_) * W) + 10, W)
            y2 = min(int(max(y_) * H) + 10, H)

            rect_color = (0, 255, 0) if predicted_letter == target_letter else (0, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 3)
            cv2.putText(frame, predicted_letter, (x1, y1 - 10),
                        font, 1.5, rect_color, 3, cv2.LINE_AA)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # ✅ If correct, add score + 1 sec to timer, auto-next
            if predicted_letter == target_letter and round_active:
                score += 1
                round_time += 1  # bonus 1 sec
                round_active = False
                cv2.putText(frame, "✅ CORRECT!", (150, 400), font, 2, (0, 255, 0), 4)
                cv2.imshow("✋ Gesture Game", frame)
                cv2.waitKey(800)
                target_letter = random.choice(list(labels_dict.values()))
                round_active = True

    # --------------------------
    # Timer check
    # --------------------------
    if remaining_time <= 0:
        break

    # Display target, timer, score
    cv2.putText(frame, f"Target: {target_letter}", (30, 60), font, 1.5, (0, 255, 255), 3)
    cv2.putText(frame, f"Time: {remaining_time}s", (30, 110), font, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Score: {score}", (30, 160), font, 1.2, (0, 255, 0), 2)

    cv2.imshow("✋ Gesture Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------
# Game Over
# --------------------------
cap.release()
cv2.destroyAllWindows()

# --------------------------
# Show final score popup
# --------------------------
root = tk.Tk()
root.withdraw()  # hide main window
messagebox.showinfo("Game Over", f"🏁 Time's up! Final Score: {score}")
root.destroy()

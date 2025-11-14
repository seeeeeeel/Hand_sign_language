import os
import cv2
from tkinter import simpledialog, Tk

# Initialize Tkinter (para sa input popup)
root = Tk()
root.withdraw()  # hide the main window

# Base data directory
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of images per class
dataset_size = 100

# Start video capture
cap = cv2.VideoCapture(0)

try:
    while True:
        # Popup: ask user for folder name
        folder_name = simpledialog.askstring("Input", "Enter the name for this class/folder:")
        if folder_name is None:
            print("No folder name provided. Exiting...")
            break

        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f'Collecting data for class "{folder_name}"...')

        # Ready screen before starting capture
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, 'Ready? Press "Q" to start :)', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            if key == ord('q') or key == ord('Q'):
                break

        # Capture images
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            img_path = os.path.join(folder_path, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            counter += 1

        print(f'Done collecting {dataset_size} images for "{folder_name}"!')

        # Ask if user wants to collect another class
        collect_another = simpledialog.askstring("Input", "Collect another class? (y/n)")
        if collect_another is None or collect_another.lower() != 'y':
            print("Exiting collection.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

from PIL import Image
import cv2 as cv
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

# Load Haar cascade correctly
cascade_path = os.path.join(BASE_DIR, 'data', 'haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier(cascade_path)

# Check if cascade loaded
if face_cascade.empty():
    print("Error loading cascade classifier")
    exit()

recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(("png", "jpg", "jfif")):  # Removed "heic"
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id = label_ids[label]

            # Open image safely
            try:
                pil_image = Image.open(path).convert("L")  # Convert to grayscale
            except Exception as e:
                print(f"Error opening {file}: {e}")
                continue

            image_array = np.array(pil_image, "uint8")

            # Detect faces
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                print(f"No face detected in {file}")
                continue

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)

# Save label mappings
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# Ensure training data is not empty before training
if len(x_train) > 0:
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('trainer.yml')
    print("Training complete. Model saved as trainer.yml")
else:
    print("No valid training data found. Training aborted.")

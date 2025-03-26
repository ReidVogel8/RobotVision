import os
import cv2 as cv
import numpy as np
import pickle
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load Haar Cascade for face detection
face_cascade = cv.CascadeClassifier(CASCADE_PATH)

# Create LBPH Face Recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()

# Training data storage
current_id = 0
label_ids = {}
x_train = []
y_labels = []

# Loop through images folder
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith(("png", "jpg", "jpeg", "HEIC")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            # Assign unique ID to each label
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            # Open image and convert to grayscale
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")  # Ensure correct dtype

            # Detect face
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]  # Extract face region
                x_train.append(roi)
                y_labels.append(id_)

# Save label IDs for later recognition
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# Train recognizer and save model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

print("Training Complete!")

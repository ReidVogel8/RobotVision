import os
import cv2
import numpy as np
import pickle
from PIL import Image

# Initialize face recognizer and detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

data_path = "Images/"
labels = {}
current_id = 0
x_train, y_labels = [], []

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)
    labels[person] = current_id
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = Image.open(img_path).convert("L")
        img_array = np.array(image, "uint8")

        faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            x_train.append(img_array[y:y+h, x:x+w])
            y_labels.append(current_id)

    current_id += 1

# Save label mappings
with open("labels.pickle", "wb") as f:
    pickle.dump(labels, f)

# Train recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

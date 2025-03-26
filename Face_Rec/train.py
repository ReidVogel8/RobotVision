from PIL import Image
import cv2 as cv
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
image_dir = os.path.join(BASE_DIR, "Images/Hunter")

face_cascade = cv.CascadeClassifier(BASE_DIR + 'haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root).replace(" ", "_").lower()
        print(os.path.basename(root))
        print(label, path)
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id = label_ids[label]
        print(label_ids)
        pil_image = Image.open(path).convert("L")
        image_array = np.array(pil_image, "unit8")
        faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
        for(x,y,w,h) in faces:
            roi = image_array[y:y+h, x:x+w]
            x_train.append(roi)
            y_labels.append(id)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save('traner.yml')

import cv2
import pickle
import threading
import time
import tkinter as tk

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
reverse_labels = {v: k for k, v in labels.items()}  # Convert IDs to names
#camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


# Create Tkinter window
root = tk.Tk()
root.title("Face Recognition")

label_var = tk.StringVar()
label_var.set("Waiting for face...")

label = tk.Label(root, textvariable=label_var, font=("Arial", 20))
label.pack()

def update_gui(name):
    label_var.set(name)
    root.update_idletasks()

def recognize_faces():
    global face_detected, last_detected, last_update_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 0:
            if time.time() - last_update_time > 3:
                update_gui("👀 Looking for faces...")
            continue

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)

            if confidence < 60:  # Threshold for recognition
                name = reverse_labels[id_]
                if name != last_detected:
                    last_detected = name
                    update_gui(name)
                    move_robot("forward")
            else:
                if "Stranger Danger" != last_detected:
                    last_detected = "Stranger Danger"
                    update_gui("Stranger Danger")
                    move_robot("backward")

        last_update_time = time.time()

def move_robot(direction):
    if direction == "forward":
        print("Moving forward 2 feet...")
        # Send command to robot motors
    elif direction == "backward":
        print("Backing up 3 feet...")


thread = threading.Thread(target=recognize_faces, daemon=True)
thread.start()

root.mainloop()
cap.release()
cv2.destroyAllWindows()

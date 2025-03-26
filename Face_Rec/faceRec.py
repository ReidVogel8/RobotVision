import cv2 as cv
import numpy as np
import pickle
import pyrealsense2 as rs
import tkinter as tk
from tkinter import Label
import time
from maestro import Controller  # Robot movement library

# Load trained recognizer and labels
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}  # Reverse mapping ID -> Name

# Load Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Robot movement control class
MIDDLE = 6000
HEAD_UP_DOWN_PORT = 4
HEAD_LEFT_RIGHT_PORT = 2
LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1


class RobotControl:
    _instance = None

    @staticmethod
    def getInst():
        if RobotControl._instance is None:
            RobotControl._instance = RobotControl()
        return RobotControl._instance

    def __init__(self):
        self.m = Controller()

    def move_forward(self):
        print("Moving Forward 2 Feet")
        self.m.setTarget(LEFT_WHEEL_PORT, 7000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(2) 
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def move_backward(self):
        print("Stranger Danger! Moving Backward 3 Feet")
        self.m.setTarget(LEFT_WHEEL_PORT, 5000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(3) 
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)


robot = RobotControl().getInst()

# Tkinter GUI setup
root = tk.Tk()
root.title("Face Recognition")
root.geometry("800x800")
label_text = tk.StringVar()
label_display = Label(root, textvariable=label_text, font=("Arial", 16))
label_display.pack()


def recognize_faces():
    while True:
        # Get frame from RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]

            # Recognize face
            id_, confidence = recognizer.predict(roi_gray)
            if confidence < 50:  # Confidence threshold (lower is better)
                name = labels[id_]
                label_text.set(f"Hello, {name}!")
                #robot.move_forward()
            else:
                label_text.set("Stranger Danger!")
                #robot.move_backward()

            # Draw rectangle around detected face
            color = (0, 255, 0) if confidence < 50 else (0, 0, 255)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Show the live video feed
        cv.imshow("Face Recognition", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv.destroyAllWindows()


# Run face recognition in a separate thread
import threading

threading.Thread(target=recognize_faces, daemon=True).start()

# Run Tkinter GUI
root.mainloop()

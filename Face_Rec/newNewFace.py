import cv2 as cv
import numpy as np
import pickle
import pyrealsense2 as rs
import tkinter as tk
from tkinter import Label
import time
import threading
from collections import deque
from maestro import Controller 

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

# Robot movement control
MIDDLE = 6000
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
        print("Moving Forward")
        self.m.setTarget(LEFT_WHEEL_PORT, 7000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def move_backward(self):
        print("Moving Backward")
        self.m.setTarget(LEFT_WHEEL_PORT, 5000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

robot = RobotControl().getInst()

# Tkinter GUI setup
root = tk.Tk()
root.title("Face Recognition")
root.geometry("400x200")
label_text = tk.StringVar()
label_display = Label(root, textvariable=label_text, font=("Arial", 20))
label_display.pack(pady=20)

canvas = tk.Canvas(root, width=200, height=100, bg="white")
canvas.pack()

def draw_eyes():
    canvas.delete("all")
    canvas.create_oval(50, 20, 90, 60, fill="black")
    canvas.create_oval(110, 20, 150, 60, fill="black")

last_detected = None
detection_start_time = None
DETECTION_THRESHOLD = 2  # Time before movement triggers
recognition_queue = deque(maxlen=10)  # Store recent recognitions for smoothing

def recognize_faces():
    global last_detected, detection_start_time

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            root.after(0, lambda: label_text.set("Looking..."))
            root.after(0, draw_eyes)
            recognition_queue.clear()
            last_detected = None
            detection_start_time = None
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv.resize(roi_gray, (100, 100))  # Resize for consistency

                id_, confidence = recognizer.predict(roi_gray)
                print(f"Detected ID: {id_}, Confidence: {confidence}")

                if confidence < 60:  # Adaptive confidence threshold
                    name = labels[id_]
                    recognition_queue.append(name)  # Store in queue
                else:
                    recognition_queue.append("stranger")

                # Ensure consistency: Choose the most common recognition over the last few frames
                if len(recognition_queue) == recognition_queue.maxlen:
                    most_common = max(set(recognition_queue), key=recognition_queue.count)
                    
                    if most_common != last_detected:
                        last_detected = most_common
                        detection_start_time = time.time()

                    if time.time() - detection_start_time >= DETECTION_THRESHOLD:
                        if most_common != "stranger":
                            print(f"Recognized {most_common} consistently. Moving forward.")
                            robot.move_forward()
                        else:
                            print("Stranger detected consistently. Moving backward.")
                            robot.move_backward()
                        last_detected = None  # Prevent multiple triggers

                    root.after(0, lambda: label_text.set(f"Hello, {most_common}!" if most_common != "stranger" else "Stranger Danger!"))

                color = (0, 255, 0) if confidence < 50 else (0, 0, 255)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv.destroyAllWindows()

# Start face recognition in a separate thread
threading.Thread(target=recognize_faces, daemon=True).start()

# Start Tkinter GUI
root.mainloop()

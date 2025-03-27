import cv2 as cv
import numpy as np
import pickle
import pyrealsense2 as rs
import tkinter as tk
from tkinter import Label
import time
import threading
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
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 5000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)


    def move_backward(self):
        print("Stranger Danger! Moving Backward 3 Feet")
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 7000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)


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
    """Draws animated eyes when no face is detected."""
    canvas.delete("all")
    canvas.create_oval(50, 20, 90, 60, fill="black")  # Left eye
    canvas.create_oval(110, 20, 150, 60, fill="black")  # Right eye


last_detected_faces = {}
DETECTION_THRESHOLD = 1  # Time in seconds before triggering movement


def recognize_faces():
    global last_detected_faces

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Histogram equalization to improve contrast
        gray = cv.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        recognized_names = []

        if len(faces) == 0:
            root.after(0, lambda: label_text.set("Looking..."))
            root.after(0, draw_eyes)
            last_detected_faces.clear()
        else:
            current_time = time.time()
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                id_, confidence = recognizer.predict(roi_gray)

                print(f"Detected ID {id_} with confidence {confidence}")

                if confidence < 70:  # Adjusted threshold to be more lenient
                    name = labels.get(id_, "Unknown")
                    recognized_names.append(name)

                    if name in last_detected_faces:
                        if current_time - last_detected_faces[name] >= DETECTION_THRESHOLD:
                            print(f"Recognized {name} for {DETECTION_THRESHOLD} seconds. Moving forward.")
                            robot.move_forward()
                            last_detected_faces.pop(name)  # Prevent multiple triggers
                    else:
                        last_detected_faces[name] = current_time

                    color = (0, 255, 0)
                else:  # Unknown face
                    recognized_names.append("Stranger")

                    if "Stranger" in last_detected_faces:
                        if current_time - last_detected_faces["Stranger"] >= DETECTION_THRESHOLD:
                            print("Stranger detected for 2 seconds. Moving backward.")
                            robot.move_backward()
                            last_detected_faces.pop("Stranger")  # Prevent multiple triggers
                    else:
                        last_detected_faces["Stranger"] = current_time

                    color = (0, 0, 255)

                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            root.after(0, lambda: label_text.set(f"Hello, {', '.join(recognized_names)}!"))

        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv.destroyAllWindows()


# Start face recognition in a separate thread
threading.Thread(target=recognize_faces, daemon=True).start()

# Start Tkinter GUI
root.mainloop()

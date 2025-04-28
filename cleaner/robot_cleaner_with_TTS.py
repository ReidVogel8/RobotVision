import cv2
import pickle
import numpy as np
import pyrealsense2 as rs
import time
import edge_tts
import asyncio
import threading
import playsound  # or your preferred playback method
from maestro import Controller  # Your Maestro controller class

class Talk:
    def __init__(self, voice="en-US-ChristopherNeural"):
        self.voice = voice

    async def _speak_async(self, text: str):
        temp_file = "tts.wav"
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(temp_file)
        playsound.playsound(temp_file)

    def say(self, text: str):
        # Fire-and-forget TTS in a background thread
        threading.Thread(target=self._run_async, args=(text,), daemon=True).start()

    def _run_async(self, text: str):
        asyncio.run(self._speak_async(text))

# Initialize Text-to-Speech
speaker = Talk(voice="en-US-GuyNeural")  # change voice as desired

# Load Calibration
with open("calibration.pkl", "rb") as f:
    calib = pickle.load(f)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]

# Load Trained Objects
with open("trainedObjects.pkl", "rb") as f:
    trained_objects = pickle.load(f)

# Rebuild Keypoints
for obj in trained_objects:
    obj['keypoints'] = [
        cv2.KeyPoint(pt[0][0], pt[0][1], pt[1], pt[2], pt[3], int(pt[4]), int(pt[5]))
        for pt in obj['keypoints']
    ]

# ORB Matcher
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ArUco Setup
dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict_4x4, params)

# RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Robot Setup
robot = Controller()

# Motor Ports
LEFT_WHEEL = 0
RIGHT_WHEEL = 1
LEFT_ELBOW = 7
LEFT_SHOULDER = 5

# Movement Functions
def raise_arm():
    robot.setTarget(LEFT_SHOULDER, 5600)
    time.sleep(0.5)
    robot.setTarget(LEFT_ELBOW, 8700)
    time.sleep(1)

def lower_arm():
    robot.setTarget(LEFT_SHOULDER, 6500)
    time.sleep(0.3)
    robot.setTarget(LEFT_ELBOW, 5600)
    time.sleep(1)

def move_forward(duration: float):
    robot.setTarget(LEFT_WHEEL, 6000)
    time.sleep(0.3)
    robot.setTarget(LEFT_WHEEL, 8000)
    time.sleep(duration)
    robot.setTarget(LEFT_WHEEL, 6000)

def move_backward(duration: float):
    robot.setTarget(LEFT_WHEEL, 6000)
    time.sleep(0.3)
    robot.setTarget(LEFT_WHEEL, 5000)
    time.sleep(duration)
    robot.setTarget(LEFT_WHEEL, 6000)

def small_rotate_left():
    robot.setTarget(RIGHT_WHEEL, 5800)
    time.sleep(0.5)
    robot.setTarget(RIGHT_WHEEL, 6000)
    time.sleep(0.3)
    robot.setTarget(RIGHT_WHEEL, 5800)

# Main Routine
def main():
    try:
        # Greet and notify
        speaker.say("Robot ready. Scanning for face...")
        print("Robot ready. Scanning for face...")

        # Face detection loop
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if any(w > 100 and h > 100 for (_, _, w, h) in faces):
                speaker.say("Ugh. What now?")
                print("Ugh. What now?")
                break

        # Ask for object
        speaker.say("What am I supposed to clean up this time?")
        print("What am I supposed to clean up this time?")
        time.sleep(2)

        # Capture and identify object
        frames = pipeline.wait_for_frames()
        frame = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        best_match = None
        best_matches = []
        for obj in trained_objects:
            matches = bf.match(des, obj['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > len(best_matches):
                best_matches, best_match = matches, obj

        if best_match:
            name, obj_id = best_match['name'], best_match['id']
            speaker.say(f"Fine. That's the {name}. Guiding to box {obj_id}.")
            print(f"Fine. That’s the {name}. Guess I’ll put it in box {obj_id}.")
            speaker.say("Initiating ring ritual. Raising arm.")
            raise_arm()
            time.sleep(1)
        else:
            speaker.say("I have no idea what that is. Going to sleep.")
            print("I have no idea what that is. I'm going back to sleep.")
            return

        # Scan for ArUco marker
        speaker.say("Scanning for marker while rotating...")
        print("Scanning for marker while rotating...")
        found_marker = False
        while not found_marker:
            frames = pipeline.wait_for_frames()
            frame = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid == obj_id:
                        speaker.say(f"Found marker {obj_id}.")
                        print(f"Found Marker {obj_id}")
                        found_marker = True
                        break
            if not found_marker:
                speaker.say("Marker not found, rotating slightly.")
                print("Marker not found, rotating slightly...")
                small_rotate_left()

        # Approach marker
        speaker.say("Approaching the marker.")
        print("Approaching the marker...")
        close_enough = False
        while not close_enough:
            frames = pipeline.wait_for_frames()
            frame = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid == obj_id:
                        _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                        z = tvec[0][0][2]
                        speaker.say(f"Distance to marker is {z:.2f} meters.")
                        print(f"Distance to marker (Z): {z:.3f} meters")
                        if z <= 0.1:
                            speaker.say("Reached close enough to marker.")
                            print("Reached close enough to marker.")
                            close_enough = True
                            break
            if not close_enough:
                move_backward(0.2)
                time.sleep(0.1)

        # Final actions
        speaker.say("Lowering arm and dropping ring.")
        print("Lowering arm and dropping ring.")
        lower_arm()
        time.sleep(1)
        move_forward(1)
        speaker.say("Task complete.")
        print("Task complete.")

    except KeyboardInterrupt:
        speaker.say("Interrupted by user.")
        print("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

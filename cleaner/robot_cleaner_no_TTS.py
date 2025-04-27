import cv2
import pickle
import numpy as np
import pyrealsense2 as rs
import time
from maestro import Controller  # Your Maestro controller class

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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

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
    robot.setTarget(LEFT_SHOULDER, 8000)
    time.sleep(0.3)
    robot.setTarget(LEFT_ELBOW, 5600)
    time.sleep(1)

def move_forward(duration):
    robot.setTarget(LEFT_WHEEL, 6000)
    time.sleep(0.3)
    robot.setTarget(LEFT_WHEEL, 8000)
    time.sleep(duration)
    robot.setTarget(LEFT_WHEEL, 6000)

def rotate_left():
    robot.setTarget(RIGHT_WHEEL, 6000)
    time.sleep(0.3)
    robot.setTarget(RIGHT_WHEEL, 6500)
    time.sleep(0.5)
    robot.setTarget(RIGHT_WHEEL, 6000)

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main Routine
try:
    print("Robot ready. Scanning for face...")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_detected = any(w > 100 and h > 100 for (x, y, w, h) in faces)

        if face_detected:
            print("Ugh. What now?")
            break

    # Ask for Object
    print("What am I supposed to clean up this time?")
    time.sleep(2)

    # Capture object input
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ORB Detect on Current Frame
    kp, des = orb.detectAndCompute(gray, None)

    best_match = None
    best_matches = []

    for obj in trained_objects:
        matches = bf.match(des, obj['descriptors'])
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > len(best_matches):
            best_matches = matches
            best_match = obj

    if best_match:
        name = best_match['name']
        obj_id = best_match['id']
        print(f"Fine. That’s the {name}. Guess I’ll put it in box {obj_id}.")
        print("Initiating ring ritual. Raising arm.")
        raise_arm()
    else:
        print("I have no idea what that is. I'm going back to sleep.")
        exit()

    # Rotate to find marker
    print(f"Rotating to find Marker ID {obj_id}...")

    found = False
    spin_attempts = 0
    max_attempts = 10  # safety cap

    while not found and spin_attempts < max_attempts:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, marker_id in enumerate(ids):
                if int(marker_id) == obj_id:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                    x = tvec[0][0][0]
                    y = tvec[0][0][1]
                    z = tvec[0][0][2]
                    print(f"Found Marker {obj_id} - X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m")
                    found = True
                    break

        if not found:
            rotate_left()
            spin_attempts += 1

    if not found:
        print(f"Could not find marker ID {obj_id} after spinning. Giving up.")
        exit()

    # Move forward after finding marker
    print("Approaching the marker...")

    # Define stopping distance threshold
    target_distance = 0.5  # meters
    close_enough = False
    approach_attempts = 0
    max_approach_attempts = 10

    while not close_enough and approach_attempts < max_approach_attempts:
        # Get latest frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco marker
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids):
                if int(marker_id) == obj_id:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                    z = tvec[0][0][2]
                    print(f"📏 Distance to marker (Z): {z:.2f} meters")

                    if z <= target_distance:
                        print("Close enough to the box.")
                        close_enough = True
                        break

        if not close_enough:
            print("🚶 Moving a bit closer...")
            move_forward(0.3)
            time.sleep(0.2)  # small pause
            approach_attempts += 1

    if not close_enough:
        print("Stopped approaching: Max attempts reached.")

    # Lower the arm after moving
    print("Lowering arm and dropping ring.")
    lower_arm()

    # Return to Center (marker ID 0)
    print("Returning to the start. Barely.")
    returning = False
    while not returning:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids):
                if int(marker_id) == 0:
                    returning = True
                    move_forward()
                    break

    print("Cleaning complete. Barely.")

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

import cv2
import numpy as np
import pyrealsense2 as rs
import pickle

# Load camera calibration
with open("calibration.pkl", "rb") as f:
    calib = pickle.load(f)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]

# Setup RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

try:
    print("Looking for ArUco markers...")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            for i, marker_id in enumerate(ids):
                z = tvecs[i][0][2]
                print(f"Marker {marker_id[0]} is {z:.2f} meters away.")

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import pickle
import time
from maestro import Controller

# Load calibration data from the pickle file
with open("calibration.pkl", "rb") as f:
    calibration_data = pickle.load(f)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

# ArUco setup
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Robot setup
MIDDLE = 6000
HEAD_UP_DOWN_PORT = 4
HEAD_LEFT_RIGHT_PORT = 2
LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1

class RobotControl:
    def __init__(self):
        self.m = Controller()

    def start(self):
        self.m.setTarget(LEFT_WHEEL_PORT, MIDDLE)
        self.m.setTarget(RIGHT_WHEEL_PORT, MIDDLE)
        
    def move_forward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 7000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.stop()

    def move_backward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 5000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(1)
        self.stop()

    def turn_left(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 5000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.stop()

    def turn_right(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 7000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(1)
        self.stop()

    def pan_left(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 5000)

    def pan_right(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 7000)

    def stop(self):
        self.m.setTarget(LEFT_WHEEL_PORT, MIDDLE)
        self.m.setTarget(RIGHT_WHEEL_PORT, MIDDLE)

robot = RobotControl()
visited_ids = set()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                id_num = int(ids[i][0])
                if id_num in visited_ids:
                    continue

                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                x, z = tvec[0][0][0], tvec[0][0][2]
                cx = corners[i][0][:, 0].mean()
                frame_center_x = frame.shape[1] // 2

                print(f"Detected Marker ID: {id_num}, X: {x:.2f}m, Z: {z:.2f}m")

                # Pan to keep centered
                if cx < frame_center_x - 30:
                    RobotControl.pan_left()
                elif cx > frame_center_x + 30:
                    RobotControl.pan_right()

                # Navigation logic
                if id_num % 2 == 0:
                    print("Passing on right")
                    RobotControl.start(self)
                    RobotControl.turn_right(self)
                    RobotControl.move_forward(self)
                    RobotControl.turn_left(self)
                else:
                    print("Passing on left")
                    RobotControl.start(self)
                    RobotControl.turn_left(self)
                    RobotControl.move_forward(self)
                    RobotControl.turn_right(self)

                visited_ids.add(id_num)

                # Stop condition
                if len(visited_ids) >= 4:
                    print("Finished")
                    RobotControl.stop(self)
                    raise KeyboardInterrupt

        cv.imshow("ArUco Navigation", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    pipeline.stop()

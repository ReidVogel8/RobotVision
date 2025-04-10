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
HEAD_UP_DOWN_PORT = 2
HEAD_LEFT_RIGHT_PORT = 4
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

    def head_center(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 6000)
        self.m.setTarget(HEAD_UP_DOWN_PORT, 6000)

    def pan_left(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 7000)

    def pan_right(self):
        self.m.setTarget(HEAD_LEFT_RIGHT_PORT, 5000)

    def turn_left(self, duration):
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        time.sleep(0.5)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(duration)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def turn_right(self, duration):
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        time.sleep(0.5)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(duration)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def move_forward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(LEFT_WHEEL_PORT, 5000)
        time.sleep(1.3)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)

    def slight_left(self):
        self.m.setTarget(RIGHT_WHEEL_PORT, 6200)
        self.m.setTarget(LEFT_WHEEL_PORT, 5800)
        time.sleep(0.3)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)

    def slight_right(self):
        self.m.setTarget(RIGHT_WHEEL_PORT, 5800)
        self.m.setTarget(LEFT_WHEEL_PORT, 6200)
        time.sleep(0.3)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)

# Function to calculate camera position relative to the marker
def get_camera_position_from_marker(marker_world_pos, rvec, tvec):
    R_ct, _ = cv.Rodrigues(rvec)
    tvec = tvec.reshape((3, 1))

    R_tc = R_ct.T
    t_tc = -np.dot(R_tc, tvec)

    marker_x, marker_y = marker_world_pos
    marker_world = np.array([[marker_x], [marker_y], [0.0]])

    camera_world = marker_world + t_tc[0:3]
    return float(camera_world[0]), float(camera_world[1])

def adjust_course_based_on_marker(corners, frame_width):
    cx = corners[:, 0].mean()
    deviation = cx - frame_width / 2
    tolerance = 30
    if deviation > tolerance:
        print("Adjusting course slightly left")
        robot.slight_left()
        robot.pan_right()
    elif deviation < -tolerance:
        print("Adjusting course slightly right")
        robot.slight_right()
        robot.pan_left()
    else:
        print("Course centered")
        robot.head_center()

def navigate_around_marker(id_num):
    if id_num % 2 != 0:
        print("Navigating around left side (odd ID)")
        robot.turn_left(0.8)
        robot.pan_right()
        robot.move_forward()
        robot.turn_right(0.71)
        robot.move_forward()
        robot.turn_right(0.72)
        robot.move_forward()
        robot.turn_left(0.77)
        robot.pan_left()
    else:
        print("Navigating around right side (even ID)")
        robot.turn_right(0.8)
        robot.pan_left()
        robot.move_forward()
        robot.turn_left(0.89)
        robot.move_forward()
        robot.turn_left(0.86)
        robot.move_forward()
        robot.turn_right(0.77)
        robot.pan_right()

robot = RobotControl()
visited_ids = set()
count = 0
marker_last_seen_time = {}
MARKER_COOLDOWN_SECONDS = 1
position_x = 0.0
position_z = 0.0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        current_time = time.time()

        if ids is not None:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                id_num = int(ids[i][0])
                last_seen = marker_last_seen_time.get(id_num, 0)
                if current_time - last_seen < MARKER_COOLDOWN_SECONDS:
                    continue

                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                x, z = tvec[0][0][0], tvec[0][0][2]
                print(f"Detected Marker ID: {id_num}, Distance Z: {z:.2f}m")

                camera_x, camera_z = get_camera_position_from_marker((x, z), rvec[0], tvec[0])
                position_x = camera_x
                position_z = camera_z
                print(f"Robot Coordinates: ({position_x:.2f} ft, {position_z:.2f} ft)")

                adjust_course_based_on_marker(corners[i][0], frame.shape[1])

                if z > 0.4:
                    print("Marker is far, moving forward...")
                    robot.move_forward()
                else:
                    print("Marker is close enough, initiating navigation...")
                    navigate_around_marker(id_num)
                    count += 1
                    marker_last_seen_time[id_num] = current_time
                    robot.head_center()

                    if count >= 4:
                        print("Finished navigating all markers.")
                        raise KeyboardInterrupt

        cv.imshow("ArUco Navigation", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    pipeline.stop()
    cv.destroyAllWindows()

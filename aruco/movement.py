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
    _instance = None

    @staticmethod
    def getInst():
        if RobotControl._instance is None:
            RobotControl._instance = RobotControl()
        return RobotControl._instance

    def __init__(self):
        self.m = Controller()

    def start(self):
        print("start")

    def pan_left(self):
        self.m.setTarget(4, 6000)
        self.m.setTarget(4, 5500)
        time.sleep(.2)

    def pan_right(self):
        self.m.setTarget(4, 6000)
        self.m.setTarget(4, 6500)
        time.sleep(.2)

    def turn_left(self):
        print("left")
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(.3)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def turn_right(self):
        print("right")
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(.3)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)

    def move_forward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 5000)
        time.sleep(.5)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)

    def move_backward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 7000)
        time.sleep(.6)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)


# Function to calculate camera position relative to the marker
def get_camera_position_from_marker(marker_world_pos, rvec, tvec):
    R_ct, _ = cv.Rodrigues(rvec)
    tvec = tvec.reshape((3, 1))

    # Invert the transformation (Rotation and Translation)
    R_tc = R_ct.T
    t_tc = -np.dot(R_tc, tvec)

    marker_x, marker_y = marker_world_pos
    marker_world = np.array([[marker_x], [marker_y], [0.0]])

    # Camera position in world coordinates
    camera_world = marker_world + t_tc[0:3]
    return float(camera_world[0]), float(camera_world[1])


robot = RobotControl()
visited_ids = set()


# Function to check if the robot has passed the marker based on the camera's position
def passed_marker(camera_x):
    return camera_x > 0  # Assuming the robot moves forward after passing the marker


# Initialize state variables
searching_for_marker = True
first_marker_passed = False

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
                # Estimate pose of the marker
                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                # Get the position of the camera relative to the marker
                x, z = tvec[0][0][0], tvec[0][0][2]

                # Get camera position in world coordinates
                marker_world_pos = (x, z)
                camera_x, camera_z = get_camera_position_from_marker(marker_world_pos, rvec, tvec)

                print(f"Detected Marker ID: {id_num}, Camera X: {camera_x:.2f}m, Camera Z: {camera_z:.2f}m")

                # Check if the marker is to the left or right
                if id_num % 2 == 0:
                    if x < 0:  # Marker to the left
                        print("Marker on the Left")
                        robot.turn_left()
                        robot.move_forward()
                        robot.turn_right()
                    else:
                        robot.move_forward()
                else:
                    if x > 0:  # Marker to the right
                        print("Marker on the Right")
                        robot.turn_right()
                        robot.move_forward()
                        robot.turn_left()
                    else:
                        robot.move_forward()

                # Stop looking for the first marker once passed
                if not first_marker_passed and passed_marker(camera_x):
                    print("First marker passed, now searching for the next marker")
                    first_marker_passed = True

                # If the first marker is passed, start searching for the next one
                if first_marker_passed:
                    # Pan head to search for the next marker
                    if searching_for_marker:
                        if camera_x < 0:  # Left of the camera
                            robot.pan_left()
                        elif camera_x > 0:  # Right of the camera
                            robot.pan_right()
                    else:
                        # We found the next marker, stop searching
                        searching_for_marker = False

                visited_ids.add(id_num)

        cv.imshow("ArUco Navigation", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    pipeline.stop()
    cv.destroyAllWindows()

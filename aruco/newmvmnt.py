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
        print("turn head left")
        
    def pan_right(self):
        print("turn head right")
        
    def turn_left(self):
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(1)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        time.sleep(5)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(LEFT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        
    def turn_right(self):
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(LEFT_WHEEL_PORT, 7000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        self.m.setTarget(RIGHT_WHEEL_PORT, 5000)
        time.sleep(1)
        self.m.setTarget(RIGHT_WHEEL_PORT, 6000)
        
    def move_forward(self):
        print("Moving Forward 2 Feet")
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 5000)
        time.sleep(1)
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)

    def move_backward(self):
        self.m.setTarget(LEFT_WHEEL_PORT, 6000)
        self.m.setTarget(0, 7000)
        time.sleep(1)
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

                # Estimate pose of the marker
                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.055, camera_matrix, dist_coeffs)
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                # Get the position of the camera relative to the marker
                x, z = tvec[0][0][0], tvec[0][0][2]

                # Get camera position in world coordinates
                marker_world_pos = (x, z)
                camera_x, camera_z = get_camera_position_from_marker(marker_world_pos, rvec, tvec)

                print(f"Detected Marker ID: {id_num}, Camera X: {camera_x:.2f}m, Camera Z: {camera_z:.2f}m")

                # Determine robot movement based on camera position relative to marker
                frame_center_x = frame.shape[1] // 2
                cx = corners[i][0][:, 0].mean()

                # Pan to keep centered
                if cx < frame_center_x - 30:
                    robot.pan_left()
                elif cx > frame_center_x + 30:
                    robot.pan_right()

                # Navigation logic based on camera position
                if camera_x < 0:  # Assuming robot is left of the marker
                    print("Turning Left")
                    robot.turn_left()
                    robot.move_forward()
                    robot.turn_right()
                elif camera_x > 0:  # Assuming robot is right of the marker
                    print("Turning Right")
                    robot.turn_right()
                    robot.move_forward()
                    robot.turn_left()
                else:  # Robot is centered in front of marker
                    print("Going Forward")
                    robot.move_forward()

                visited_ids.add(id_num)

                # Stop condition
                if len(visited_ids) >= 4:
                    print("Finished")
                    robot.stop()
                    raise KeyboardInterrupt

        cv.imshow("ArUco Navigation", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    pipeline.stop()
    cv.destroyAllWindows()

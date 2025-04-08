import cv2
import numpy as np
import pyrealsense2 as rs
import time
import math
import maestro  # Your external servo/movement module

# === Load Camera Calibration ===
def load_calibration(file_path='calibration.npz'):
    data = np.load(file_path)
    return data['camera_matrix'], data['dist_coeffs']

# === Movement Functions ===
def move_forward():
    print("[Move] Forward")
    self.m.forward()  # Assuming your maestro module has this

def turn_left():
    print("[Move] Turn Left")
    self.m.left()

def turn_right():
    print("[Move] Turn Right")
    self.m.right()

def stop_robot():
    print("[Move] Stop")
    self.m.stop()

def center_pan_tilt():
    self.m.setTarget(0, 6000)  # Pan center
    self.m.setTarget(1, 6000)  # Tilt center

# === Estimate robot's X, Y from marker pose ===
def get_xy_from_pose(tvec):
    return tvec[0][0], tvec[0][2]  # X (sideways), Z (forward)

# === Main Vision + Navigation ===
def main():
    # Load calibration
    camera_matrix, dist_coeffs = load_calibration()

    # Set up RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Set up ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Marker tracking
    marker_ids_seen = set()
    final_marker_id = 11

    try:
        center_pan_tilt()

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id not in [8, 9, 10, 11]:
                        continue

                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], 0.055, camera_matrix, dist_coeffs
                    )
                    cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                    x, z = get_xy_from_pose(tvec)
                    print(f"Marker {marker_id} | X: {x:.2f} ft, Z: {z:.2f} ft")

                    # Center pan servo
                    center_x = np.mean(corners[i][0][:, 0])
                    error = center_x - frame.shape[1] / 2
                    pan_value = int(6000 + error * 4)  # Adjust multiplier if needed
                    maestro.setTarget(0, pan_value)

                    # Decide side to pass
                    direction = "left" if marker_id % 2 == 1 else "right"
                    print(f"→ Passing marker {marker_id} on the {direction}")

                    if direction == "left":
                        turn_left()
                    else:
                        turn_right()

                    move_forward()
                    time.sleep(1.5)
                    stop_robot()

                    marker_ids_seen.add(marker_id)

                    if final_marker_id in marker_ids_seen:
                        stop_robot()
                        print("Finished ✅")
                        return

            cv2.imshow("ArUco Navigation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        center_pan_tilt()
        stop_robot()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import pyrealsense2 as rs

# === Checkerboard Settings ===
CHECKERBOARD = (8, 6)           # Inner corners (columns, rows)
SQUARE_SIZE = 0.055             # Size of a square in meters (55mm)

# === Prepare object points ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# === Start RealSense Camera ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Press 'c' to capture a frame. Press 'q' to quit and calibrate.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = frame.copy()
        if ret:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)

        cv2.imshow('Calibration - Press "c" to capture, "q" to finish', display)
        key = cv2.waitKey(1)

        if key == ord('c') and ret:
            print("Captured frame.")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# === Calibrate ===
if len(objpoints) > 5:
    print("Calibrating...")
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savez("calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("Calibration saved to calibration.npz")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Not enough frames captured for calibration.")

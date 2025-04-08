import cv2
import numpy as np
import glob
import pickle

# Define checkerboard dimensions
CHECKERBOARD = (8, 6)
square_size = 0.055  # in meters

# Termination criteria for corner sub-pix refining
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), ... (7,5,0) scaled by square size
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture(0)

print("Press 's' to save a frame for calibration, 'q' to quit and calibrate")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)

    cv2.imshow('Calibration - Press s to save frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and found:
        print("Saved frame for calibration")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Calibrating camera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration results
with open("calibration_data.pickle", "wb") as f:
    pickle.dump({"camera_matrix": mtx, "dist_coeff": dist}, f)

print("Camera matrix:")
print(mtx)
print("Distortion coefficients:")
print(dist)
print("Calibration complete. Data saved to 'calibration_data.pickle'")

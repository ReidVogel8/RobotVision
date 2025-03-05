import cv2
import numpy as np

# Global variables for HSV thresholds
minH, minS, minV = 0, 0, 0
maxH, maxS, maxV = 179, 255, 255

clicked_hsv = (0, 0, 0)

def on_trackbar(val):
    pass

# Mouse callback function to get HSV values on click
def get_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = param
        clicked_hsv = frame_hsv[y, x]
        print(f"Clicked HSV: {clicked_hsv}")

cap = cv2.VideoCapture(0)

cv2.namedWindow("HSV Tracker")
cv2.createTrackbar("Min H", "HSV Tracker", 0, 179, on_trackbar)
cv2.createTrackbar("Max H", "HSV Tracker", 179, 179, on_trackbar)
cv2.createTrackbar("Min S", "HSV Tracker", 0, 255, on_trackbar)
cv2.createTrackbar("Max S", "HSV Tracker", 255, 255, on_trackbar)
cv2.createTrackbar("Min V", "HSV Tracker", 0, 255, on_trackbar)
cv2.createTrackbar("Max V", "HSV Tracker", 255, 255, on_trackbar)

cv2.setMouseCallback("HSV Tracker", get_hsv)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.setMouseCallback("HSV Tracker", get_hsv, hsv_frame)

    # Get values from trackbars
    minH = cv2.getTrackbarPos("Min H", "HSV Tracker")
    maxH = cv2.getTrackbarPos("Max H", "HSV Tracker")
    minS = cv2.getTrackbarPos("Min S", "HSV Tracker")
    maxS = cv2.getTrackbarPos("Max S", "HSV Tracker")
    minV = cv2.getTrackbarPos("Min V", "HSV Tracker")
    maxV = cv2.getTrackbarPos("Max V", "HSV Tracker")

    lower_bound = np.array([minH, minS, minV])
    upper_bound = np.array([maxH, maxS, maxV])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #cv2.imshow("Original", frame)
    cv2.imshow("HSV Tracker", hsv_frame)
    cv2.imshow("Tracked Object", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

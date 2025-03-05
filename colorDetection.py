import pyrealsense2 as rs
import numpy as np
import cv2
import time
from maestro import Controller  # Ensure the Maestro Controller class is available

# Initialize the servo controller
servo = Controller()

# Servo channel assignments (these may need to be adjusted based on your setup)
HEAD_TILT = 0   # Vertical movement
HEAD_PAN = 1    # Horizontal movement
MOTOR_LEFT = 2  # Left motor
MOTOR_RIGHT = 3 # Right motor

def track_color_and_move():
    print("Starting color tracking...")

    # Configure RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Define color range (adjust for object color)
    lower_color = np.array([10, 100, 100])  # Lower bound for orange
    upper_color = np.array([25, 255, 255])  # Upper bound for orange

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Convert image to HSV color space
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # Create a mask to detect the selected color
            mask = cv2.inRange(hsv_image, lower_color, upper_color)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                frame_area = color_image.shape[0] * color_image.shape[1]
                coverage = (area / frame_area) * 100  # Convert to percentage

                print(f"Color coverage: {coverage:.2f}%")

                if coverage > 25:  # Move forward if object covers more than 25% of frame
                    print("Object detected. Moving forward...")
                    servo.setTarget(MOTOR_LEFT, 7000)  # Move forward
                    servo.setTarget(MOTOR_RIGHT, 7000)
                    time.sleep(2)  # Move for 2 seconds (approx 4 feet)
                    servo.setTarget(MOTOR_LEFT, 6000)  # Stop
                    servo.setTarget(MOTOR_RIGHT, 6000)
                    print("Stopping robot.")
                    break  # Stop tracking after moving forward

            cv2.imshow('Color Tracking', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

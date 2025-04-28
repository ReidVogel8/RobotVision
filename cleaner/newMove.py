import cv2
import pickle
import numpy as np
import pyrealsense2 as rs

# ORB Feature Detector
orb = cv2.ORB_create(nfeatures=1000)

# Data Structure to Hold Object Data
trained_objects = []
object_count = 0
MAX_OBJECTS = 3

# Mouse Callback for Bounding Box
drawing = False
ix, iy = -1, -1
current_roi = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_roi = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_roi = (ix, iy, x, y)

# Start RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow('Training Mode')
cv2.setMouseCallback('Training Mode', draw_rectangle)

print("Training Mode Active")
print("Draw a box around each object and press 's' to save.")

try:
    while object_count < MAX_OBJECTS:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        display = frame.copy()

        # Draw the bounding box if in progress
        if current_roi:
            x1, y1, x2, y2 = current_roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display, f"Object {object_count + 1}/{MAX_OBJECTS}: Draw box and press 's'",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Training Mode', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and current_roi:
            x1, y1, x2, y2 = current_roi
            roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

            if roi.size == 0:
                print("Invalid ROI size. Try again.")
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray_roi, None)

            if descriptors is None or len(keypoints) < 5:
                print("Not enough keypoints. Try again.")
                continue

            name = input(f"Enter name for object {object_count + 1}: ")
            obj_id = object_count + 1  # This will map to ArUco marker ID (1–4)
            trained_objects.append({
                'name': name,
                'id': obj_id,
                'keypoints': keypoints,
                'descriptors': descriptors
            })

            print(f"Object '{name}' saved with ID {obj_id}")
            object_count += 1
            current_roi = None  # Reset ROI

        elif key == ord('q'):
            print("Quit without saving.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Save Descriptors to File
if len(trained_objects) == MAX_OBJECTS:
    # Convert keypoints to savable form
    for obj in trained_objects:
        obj['keypoints'] = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                            for kp in obj['keypoints']]

    with open("trainedObjects.pkl", "wb") as f:
        pickle.dump(trained_objects, f)

    print("All objects saved to trainedObjects.pkl")
else:
    print("Not enough objects trained. Please restart.")

from maestro import Controller
import time
import pyrealsense2 as rs
import numpy as np
import cv2

MIDDLE = 5800
LEFT_WHEEL_PORT = 0
RIGHT_WHEEL_PORT = 1


class ColorMovement:
    _instance = None

    @staticmethod
    def getInst():
        if ColorMovement._instance == None:
            ColorMovement._instance = ColorMovement()
        return ColorMovement._instance

    def __init__(self):
        self.m = Controller()
        pass

    def forward(self):
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                     interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                #move foward if image is covered 25 percent with orange
                lower_color = np.array([0, 140, 100])
                upper_color = np.array([15, 200, 170])
                hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_image, lower_color, upper_color)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    frame_area = color_image.shape[0] * color_image.shape[1]
                    coverage = (area / frame_area) * 100  # Convert to percentage

                    print(f"Color coverage: {coverage:.2f}%")

                    if coverage > 25:  # Move forward if object covers more than 25% of frame
                        print("Object detected. Moving forward...")
                        self.m.setTarget(1,6000)
                        self.m.setTarget(1, 5000)  # Move forward
                        time.sleep(2)  # Move for 2 seconds (approx 4 feet)
                        self.m.setTarget(1, 6000)  # Stop
                        print("Stopping robot.")
                        break  # Stop tracking after moving forward


                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)
        finally:
            # Stop streaming
            pipeline.stop()

x = ColorMovement().getInst()
x.forward()

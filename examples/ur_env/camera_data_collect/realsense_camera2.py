import pyrealsense2 as rs
import cv2
import numpy as np

# 1. Get list of connected devices
ctx = rs.context()
connected_devices = ctx.query_devices()
serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in connected_devices]

# 2. Set up a pipeline per device
pipelines = []
for serial in serial_numbers:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    pipelines.append((serial, pipeline))

# 3. Stream all devices
try:
    while True:
        for serial, pipeline in pipelines:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Show color and depth side-by-side
            images = np.hstack((color_image, depth_colormap))
            window_name = f"Camera {serial}"
            cv2.imshow(window_name, images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for _, pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()

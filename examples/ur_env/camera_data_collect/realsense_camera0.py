import os
import numpy as np
import cv2
import datetime
import pyrealsense2 as rs


class DepthCamera:
    def __init__(self, resolution_width, resolution_height):
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()

        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("Device product line:", device_product_line)

        config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 30)

        self.pipeline.start(config)

        # Get color intrinsics
        color_profile = pipeline_profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, depth_image, color_image

    def get_depth_scale(self):
        return self.depth_scale

    def release(self):
        self.pipeline.stop()


def save_dataset(camera, base_dir, cam_name="camera_0", max_frames=10):
    color_dir = os.path.join(base_dir, cam_name, "color")
    depth_dir = os.path.join(base_dir, cam_name, "depth")

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Save camera parameters (only depth_scale here, add more if you want)
    camera_params = {
        "depth_scale": camera.get_depth_scale(),
        # Optionally add intrinsics here if needed
        "fx": camera.color_intrinsics.fx,
        "fy": camera.color_intrinsics.fy,
        "cx": camera.color_intrinsics.ppx,
        "cy": camera.color_intrinsics.ppy,
        "width": camera.color_intrinsics.width,
        "height": camera.color_intrinsics.height,
    }
    np.save(os.path.join(base_dir, cam_name, "camera_params.npy"), camera_params)

    # Dummy extrinsics (identity matrix) - replace with your calibration if available
    extrinsics = np.eye(4, dtype=np.float32)
    np.save(os.path.join(base_dir, cam_name, "camera_extrinsics.npy"), extrinsics)

    for i in range(max_frames):
        ret, depth_img, color_img = camera.get_frame()
        if not ret:
            print(f"Failed to get frame {i}")
            continue

        color_path = os.path.join(color_dir, f"{i}.png")
        depth_png_path = os.path.join(depth_dir, f"{i}.png")
        depth_npy_path = os.path.join(depth_dir, f"{i}.npy")

        # Save color image (convert BGR to RGB)
        cv2.imwrite(color_path, cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        # Save depth as 16-bit PNG
        cv2.imwrite(depth_png_path, depth_img)

        # Save depth as float32 numpy array scaled to meters
        depth_meters = depth_img.astype(np.float32) * camera.get_depth_scale()
        np.save(depth_npy_path, depth_meters)

        print(f"Saved frame {i+1}/{max_frames} for {cam_name}")

    camera.release()
    print("Finished saving dataset.")


if __name__ == "__main__":
    base_dir = "/home/shreya/Desktop/PI/openpi/examples/ur_env/camera_data_collect"
    cam_name = "camera_1"  # change this for different cameras

    camera = DepthCamera(640, 480)
    save_dataset(camera, base_dir, cam_name=cam_name, max_frames=10)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# depth = cv2.imread("/home/shreya/Desktop/PI/openpi/examples/ur_env/camera_data_collect/camera_0/depth/5.png", cv2.IMREAD_UNCHANGED)
# print(f"Depth min: {depth.min()}, max: {depth.max()}")

# # Mask invalid depth (0 or max)
# invalid_mask = (depth == 0) | (depth == 65535)
# depth_filtered = depth.astype(np.float32)
# depth_filtered[invalid_mask] = np.nan

# min_val = np.nanmin(depth_filtered)
# max_val = np.nanmax(depth_filtered)

# depth_normalized = (depth_filtered - min_val) / (max_val - min_val)

# colormap = plt.get_cmap('plasma')
# depth_colored = colormap(depth_normalized)

# # Replace NaNs with black
# depth_colored[np.isnan(depth_normalized)] = [0, 0, 0, 1]

# plt.imshow(depth_colored)
# plt.axis('off')
# plt.title("Filtered Depth Visualization")
# plt.show()

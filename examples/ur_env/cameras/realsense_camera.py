import os
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    from cameras.realsense_camera import RealSenseCamera, get_device_ids
    REALSENSE_AVAILABLE = True
except ImportError:
    print("PyRealSense2 not available, using dummy cameras")
    REALSENSE_AVAILABLE = False
    
    # Define dummy versions
    def get_device_ids():
        """Dummy function that returns an empty list when PyRealSense2 is not available."""
        return []
    
    class RealSenseCamera:
        """Dummy camera class when PyRealSense2 is not available."""
        def __init__(self, device_id=None, flip=False):
            self._pipeline = self.DummyPipeline()
            
        def read(self):
            """Return dummy color and depth images."""
            color = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.zeros((480, 640), dtype=np.uint16)
            
            # Add a basic pattern to the dummy image
            color[200:280, 300:380] = [255, 0, 0]  # Red rectangle
            return color, depth
        
        class DummyPipeline:
            def stop(self):
                pass
# from gello.cameras.camera import CameraDriver

def check_usb_devices():
    """Check USB devices to help debug camera connection issues."""
    try:
        import subprocess
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        print("USB devices detected:")
        print(result.stdout)
        
        # Look specifically for Intel devices
        intel_devices = [line for line in result.stdout.split('\n') if 'Intel' in line]
        if intel_devices:
            print("Intel USB devices found:")
            for device in intel_devices:
                print(f"  - {device}")
        else:
            print("No Intel USB devices found - RealSense cameras may not be connected")
    except Exception as e:
        print(f"Error checking USB devices: {e}")

def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


# class RealSenseCamera(CameraDriver):
class RealSenseCamera():
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs

        self._device_id = device_id

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)
        self._flip = flip

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    # check_usb_devices()
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)

#!/usr/bin/env python3
# filepath: /home/shreya/Desktop/Pi_zero/openpi/examples/ur3/ur_env/ur3_inference.py

"""
UR3 Robot Controller for OpenPI Inference.

This script implements a controller for running inference with OpenPI models on UR3 robots,
supporting multi-camera input and natural language instructions. It uses the OpenPI policy 
framework to directly generate robot actions from visual observations and language commands.
"""

import os                  # For file operations and environment variables
import sys                 # For system-specific parameters and functions
import time                # For time-related functions like sleep
import traceback           # For detailed error tracing
import argparse            # For parsing command line arguments
import json                # For JSON serialization
import datetime            # For date and time operations
from pathlib import Path   # For cross-platform path manipulations
import dataclasses
import jax
from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import data_loader as _data_loader

import numpy as np         # For numerical operations
import cv2                 # For computer vision operations

# Set up Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
src_path = os.path.join(project_root, 'openpi')
sys.path.append(src_path)
sys.path.append('/home/shreya/Desktop/PI/openpi')  # For OpenPI imports

# External hardware dependencies
try:
    from urtde_controller2 import URTDEController, URTDEControllerConfig
    print("Successfully imported UR controller")
except ImportError as e:
    print(f"Error importing UR controller: {e}")
    sys.exit(1)

# Import OpenPI modules
try:
    from openpi.policies import policy_config
    # Import with correct name
    from openpi.training import config as openpi_config
    from openpi.shared import download
    from openpi.policies import ur3_policy
    from src.openpi.policies.ur3_policy import UR3ThreeCameraInputs, UR3Outputs
    print("Successfully imported OpenPI policy modules")
except ImportError as e:
    print(f"Error importing OpenPI policy modules: {e}")
    try:
        from openpi.policies.ur3_policy import UR3ThreeCameraInputs, UR3Outputs
        print("Successfully imported from alternative path")
    except ImportError:
        print("Could not import UR3 policy modules")
        sys.exit(1)

# Import utility functions
try:
    from hardware_env.ur3e_utils import Rate
    from hardware_env.two_stage import TwoStage
    from hardware_env.save_csv import save_frame
    print("Successfully imported hardware utilities")
except ImportError as e:
    print(f"Error importing hardware utilities: {e}")
    
    # Define fallback utility classes if imports fail
    class Rate:
        """Simple rate limiter for controlling loop frequency."""
        def __init__(self, rate_hz):
            self.period = 1.0 / rate_hz
            self.last_time = time.time()
        
        def sleep(self):
            current_time = time.time()
            elapsed = current_time - self.last_time
            sleep_time = max(0, self.period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.last_time = time.time()
    
    def save_frame(path, timestamp, joints, ee_pose, filename):
        """Save robot state to CSV file for trajectory logging."""
        filepath = os.path.join(path, filename)
        exists = os.path.exists(filepath)
        with open(filepath, 'a') as f:
            if not exists:
                f.write("timestamp,j1,j2,j3,j4,j5,j6,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w\n")
            joint_str = ','.join([f"{j:.6f}" for j in joints])
            ee_str = ','.join([f"{p:.6f}" for p in ee_pose])
            f.write(f"{timestamp},{joint_str},{ee_str}\n")
    
    class TwoStage:
        """Fallback task evaluator for computing rewards."""
        def __init__(self, verbose=False):
            self.verbose = verbose
        
        def reward(self, obs):
            # Simple placeholder reward function
            return 0.0


class OpenPIUR3Controller:
    """Main controller class that integrates OpenPI with UR3 robot.
    
    This class handles:
    1. Camera setup and image acquisition
    2. Robot state management
    3. OpenPI policy inference
    4. Action execution and safety monitoring
    5. Logging and data collection
    """
    
    def __init__(self, model_path, controller_config=None, camera_indices=(4,6,14), verbose=True, model_name='pi0_fast_ur3'):
        """Initialize the controller with model and hardware connections.
        
        Args:
            model_path: Path to the OpenPI model checkpoint (local or S3)
            config: Optional controller configuration
            camera_indices: Indices for camera devices (base, wrist1, wrist2)
            verbose: Whether to print detailed logs
            model_name: Name of the OpenPI model config to use
        """
        self.verbose = verbose  # Store verbose flag for logging control
        self.model_name = model_name  # Store model name for policy creation
        self.model_path = model_path  # Store model path for later use
        
        # Create output directory for run data with timestamp
        self.output_dir = Path("openpi_ur3_runs") / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create directory, ok if it exists
        print(f"Saving run data to: {self.output_dir}")  # Inform user where data is saved
        
        # Initialize log file for this run
        self.log_file = open(self.output_dir / "run_log.txt", "w")  # Open log file for writing
        self.log("OpenPI UR3 Controller initializing...")  # Log initialization start
        
        # Create controller configuration if none provided
        if controller_config is None:
            # Set up default configuration for UR3 robot
            self.config = URTDEControllerConfig(
                task="two_stage",            # Type of task to perform
                controller_type="CARTESIAN_DELTA",  # Control in Cartesian space
                max_delta=0.06,              # Maximum movement per step (in meters)
                mock=1,                      # 0 = real robot, 1 = mock/simulation
                hostname="192.168.77.232",   # Hostname for robot server
                robot_port=50002,            # Port for robot communication
                robot_ip="192.168.77.22",    # IP address of the robot
                hz=100                       # Control frequency in Hz
            )
        else:
            self.config = controller_config  # Use provided configuration
        
        # In OpenPIUR3Controller.__init__
        # Initialize the robot controller
        self.log("Initializing UR3 controller...")
        try:
            # Check if we're in mock mode
            if hasattr(self.config, 'mock') and self.config.mock:
                # Use mock controller
                self.log("Using mock controller (mock mode)")
                from hardware_env.mocks import MockURTDEController
                self.controller = MockURTDEController(self.config, task=self.config.task)
            else:
                # Create real controller instance with the configuration
                self.controller = URTDEController(self.config, task=self.config.task)
            
            self.log("Controller initialized successfully")
        except Exception as e:
            # Log error and re-raise exception
            self.log(f"Error initializing controller: {e}")
            raise

        # Initialize task evaluator for reward calculation
        self.task = TwoStage(verbose=verbose)  # Task-specific evaluator
        
                
        # Initialize control rate 
        self.control_rate = 10.0  # 10 Hz default control rate
        if hasattr(self.config, 'hz') and self.config.hz > 0:
            # Cap control rate to be no more than 1/3 of robot's control frequency
            self.control_rate = min(self.config.hz / 3.0, 10.0)
        self.log(f"Control rate set to {self.control_rate} Hz")
        
        # Initialize cameras
        self.camera_indices = camera_indices
        self.cameras = []  # Will hold camera objects
        print(type(self.camera_indices))
        self.init_cameras()  # Initialize camera connections
        
        # Initialize OpenPI policy framework
        self.log(f"Loading OpenPI model config: {model_name}")
        try:
            # Get model config
            self.model_config = openpi_config.get_config(model_name)
            
            # Download checkpoint if needed
            if model_path.startswith('s3://'):
                self.log(f"Downloading checkpoint from: {model_path}")
                self.checkpoint_dir = download.maybe_download(model_path)
                self.log(f"Checkpoint downloaded to: {self.checkpoint_dir}")
            else:
                self.checkpoint_dir = model_path
            
            # Note: We create a fresh policy for each inference to prevent memory leaks
            self.log("OpenPI policy framework initialized successfully")
        except Exception as e:
            # Log error and re-raise exception
            self.log(f"Error initializing OpenPI policy framework: {e}")
            raise
        
        # Initialize trajectory logging
        self.csv_file_name = f"openpi_trajectory_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
        self.csv_path = self.output_dir / self.csv_file_name  # Full path to CSV file
        
        # Initialize state tracking variables
        self.current_step = 0  # Step counter
        self.total_reward = 0  # Accumulated reward
        
        self.log("Initialization complete")  # Log successful initialization

    def run_inference_with_openpi(self, prompt, images=None, state=None):
        """Run inference using the OpenPI policy framework with camera images and robot state."""
        # try:
        # Force JAX to use CPU instead of GPU
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        # Use mock policy if we're in mock mode
        if hasattr(self.config, 'mock') and self.config.mock:
            self.log("Using mock policy for inference (mock mode)")
            from hardware_env.mocks import MockPolicy
            policy = MockPolicy()
        else:
            # Create policy from stored model config and checkpoint directory
            self.log("Creating policy for inference")
            policy = policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        
        # Use real camera images and robot state if provided, otherwise use dummy example
        if images is not None and len(images) >= 3 and state is not None:
            self.log("Using real camera images and robot state")
            
            # Create raw example with the structure used by UR3ThreeCameraInputs
            raw_example = {
                # Camera images
                "base_rgb": images[0],
                "wrist_rgb": images[1],
                # "wrist2_rgb": images[2],
                
                # State - only joints and gripper as expected by UR3ThreeCameraInputs
                "joints": state["joints"],
                "gripper": state["gripper"],
                
                # Prompt
                "prompt": prompt
            }
            
            input_transformer = ur3_policy.UR3ThreeCameraInputs()

            # Transform raw example into the format expected by the model
            example = input_transformer(raw_example)

            # Verify the state dimension after transformation
            self.log(f"State dimension after transformation: {example['state'].shape}")

            # # If for some reason the state isn't properly padded, do it manually
            # if example['state'].shape[0] != 32:
            #     self.log(f"WARNING: Transform didn't pad correctly! Manually padding from {example['state'].shape[0]} to 32")
            #     padded_state = np.zeros(32, dtype=example['state'].dtype)
            #     padded_state[:min(32, len(example['state']))] = example['state'][:min(32, len(example['state']))]
            #     example['state'] = padded_state

            # Log state and transformed state dimensions for debugging
            state_vector = np.concatenate([state["joints"], state["gripper"]])
            self.log(f"Original state vector shape: {state_vector.shape}")
            self.log(f"Transformed state shape: {example['state'].shape}")
        else:
            # Use dummy example as fallback
            self.log("Using dummy example for inference")
            raw_example = ur3_policy.make_ur3_example()
            raw_example["prompt"] = prompt
            
            # Transform dummy example using UR3ThreeCameraInputs
            input_transformer = ur3_policy.UR3ThreeCameraInputs()
            example = input_transformer(raw_example)
        
        # try:
            # Run inference with the transformed example
        self.log("Running inference with OpenPI policy")
        print("policy inferring...........")
        # Use only the action-related fields to avoid broadcasting errors
        # action_inputs = {
        #     "image": example["image"],
        #     "image_mask": example["image_mask"],
        #     "state": example["state"],
        #     "prompt": example["prompt"],
        #     # ADD THESE LINES TO FIX THE ERROR:
        #     "joints": raw_example["joints"],  # Add original joints
        #     "gripper": raw_example["gripper"] # Add original gripper
        #     "base_rgb": images[0]
        # }
        # Create a properly structured action_inputs dictionary with both transformed and raw values
        # Create a properly structured action_inputs dictionary with both transformed and raw values
        # action_inputs = {
        #     # Transformed values from example
        #     "image": example["image"],
        #     "image_mask": example["image_mask"],
        #     "state": example["state"],  # This has 32 dimensions
        #     "prompt": example["prompt"],
        #     "joints": raw_example["joints"],  # Add original joints # Add original gripper
        #     "gripper": raw_example["gripper"], # Add original gripper
        #     "base_rgb": raw_example["base_rgb"],
        #     "wrist1_rgb": raw_example["wrist1_rgb"],
        #     "wrist2_rgb": raw_example["wrist2_rgb"],
        #         # "wrist1_rgb": images[1],
        #         # "wrist2_rgb": images[2],


        # }
        # This structured approach will work with the LeRobotUR3DataConfig transform pipeline
        action_inputs = {
            # Necessary transformed fields
            "image": example["image"],
            "image_mask": example["image_mask"],
            "state": example["state"].astype(np.float32),  # 32D state
            "prompt": example["prompt"],
            
            # Raw fields needed by internal transforms
            "joints": np.asarray(raw_example["joints"]).astype(np.float32),
            "gripper": np.asarray(raw_example["gripper"]).astype(np.float32)
        }

        # Verify state is 32D (required for normalization)
        print(f"state shape: {action_inputs['state'].shape}")

        # Run inference with the properly structured inputs
        result = policy.infer(action_inputs)

        print("policy inferred")
        # except Exception as inner_e:
        #     # If standard inference fails, use MockPolicy as fallback
        #     self.log(f"Standard inference failed: {inner_e}. Using fallback approach.")
            
        #     # Create and use MockPolicy for fallback
        #     from hardware_env.mocks import MockPolicy
        #     mock_policy = MockPolicy()
        #     result = mock_policy.infer({})
        
        # Delete the policy to free up memory
        del policy
        import gc
        gc.collect()  # Force garbage collection

        # Log original action shape
        self.log(f"Original actions shape: {result['actions'].shape}")
        
        # Use UR3Outputs to process the model output
        output_transformer = ur3_policy.UR3Outputs()
        processed_result = output_transformer(result)
        
        self.log(f"Final action shape: {processed_result['actions'].shape}")
        
        return processed_result
            
        # except Exception as e:
        #     self.log(f"Error during OpenPI inference: {e}")
        #     traceback.print_exc(file=self.log_file)
        #     # Return safe default action on error
        #     return {"actions": np.zeros(7)}

    def log(self, message):
        """Log message to both console and log file with timestamp.
        
        Args:
            message: Message to log
        """
        print(message)  # Print to console
        
        # Check if log_file exists and is open before writing
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            try:
                # Add timestamp to log entries
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"[{timestamp}] {message}\n")  # Write to file
                self.log_file.flush()  # Ensure it's written immediately
            except (ValueError, IOError) as e:
                # Log file might be closed - just print to console
                print(f"Warning: Could not write to log file: {e}")


    # In OpenPIUR3Controller.init_cameras
    def init_cameras(self):
        """Initialize connections to all cameras."""
        self.cameras = []  # Clear existing camera list
        
        # Check if we're in mock mode
        using_mock = hasattr(self.config, 'mock') and self.config.mock
        
        for i, cam_idx in enumerate(self.camera_indices):
            try:
                if using_mock:
                    # Use mock cameras in mock mode
                    from hardware_env.mocks import MockCamera
                    cap = MockCamera(camera_id=i)
                    self.log(f"Initialized mock camera {i}")
                else:
                    # Try to open real camera with OpenCV
                    cap = cv2.VideoCapture(cam_idx)
                    if not cap.isOpened():
                        # Log warning if camera couldn't be opened
                        self.log(f"Warning: Failed to open camera {cam_idx}")
                        cap = None
                    else:
                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
                        self.log(f"Camera {cam_idx} initialized")
                
                self.cameras.append(cap)
            except Exception as e:
                # Log error and add None placeholder
                self.log(f"Error initializing camera {cam_idx}: {e}")
                self.cameras.append(None)
                
        
    def capture_images(self):
        """Capture images from all cameras.
        
        Returns:
            List of images from base, wrist1, and wrist2 cameras
        """
        images = []  # List to store captured images
        for i, cap in enumerate(self.cameras):
            # print(f"printing cap:{cap}")
            if cap is not None and cap.isOpened():
                try:
                    # Try to capture a frame
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR (OpenCV format) to RGB (model input format)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        images.append(rgb_frame)
                        
                        # Save image to disk for logging
                        img_path = self.output_dir / f"step_{self.current_step:04d}_camera_{i}.jpg"
                        cv2.imwrite(str(img_path), frame)  # Save original BGR image
                    else:
                        # Use blank image if capture failed
                        self.log(f"Warning: Failed to capture from camera {i}")
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8)) # scale down (224,224,3)
                except Exception as e:
                    # Use blank image if exception occurred
                    self.log(f"Error capturing from camera {i}: {e}")
                    images.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                # Use blank image if camera not available
                self.log(f"Warning: Camera {i} not available")
                images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Ensure we have exactly 3 images (base, wrist1, wrist2)
        while len(images) < 3:
            # Add blank images if needed
            images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return images[:3]  # Return only the first 3 images
    

    def get_robot_state(self):
        """Get current robot state from the controller's get_state method."""
        # try:
        # Get state from robot controller
        obs, in_good_range = self.controller.get_state()
        
        # IMPORTANT: Handle array vs scalar for in_good_range explicitly
        if isinstance(in_good_range, (np.ndarray, list)):
            # Convert array to scalar boolean safely
            scalar_in_good_range = bool(in_good_range[0]) if len(in_good_range) > 0 else False
        else:
            # Already a scalar
            scalar_in_good_range = bool(in_good_range)
        
        # Log if not in good range
        if not scalar_in_good_range and self.verbose:
            self.log("Warning: Robot not in good range")
        
        # Extract joint positions safely - handle both formats
        if 'joint_positions' in obs:
            joint_positions = np.array(obs['joint_positions'], dtype=np.float32)
        else:
            # Fallback to zeros
            joint_positions = np.zeros(7, dtype=np.float32)
        
        # # Extract end effector position and orientation
        # ee_pos = np.array(obs.get('robot0_eef_pos', np.zeros(3)), dtype=np.float32)
        # ee_quat = np.array(obs.get('robot0_eef_quat', np.array([0, 0, 0, 1])), dtype=np.float32)
        
        # Handle gripper position safely
        if 'robot0_gripper_qpos' in obs and obs['robot0_gripper_qpos'] is not None:
            gripper_qpos = obs['robot0_gripper_qpos']
            # Extract scalar value safely
            if isinstance(gripper_qpos, (list, np.ndarray)) and len(gripper_qpos) > 0:
                gripper_qpos = float(gripper_qpos[0])
            else:
                try:
                    gripper_qpos = float(gripper_qpos)
                except (TypeError, ValueError):
                    gripper_qpos = 0.5
        else:
            gripper_qpos = 0.5
        
        # Create state dictionary for the model
        state = {
            "joints": joint_positions[:6],  # First 6 joints
            "gripper": np.array([gripper_qpos], dtype=np.float32),  # Gripper position
            # "ee_pos": ee_pos,  # End effector position
            # "ee_quat": ee_quat  # End effector orientation
        }
        
        # Return the state and original observations
        return state, obs
            
        # except Exception as e:
        #     # Log error and return default values
        #     self.log(f"Error getting robot state: {e}")
        #     traceback.print_exc(file=self.log_file)
            
        #     # Return default state for mock mode
        #     return {
        #         "joints": np.zeros(6, dtype=np.float32),
        #         "gripper": np.array([0.5], dtype=np.float32),
        #         # "ee_pos": np.zeros(3, dtype=np.float32),
        #         # "ee_quat": np.array([0, 0, 0, 1], dtype=np.float32)
        #     }, {}
                
    def execute_action(self, action):
        """Execute action on the robot using controller's update method.
        
        Args:
            action: 7D action vector [dx, dy, dz, drx, dry, drz, gripper]
                
        Returns:
            Boolean indicating success
        """
        # try:
            # Ensure action is a 1D numpy array with proper dimensions
        action = np.asarray(action).flatten()
        if len(action) != 7:
            # Pad or truncate to ensure exactly 7 dimensions
            if len(action) < 7:
                action = np.pad(action, (0, 7 - len(action)))
            else:
                action = action[:7]

        # Important: Clip gripper action to [0, 1] range
        action[6] = np.clip(action[6], 0.0, 1.0)

        # Convert negative gripper values to positive if needed
        if action[6] < 0:
            action[6] = 0.5 + 0.5 * action[6]  # Map [-1, 0] to [0, 0.5]
                    
        # Log the action for debugging
        self.log(f"Executing action: {action}")
        
        # IMPORTANT: Reshape/modify the action to match what update expects
        # Create a properly shaped action - this fixes the dimension mismatch
        pos_delta = action[:3].reshape(-1)  # Ensure it's a flat array
        rot_delta = action[3:6].reshape(-1)  # Ensure it's a flat array
        gripper = float(action[6])  # Scalar value
        
        # Pass separate components to a custom update function to avoid dimension issues
        success = self.custom_update(pos_delta, rot_delta, gripper)
        
        return success
            
        # except Exception as e:
        #     # Log error and return failure
        #     self.log(f"Error executing action: {e}")
        #     traceback.print_exc(file=self.log_file)
        #     return False
        
    def custom_update(self, pos_delta, rot_delta, gripper):
        """Custom update function that handles different components separately.
        
        Args:
            pos_delta: Position delta [dx, dy, dz]
            rot_delta: Rotation delta [drx, dry, drz]
            gripper: Gripper value [0-1]
            
        Returns:
            Boolean indicating success
        """
        # try:
            # Get current state
        obs, _ = self.controller.get_state()
        
        # Get current EE pose
        current_pos = np.array(obs["robot0_eef_pos"])
        current_rot = np.array(obs["robot0_eef_quat"])

        
        # Compute new EE pose - position is straightforward addition
        new_pos = current_pos + pos_delta
        
        # For rotation, handle different quaternion lengths properly
        if len(current_rot) == 4 and len(rot_delta) == 3:
            # Current is quaternion [x,y,z,w], delta is euler [rx,ry,rz]
            # Convert rotation delta to quaternion before adding
            from scipy.spatial.transform import Rotation
            # Convert small euler angles to quaternion
            rot_quat = Rotation.from_euler('xyz', rot_delta).as_quat()  # [x,y,z,w] format
            
            # Apply the rotation (quaternion multiplication, not addition)
            current_rot_obj = Rotation.from_quat(current_rot)
            delta_rot_obj = Rotation.from_quat(rot_quat)
            new_rot_obj = delta_rot_obj * current_rot_obj
            new_rot = new_rot_obj.as_quat()
        elif len(current_rot) == len(rot_delta):
            # Same lengths, we can add directly (simple case)
            new_rot = current_rot + rot_delta
        else:
            # Incompatible lengths - log warning and just use current rotation
            self.log(f"Warning: Incompatible rotation shapes: current {current_rot.shape}, delta {rot_delta.shape}")
            new_rot = current_rot
        
        # Combine into full pose
        new_pose = np.concatenate([new_pos, new_rot, np.array([0])])
        
        # Log the pose we're moving to
        self.log(f"Moving to pose: pos={new_pos}, rot={new_rot}")
        
        # Execute movement operations - avoid array truth value checks
        # try:
        self.controller.move_to_eef_positions(new_pose, delta=False)
        move_succeeded = True
        # except Exception as move_e:
        #     self.log(f"Error in move_to_eef_positions: {move_e}")
        #     traceback.print_exc(file=self.log_file)
        #     move_succeeded = False
        
        # Only update gripper if move succeeded and gripper is available
        gripper_succeeded = False
        if move_succeeded:
            # Check if the controller has a use_gripper attribute 
            if hasattr(self.controller, 'use_gripper'):
                # Convert use_gripper to scalar boolean safely
                use_gripper_value = self.controller.use_gripper
                if isinstance(use_gripper_value, (np.ndarray, list)):
                    use_gripper = bool(use_gripper_value[0]) if len(use_gripper_value) > 0 else False
                else:
                    use_gripper = bool(use_gripper_value)
                
                if use_gripper:
                    try:
                        # Ensure gripper is a scalar float
                        scalar_gripper = float(gripper)
                        self.controller.update_gripper(scalar_gripper, blocking=True)
                        gripper_succeeded = True
                    except Exception as gripper_e:
                        self.log(f"Error in update_gripper: {gripper_e}")
                        traceback.print_exc(file=self.log_file)
                        gripper_succeeded = False
                else:
                    gripper_succeeded = True  # No need to update gripper
            else:
                # No gripper attribute
                gripper_succeeded = True
        
        # Return overall success
        return move_succeeded and gripper_succeeded
            
        # except Exception as e:
        #     self.log(f"Error in custom_update: {e}")
        #     traceback.print_exc(file=self.log_file)
        #     return False
        
    def return_to_home(self):
        """Return the robot to the home position."""
        try:
            self.log("Returning to home position...")
            # Use the controller's reset method
            self.controller.reset(randomize=False)
            self.log("Reset env")
            return True
        except Exception as e:
            self.log(f"Error returning to home position: {e}")
            traceback.print_exc(file=self.log_file)
            return False
        
    def compute_reward(self):
        """Compute reward based on current state.
        
        Returns:
            float: Reward value for the current state
        """
        try:
            # Get current robot state
            state, obs = self.get_robot_state()
                   # Add missing fields that the reward function might need
            if obs is None:
                obs = {}
                
            # Explicitly add required fields with default values if missing
            if 'robot0_desired_gripper_qpos' not in obs:
                # Use current gripper position as default
                if 'robot0_gripper_qpos' in obs:
                    obs['robot0_desired_gripper_qpos'] = obs['robot0_gripper_qpos']
                else:
                    obs['robot0_desired_gripper_qpos'] = 0.5
            
            # Use the task evaluator to compute reward
            if hasattr(self, 'task') and hasattr(self.task, 'reward'):
                reward = self.task.reward(obs)
            else:
                # Default reward if task evaluator not available
                reward = 0.0
                
            # Log and return reward
            self.log(f"Reward: {reward}")
            return reward
        except Exception as e:
            self.log(f"Error computing reward: {e}")
            traceback.print_exc(file=self.log_file)
            return 0.0  # Default reward on error

    def check_task_complete(self):
        """Check if the current task is complete.
        
        Returns:
            bool: True if task is complete, False otherwise
        """
        # try:
        # Get current robot state
        state, obs = self.get_robot_state()
        
        # Check if task has a completion criterion
        if hasattr(self, 'task') and hasattr(self.task, 'is_done'):
            done = self.task.is_done(obs)
            return done
        
        # Alternative completion check for TwoStage tasks
        if hasattr(self, 'task') and hasattr(self.task, 'reward'):
            # Consider task complete if reward exceeds threshold
            reward = self.task.reward(obs)
            done = reward > 0.9  # Task is complete if reward is close to 1.0
            return done
            
        # Default - not complete
        return False
        # except Exception as e:
        #     self.log(f"Error checking task completion: {e}")
        #     traceback.print_exc(file=self.log_file)
        #     return False  # Default: task not complete on error
        
    def run_task(self, prompt, max_steps=50):
        """Run a task with the specified prompt."""
        self.log(f"Starting task with prompt: {prompt}")
        self.current_step = 0
        total_reward = 0.0
        
        # try:
        while self.current_step < max_steps:
            # Capture images from cameras
            images = self.capture_images()
            
            # Get current robot state
            state, _ = self.get_robot_state()
            
            # Run inference to get action
            result = self.run_inference_with_openpi(prompt, images, state)
            if not result or "actions" not in result:
                self.log("No valid action returned from inference")
                break
                
            # Execute the action
            success = self.execute_action(result["actions"])
            if not success:
                self.log("Failed to execute action")
                break
                
            # Compute reward
            reward = self.compute_reward()
            total_reward += reward
            
            # Log progress
            self.log(f"Step {self.current_step}: reward = {reward}, total = {total_reward}")
            
            # Check if task is complete
            if self.check_task_complete():
                self.log("Task completed successfully!")
                break
                
            # Increment step counter
            self.current_step += 1
            
            # Sleep to maintain control rate if needed
            time.sleep(1.0 / self.control_rate)
                
        # except KeyboardInterrupt:
        #     self.log("Task interrupted by user")
        # except Exception as e:
        #     self.log(f"Error during task execution: {e}")
        #     traceback.print_exc(file=self.log_file)
        # finally:
        #     # Always return to home position for safety
        #     self.return_to_home()
        #     self.log(f"Task completed. Total steps: {self.current_step}, Total reward: {total_reward}")
            
        return total_reward

    def cleanup_resources(self):
        """Clean up hardware resources."""
        try:
            # Close camera connections
            print("Closing camera connections...")
            for i, cap in enumerate(self.cameras):
                if cap is not None:
                    try:
                        cap.release()
                        print(f"Camera {i} released")
                    except Exception as e:
                        print(f"Error releasing camera {i}: {e}")
            
            # Additional cleanup for other resources can go here

            print("All resources cleaned up")
                        
        except Exception as e:
            print(f"Error during resource cleanup: {e}")

def main():
    """Main function to parse arguments and run the controller."""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Run OpenPI inference on UR3 robot")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the OpenPI model checkpoint")
    parser.add_argument("--prompt", type=str, default="Press the red button", 
                        help="Task prompt for the OpenPI model")
    parser.add_argument("--steps", type=int, default=10, 
                        help="Maximum number of steps for the task")
    parser.add_argument("--base_camera", type=int, default=4,
                        help="Camera ID for base camera")   
    parser.add_argument("--wrist1_camera", type=int, default=6,
                        help="Camera ID for wrist1 camera")  
    parser.add_argument("--wrist2_camera", type=int, default=14,
                        help="Camera ID for wrist2 camera")
    parser.add_argument("--robot_ip", type=str, default="192.168.77.22",
                        help="IP address of the UR3 robot")
    parser.add_argument("--hostname", type=str, default="192.168.77.232",
                        help="Hostname for the controller")
    parser.add_argument("--port", type=int, default=50002,
                        help="Port for the controller")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode without real robot")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--model_name", type=str, default="pi0_fast_ur3",
                        help="Name of the OpenPI model config to use")
    args = parser.parse_args()  # Parse command line arguments
    
    # Create robot controller configuration
    config = URTDEControllerConfig(
        task="two_stage",                   # Task type
        controller_type="CARTESIAN_DELTA",  # Control mode
        max_delta=0.05,                     # Maximum movement per step
        mock=1 if args.mock else 0,         # Mock mode if requested
        hostname=args.hostname,             # Server hostname
        robot_port=args.port,               # Server port
        robot_ip=args.robot_ip,             # Robot IP
        hz=100                              # Control frequency
    )
    
    # Create and run controller with error handling
    controller = None
    try:
        # Initialize controller with model and hardware
        controller = OpenPIUR3Controller(
            model_path=args.model_path,
            controller_config=config,
            camera_indices=(args.base_camera, args.wrist1_camera, args.wrist2_camera),
            verbose=args.verbose,
            model_name=args.model_name
        )
        
        # Run the task with the given prompt
        controller.run_task(args.prompt, max_steps=args.steps)
    except Exception as e:
        # Handle any uncaught exceptions
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even if errors occur
        if controller is not None:
            try:
                # Try to close log file first if it exists and is open
                if hasattr(controller, 'log_file') and controller.log_file and not controller.log_file.closed:
                    print("Ensuring log file is closed properly...")
                    controller.log_file.flush()
                    controller.log_file.close()
                    
                # Then call cleanup for other resources
                controller.cleanup_resources()
            except Exception as cleanup_error:
                print(f"Error during final cleanup: {cleanup_error}")



# Standard Python idiom to call the main function when the script is run directly
if __name__ == "__main__":
    main()
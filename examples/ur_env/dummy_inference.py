# #!/usr/bin/env python3
# # filepath: /home/shreya/Desktop/Pi_zero/openpi/examples/ur3/ur_env/ur3_inference.py

# """
# UR3 Robot Controller for OpenPI Inference.

# This script implements a controller for running inference with OpenPI models on UR3 robots,
# supporting multi-camera input and natural language instructions.
# """

# import os                  # For file operations and environment variables
# import sys                 # For system-specific parameters and functions
# import time                # For time-related functions like sleep
# import traceback          # For detailed error tracing
# import argparse            # For parsing command line arguments
# import json                # For JSON serialization
# import datetime            # For date and time operations
# from pathlib import Path   # For cross-platform path manipulations

# # Scientific and vision libraries
# import numpy as np         # For numerical operations
# import cv2                 # For computer vision operations

# # Set up Python path for imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# src_path = os.path.join(project_root, 'openpi')
# sys.path.append(src_path)
# sys.path.append('/home/shreya/Desktop/PI/openpi')  # For OpenPI imports

# # External hardware dependencies
# try:
#     from urtde_controller2 import URTDEController, URTDEControllerConfig
#     print("Successfully imported UR controller")
# except ImportError as e:
#     print(f"Error importing UR controller: {e}")
#     sys.exit(1)

# # Import OpenPI modules
# try:
#     from openpi.policies import policy_config
#     from openpi.training import config
#     from openpi.shared import download
#     from openpi.policies import ur3_policy
#     from src.openpi.policies.ur3_policy import UR3ThreeCameraInputs, UR3Outputs
#     print("Successfully imported OpenPI policy modules")
# except ImportError as e:
#     print(f"Error importing OpenPI policy modules: {e}")
#     try:
#         from openpi.policies.ur3_policy import UR3ThreeCameraInputs, UR3Outputs
#         print("Successfully imported from alternative path")
#     except ImportError:
#         print("Could not import UR3 policy modules")
#         sys.exit(1)

# # Import utility functions
# try:
#     from hardware_env.ur3e_utils import Rate
#     from hardware_env.two_stage import TwoStage
#     from hardware_env.save_csv import save_frame
#     print("Successfully imported hardware utilities")
# except ImportError as e:
#     print(f"Error importing hardware utilities: {e}")
    
#     # Define fallback Rate class if import fails
#     class Rate:
#         def __init__(self, rate_hz):
#             self.period = 1.0 / rate_hz
#             self.last_time = time.time()
        
#         def sleep(self):
#             current_time = time.time()
#             elapsed = current_time - self.last_time
#             sleep_time = max(0, self.period - elapsed)
#             if sleep_time > 0:
#                 time.sleep(sleep_time)
#             self.last_time = time.time()
    
#     # Define fallback save_frame function
#     def save_frame(path, timestamp, joints, ee_pose, filename):
#         filepath = os.path.join(path, filename)
#         exists = os.path.exists(filepath)
#         with open(filepath, 'a') as f:
#             if not exists:
#                 f.write("timestamp,j1,j2,j3,j4,j5,j6,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w\n")
#             joint_str = ','.join([f"{j:.6f}" for j in joints])
#             ee_str = ','.join([f"{p:.6f}" for p in ee_pose])
#             f.write(f"{timestamp},{joint_str},{ee_str}\n")
    
#     # Define fallback TwoStage class
#     class TwoStage:
#         def __init__(self, verbose=False):
#             self.verbose = verbose
        
#         def reward(self, obs):
#             # Simple placeholder reward function
#             return 0.0


# class OpenPIUR3Controller:
#     """Main controller class that integrates OpenPI with UR3 robot."""
    
#     def __init__(self, model_path, config=None, camera_indices=(0, 1, 2), verbose=True, model_name='pi0_ur3'):
#         """Initialize the controller with model and hardware connections.
        
#         Args:
#             model_path: Path to the OpenPI model checkpoint
#             config: Optional controller configuration
#             camera_indices: Indices for camera devices (base, wrist1, wrist2)
#             verbose: Whether to print detailed logs
#             model_name: Name of the OpenPI model config to use
#         """
#         self.verbose = verbose  # Store verbose flag for logging control
#         self.model_name = model_name  # Store model name for policy creation
#         self.model_path = model_path  # Store model path for later use
        
#         # Create output directory for run data with timestamp
#         self.output_dir = Path("openpi_ur3_runs") / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         self.output_dir.mkdir(parents=True, exist_ok=True)  # Create directory, ok if it exists
#         print(f"Saving run data to: {self.output_dir}")  # Inform user where data is saved
        
#         # Initialize log file for this run
#         self.log_file = open(self.output_dir / "run_log.txt", "w")  # Open log file for writing
#         self.log("OpenPI UR3 Controller initializing...")  # Log initialization start
        
#         # Create controller configuration if none provided
#         if config is None:
#             # Set up default configuration for UR3 robot
#             self.config = URTDEControllerConfig(
#                 task="two_stage",            # Type of task to perform
#                 controller_type="CARTESIAN_DELTA",  # Control in Cartesian space
#                 max_delta=0.06,              # Maximum movement per step (in meters)
#                 mock=0,                      # 0 = real robot, 1 = mock/simulation
#                 hostname="192.168.77.134",   # Hostname for robot server
#                 robot_port=50002,            # Port for robot communication
#                 robot_ip="192.168.77.22",    # IP address of the robot
#                 hz=100                       # Control frequency in Hz
#             )
#         else:
#             self.config = config  # Use provided configuration
        
#         # Initialize the robot controller
#         self.log("Initializing UR3 controller...")
#         try:
#             # Create controller instance with the configuration
#             self.controller = URTDEController(self.config, task=self.config.task)
#             time.sleep(2)  # Wait for controller to initialize and connect
#             self.log("Controller initialized successfully")
#         except Exception as e:
#             # Log error and re-raise exception
#             self.log(f"Error initializing controller: {e}")
#             raise
        
#         # Initialize task evaluator for reward calculation
#         self.task = TwoStage(verbose=verbose)  # Task-specific evaluator
        
#         # Store camera indices for later use
#         self.camera_indices = camera_indices
#         self.cameras = []  # Will hold camera objects
#         self.init_cameras()  # Initialize camera connections
        
#         # Initialize OpenPI policy framework
#         self.log(f"Loading OpenPI model config: {model_name}")
#         try:
#             # Get model config
#             self.model_config = config.get_config(model_name)
            
#             # Download checkpoint if needed
#             if model_path.startswith('s3://'):
#                 self.log(f"Downloading checkpoint from: {model_path}")
#                 self.checkpoint_dir = download.maybe_download(model_path)
#                 self.log(f"Checkpoint downloaded to: {self.checkpoint_dir}")
#             else:
#                 self.checkpoint_dir = model_path
            
#             # Note: We don't create the policy here, will create it fresh for each inference
#             self.log("OpenPI policy framework initialized successfully")
#         except Exception as e:
#             # Log error and re-raise exception
#             self.log(f"Error initializing OpenPI policy framework: {e}")
#             raise
        
#         # Create CSV file name with timestamp for trajectory logging
#         self.csv_file_name = f"openpi_trajectory_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
#         self.csv_path = self.output_dir / self.csv_file_name  # Full path to CSV file
        
#         # Initialize state tracking variables
#         self.current_step = 0  # Step counter
#         self.total_reward = 0  # Accumulated reward
        
#         self.log("Initialization complete")  # Log successful initialization
    
#     def run_inference_with_openpi(self, prompt, images=None, state=None):
#         """Run inference using the OpenPI policy framework with camera images and robot state.
        
#         Args:
#             prompt: Natural language instruction for the model
#             images: List of RGB images [base_rgb, wrist1_rgb, wrist2_rgb]
#             state: Robot state dictionary with joint angles, gripper state, etc.
            
#         Returns:
#             Dictionary with inference results including actions
#         """
#         try:
#             # Create policy from stored model config and checkpoint directory
#             self.log("Creating policy for inference")
#             policy = policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
            
#             # Use real camera images and robot state if provided, otherwise use dummy example
#             if images is not None and len(images) >= 3 and state is not None:
#                 self.log("Using real camera images and robot state")
#                 example = {
#                     # Camera images
#                     "base_rgb": images[0],
#                     "wrist1_rgb": images[1],
#                     "wrist2_rgb": images[2],
                    
#                     # Robot state
#                     "joints": state["joints"],
#                     "gripper": state["gripper"],
#                     "ee_pos": state["ee_pos"],
#                     "ee_quat": state["ee_quat"],
                    
#                     # Prompt
#                     "prompt": prompt
#                 }
#             else:
#                 # Use dummy example as fallback
#                 self.log("Using dummy example for inference")
#                 example = ur3_policy.make_ur3_example()
#                 example["prompt"] = prompt
            
#             # Run inference
#             self.log("Running inference with OpenPI policy")
#             result = policy.infer(example)
            
#             # Log results
#             self.log(f"Actions shape: {result['actions'].shape}")
#             if self.verbose:
#                 self.log(f"Actions: {result['actions']}")
            
#             # Clean up
#             del policy
            
#             # Handle result dimensions (flatten if needed)
#             if len(result['actions'].shape) > 1 and result['actions'].shape[0] == 1:
#                 # Take first action from batch dimension
#                 result['actions'] = result['actions'][0]
                
#             return result
            
#         except Exception as e:
#             self.log(f"Error during OpenPI inference: {e}")
#             traceback.print_exc(file=self.log_file)
#             return {"actions": np.zeros(7)}

#     def log(self, message):
#         """Log message to both console and log file with timestamp.
        
#         Args:
#             message: Message to log
#         """
#         print(message)  # Print to console
#         if hasattr(self, 'log_file') and self.log_file:
#             # Add timestamp to log entries
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             self.log_file.write(f"[{timestamp}] {message}\n")  # Write to file
#             self.log_file.flush()  # Ensure it's written immediately

#     def init_cameras(self):
#         """Initialize connections to all cameras."""
#         self.cameras = []  # Clear existing camera list
#         for i, cam_idx in enumerate(self.camera_indices):
#             try:
#                 # Try to open camera with OpenCV
#                 cap = cv2.VideoCapture(cam_idx)
#                 if not cap.isOpened():
#                     # Log warning if camera couldn't be opened
#                     self.log(f"Warning: Failed to open camera {cam_idx}")
#                     cap = None
#                 else:
#                     # Set camera properties for consistent image size
#                     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width in pixels
#                     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height in pixels
#                     self.log(f"Camera {cam_idx} initialized")
#                 self.cameras.append(cap)  # Add camera to list (even if None)
#             except Exception as e:
#                 # Log error and add None placeholder
#                 self.log(f"Error initializing camera {cam_idx}: {e}")
#                 self.cameras.append(None)
    
#     def capture_images(self):
#         """Capture images from all cameras.
        
#         Returns:
#             List of images from base, wrist1, and wrist2 cameras
#         """
#         images = []  # List to store captured images
#         for i, cap in enumerate(self.cameras):
#             if cap is not None and cap.isOpened():
#                 try:
#                     # Try to capture a frame
#                     ret, frame = cap.read()
#                     if ret:
#                         # Convert BGR (OpenCV format) to RGB (model input format)
#                         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         images.append(rgb_frame)
                        
#                         # Save image to disk for logging
#                         img_path = self.output_dir / f"step_{self.current_step:04d}_camera_{i}.jpg"
#                         cv2.imwrite(str(img_path), frame)  # Save original BGR image
#                     else:
#                         # Use blank image if capture failed
#                         self.log(f"Warning: Failed to capture from camera {i}")
#                         images.append(np.zeros((480, 640, 3), dtype=np.uint8))
#                 except Exception as e:
#                     # Use blank image if exception occurred
#                     self.log(f"Error capturing from camera {i}: {e}")
#                     images.append(np.zeros((480, 640, 3), dtype=np.uint8))
#             else:
#                 # Use blank image if camera not available
#                 self.log(f"Warning: Camera {i} not available")
#                 images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
#         # Ensure we have exactly 3 images (base, wrist1, wrist2)
#         while len(images) < 3:
#             # Add blank images if needed
#             images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
#         return images[:3]  # Return only the first 3 images
    
#     def get_robot_state(self):
#         """Get current robot state from controller.
        
#         Returns:
#             Tuple of (state dict for model, raw observation dict)
#         """
#         try:
#             # Get state from robot controller
#             obs, _ = self.controller.get_state()
            
#             # Extract relevant information from observation
#             joint_positions = obs['joint_positions']  # Joint angles
#             ee_pos = obs['robot0_eef_pos']           # End effector position
#             ee_quat = obs['robot0_eef_quat']         # End effector orientation
#             # Get gripper position with default if not available
#             gripper_qpos = obs.get('robot0_gripper_qpos', 0.5)
            
#             # Log state if verbose mode is enabled
#             if self.verbose:
#                 self.log(f"Robot state: joints={joint_positions}, pos={ee_pos}, orient={ee_quat}, gripper={gripper_qpos}")
            
#             # Save state to JSON file for logging
#             state_path = self.output_dir / f"step_{self.current_step:04d}_state.json"
#             with open(state_path, 'w') as f:
#                 # Convert numpy arrays to lists for JSON serialization
#                 state_json = {
#                     "joint_positions": joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else joint_positions,
#                     "ee_pos": ee_pos.tolist() if isinstance(ee_pos, np.ndarray) else ee_pos,
#                     "ee_quat": ee_quat.tolist() if isinstance(ee_quat, np.ndarray) else ee_quat,
#                     "gripper_qpos": float(gripper_qpos)
#                 }
#                 json.dump(state_json, f, indent=2)  # Write with nice formatting
            
#             # Create a dictionary with information formatted for the OpenPI model
#             state = {
#                 "joints": np.array(joint_positions[:6], dtype=np.float32),  # First 6 joints
#                 "gripper": np.array([gripper_qpos], dtype=np.float32),      # Gripper position
#                 "ee_pos": np.array(ee_pos, dtype=np.float32),               # End effector position
#                 "ee_quat": np.array(ee_quat, dtype=np.float32)              # End effector orientation
#             }
            
#             # Record state in CSV format for trajectory logging
#             dt = datetime.datetime.now()  # Current timestamp
#             ee_pos_quat = np.concatenate((ee_pos, ee_quat))  # Combine position and orientation
#             save_frame(self.output_dir, dt, joint_positions, ee_pos_quat, self.csv_file_name)
            
#             return state, obs
            
#         except Exception as e:
#             # Log error and return default values
#             self.log(f"Error getting robot state: {e}")
#             return {
#                 "joints": np.zeros(6, dtype=np.float32),
#                 "gripper": np.array([0.5], dtype=np.float32),
#                 "ee_pos": np.zeros(3, dtype=np.float32),
#                 "ee_quat": np.array([0, 0, 0, 1], dtype=np.float32)
#             }, {}
    
#     def execute_action(self, action):
#         """Execute action on the robot.
        
#         Args:
#             action: 7D action vector [dx, dy, dz, drx, dry, drz, gripper]
            
#         Returns:
#             Boolean indicating success
#         """
#         try:
#             # Log the raw action
#             self.log(f"Executing action: {action}")
            
#             # Copy action to avoid modifying the original
#             controller_action = action.copy()
            
#             # Scale position components if needed
#             position_scale = 1.0  # Scaling factor for position deltas
#             controller_action[:3] *= position_scale
            
#             # Scale rotation components (orientation deltas)
#             rotation_scale = 0.1  # Scaling factor for rotation deltas (smaller for safety)
#             controller_action[3:6] *= rotation_scale
            
#             # Log the scaled action
#             self.log(f"Scaled action: {controller_action}")
            
#             # Send action to robot controller
#             self.controller.update(np.array(controller_action))
            
#             # Save action to JSON file for logging
#             action_path = self.output_dir / f"step_{self.current_step:04d}_action.json"
#             with open(action_path, 'w') as f:
#                 action_json = {
#                     "raw_action": action.tolist(),
#                     "scaled_action": controller_action.tolist()
#                 }
#                 json.dump(action_json, f, indent=2)  # Write with nice formatting
            
#             # Wait for execution to complete
#             time.sleep(0.5)  # Adjust based on robot's speed
            
#             return True  # Action executed successfully
            
#         except Exception as e:
#             # Log error and return failure
#             self.log(f"Error executing action: {e}")
#             return False
        
#     def run_task(self, prompt, max_steps=50):
#         """Run a task using OpenPI policy framework for control.
        
#         Args:
#             prompt: Natural language instruction for the model
#             max_steps: Maximum number of steps to run
#         """
#         # Log task start and reset counters
#         self.log(f"Starting task: '{prompt}'")
#         self.current_step = 0
#         self.total_reward = 0
        
#         # Save prompt to file for reference
#         with open(self.output_dir / "prompt.txt", "w") as f:
#             f.write(prompt)
        
#         try:
#             # Main task execution loop
#             while self.current_step < max_steps:
#                 # Log current step
#                 self.log(f"\nStep {self.current_step}/{max_steps}")
                
#                 # Get images from cameras
#                 images = self.capture_images()
                
#                 # Get current robot state
#                 state, raw_obs = self.get_robot_state()
                
#                 # Run inference using our class method
#                 result = self.run_inference_with_openpi(prompt, images, state)
#                 action = result["actions"]
                
#                 # Execute the predicted action
#                 success = self.execute_action(action)
#                 if not success:
#                     # Stop if action execution failed
#                     self.log("Failed to execute action")
#                     break
                
#                 # Check reward/success based on task
#                 try:
#                     # Calculate reward from current state
#                     reward = self.task.reward(raw_obs)
#                     self.total_reward += reward  # Accumulate reward
#                     self.log(f"Step reward: {reward}, Total reward: {self.total_reward}")
                    
#                     # Check if task is complete (reward >= 1.0 indicates success)
#                     if reward >= 1.0:
#                         self.log("Task completed successfully!")
#                         break
#                 except Exception as e:
#                     # Log error if reward calculation fails
#                     self.log(f"Error computing reward: {e}")
                
#                 # Increment step counter
#                 self.current_step += 1
                
#                 # Check for external stop signal (file-based)
#                 if os.path.exists("stop_execution.txt"):
#                     self.log("Stop file detected, stopping execution")
#                     os.remove("stop_execution.txt")  # Remove the file
#                     break
            
#             # Log task completion
#             self.log(f"Task completed. Total steps: {self.current_step}, Total reward: {self.total_reward}")
            
#         except KeyboardInterrupt:
#             # Handle user interruption
#             self.log("Task interrupted by user")
#         except Exception as e:
#             # Handle other exceptions
#             self.log(f"Error during task execution: {e}")
#             traceback.print_exc(file=self.log_file)
#         finally:
#             # Always return to home position for safety
#             try:
#                 self.log("Returning to home position...")
#                 # Create a neutral action that moves the robot home
#                 home_action = np.zeros(7)  # Zero movement
#                 home_action[-1] = 0.5      # Neutral gripper position
#                 self.controller.update(home_action)
#             except Exception as e:
#                 self.log(f"Error returning to home position: {e}")

#     def cleanup(self):
#         """Clean up resources before exiting."""
#         self.log("Cleaning up...")
        
#         # Release all camera resources
#         for cap in self.cameras:
#             if cap is not None:
#                 cap.release()
        
#         # Close log file
#         if hasattr(self, 'log_file') and self.log_file:
#             self.log_file.close()
        
#         # Close any open OpenCV windows
#         cv2.destroyAllWindows()
        
#         self.log("Cleanup complete")

# def main():
#     """Main function to parse arguments and run the controller."""
#     # Set up command line argument parser
#     parser = argparse.ArgumentParser(description="Run OpenPI inference on UR3 robot")
#     parser.add_argument("--model_path", type=str, required=True, 
#                         help="Path to the OpenPI model checkpoint")
#     parser.add_argument("--prompt", type=str, default="Pick up the object and place it in the target location", 
#                         help="Task prompt for the OpenPI model")
#     parser.add_argument("--steps", type=int, default=50, 
#                         help="Maximum number of steps for the task")
#     parser.add_argument("--base_camera", type=int, default=0, 
#                         help="Camera ID for base camera")
#     parser.add_argument("--wrist1_camera", type=int, default=1, 
#                         help="Camera ID for wrist1 camera")
#     parser.add_argument("--wrist2_camera", type=int, default=2, 
#                         help="Camera ID for wrist2 camera")
#     parser.add_argument("--robot_ip", type=str, default="192.168.77.22", 
#                         help="IP address of the UR3 robot")
#     parser.add_argument("--hostname", type=str, default="192.168.77.134", 
#                         help="Hostname for the controller")
#     parser.add_argument("--port", type=int, default=50002, 
#                         help="Port for the controller")
#     parser.add_argument("--mock", action="store_true",
#                         help="Run in mock mode without real robot")
#     parser.add_argument("--verbose", action="store_true", 
#                         help="Enable verbose logging")
#     args = parser.parse_args()  # Parse command line arguments
    
#     # Create robot controller configuration
#     config = URTDEControllerConfig(
#         task="two_stage",                   # Task type
#         controller_type="CARTESIAN_DELTA",  # Control mode
#         max_delta=0.06,                     # Maximum movement per step
#         mock=1 if args.mock else 0,         # Mock mode if requested
#         hostname=args.hostname,             # Server hostname
#         robot_port=args.port,               # Server port
#         robot_ip=args.robot_ip,             # Robot IP
#         hz=100                              # Control frequency
#     )
    
#     # Create and run controller with error handling
#     controller = None
#     try:
#         # Initialize controller with model and hardware
#         controller = OpenPIUR3Controller(
#             model_path=args.model_path,
#             config=config,
#             camera_indices=(args.base_camera, args.wrist1_camera, args.wrist2_camera),
#             verbose=args.verbose
#         )
        
#         # Run the task with the given prompt
#         controller.run_task(args.prompt, max_steps=args.steps)
#     except Exception as e:
#         # Handle any uncaught exceptions
#         print(f"Error: {e}")
#         traceback.print_exc()
#     finally:
#         # Ensure cleanup happens even if errors occur
#         if controller is not None:
#             controller.cleanup()


# # Standard Python idiom to call the main function when the script is run directly
# if __name__ == "__main__":
#     main()




import os
import sys
import numpy as np
import cv2
import logging
import pathlib
import time
from typing import Dict, List, Optional, Tuple, Any

# We'll need to make sure Isaac Sim is in the path
try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    # # Standard Isaac Sim paths - adjust as needed for your installation
    # isaac_sim_path = os.environ.get("ISAAC_SIM_PATH", "/home/shreya/isaacsim")
    # if not os.path.exists(isaac_sim_path):
    #     raise ImportError(f"Isaac Sim not found at {isaac_sim_path}. Please set ISAAC_SIM_PATH.")
    # os.environ["CARB_APP_PATH"] = f"{isaac_sim_path}/kit/apps/omni.isaac.sim.pytorch.kit"
    # sys.path.append(isaac_sim_path)
    # from omni.isaac.kit import SimulationApp

# Import Isaac Sim modules after SimulationApp is created
HEADLESS = False  # Set to True for headless mode


class UR3eIsaacEnv:
    """UR3e robot environment in Isaac Sim."""
    
    def __init__(
        self,
        task: str = "pick_and_place",
        seed: int = 0,
        headless: bool = False,
        physics_dt: float = 1/60.0,
        rendering_dt: float = 1/60.0,
        use_base_camera: bool = True,
        use_wrist_camera: bool = True
    ):
        """Initialize the UR3e environment in Isaac Sim.
        
        Args:
            task: Task to perform ("pick_and_place", "stacking", or "pushing")
            seed: Random seed
            headless: Whether to run in headless mode
            physics_dt: Physics simulation timestep
            rendering_dt: Rendering timestep
            use_base_camera: Whether to use the base camera
            use_wrist_camera: Whether to use the wrist camera
        """
        self.task = task
        self.seed = seed
        self.headless = headless
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.use_base_camera = use_base_camera
        self.use_wrist_camera = use_wrist_camera
        
        # Initialize Isaac Sim
        self._init_isaac_sim()
        
        # Set up the scene
        self._setup_scene()
        
        # Set up cameras
        self._setup_cameras()
        
        # Reset to initial state
        self.reset()
        
        logging.info("UR3e environment initialized successfully")
    
    def _init_isaac_sim(self):
        """Initialize Isaac Sim."""
        # Initialize the simulation app
        self.simulation_app = SimulationApp({
            "headless": self.headless,
            "width": 1280,
            "height": 720
        })
        
        # Now we can import the modules that require SimulationApp to be created
        import omni.isaac.core.utils.nucleus as nucleus_utils
        import omni.isaac.core.utils.stage as stage_utils
        import omni.isaac.core.utils.prims as prims_utils
        import omni.isaac.core.utils.rotations as rotations_utils
        import omni.isaac.core.objects as objects
        import omni.isaac.core.articulations as articulations
        from omni.isaac.core.robots.robot_view import RobotView
        from omni.isaac.core.utils.viewports import set_camera_view
        
        self.nucleus_utils = nucleus_utils
        self.stage_utils = stage_utils
        self.prims_utils = prims_utils
        self.rotations_utils = rotations_utils
        self.objects = objects
        self.articulations = articulations
        self.RobotView = RobotView
        self.set_camera_view = set_camera_view
        
        # Initialize physics
        from omni.isaac.core.world import World
        self.world = World(physics_dt=self.physics_dt, rendering_dt=self.rendering_dt)
        
        # Other imports
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.extensions import disable_extension
        
        # Enable required extensions
        enable_extension("omni.isaac.universal_robots")
        enable_extension("omni.isaac.motion_generation")
    
    def _setup_scene(self):
        """Set up the UR3e scene."""
        # Create a new stage
        self.stage_utils.create_new_stage()
        
        # Set up lighting
        from pxr import UsdLux
        from omni.isaac.core.utils.stage import add_reference_to_stage
        
        # Add a distant light
        light_prim = UsdLux.DistantLight.Define(self.stage_utils.get_current_stage(), "/World/Light")
        light_prim.GetIntensityAttr().Set(2000.0)
        light_prim.GetAngleAttr().Set(0.53)
        
        # Set up ground plane
        self.stage_utils.add_ground_plane(
            "/World/ground_plane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8
        )
        
        robot_usd_path = "/home/shreya/Desktop/PI/openpi/examples/ur_env/ur_sim/assets/ur3e.usd"  # Path to your custom USD file
        
        # Check if the file exists
        if not os.path.exists(robot_usd_path):
            raise FileNotFoundError(f"Custom UR3e USD file not found at: {robot_usd_path}")
            
        # Add reference to stage using the full path
        from pxr import Sdf
        ref_path = Sdf.AssetPath(robot_usd_path)
        add_reference_to_stage(ref_path, "/World/ur3e")
        

        self.world.scene.add(self.robot_view)
    
    def _setup_task_scene(self):
        """Set up the scene based on the task."""
        # Add a table
        table_usd_path = self.nucleus_utils.get_assets_root_path() + "/Isaac/Props/Tables/cafe_table.usd" 
        self.prims_utils.create_prim(
            "/World/table",
            "Xform",
            position=np.array([0.5, 0.0, 0.0]),
            scale=np.array([1.0, 1.0, 0.8])
        )
        self.stage_utils.add_reference_to_stage(table_usd_path, "/World/table")
        
        # Add task-specific objects
        if self.task == "pick_and_place":
            # Add a cube for picking
            cube_usd_path = self.nucleus_utils.get_assets_root_path() + "/Isaac/Props/Blocks/cube_red.usd"
            self.stage_utils.add_reference_to_stage(cube_usd_path, "/World/cube")
            self.prims_utils.set_prim_transform(
                "/World/cube", 
                translation=np.array([0.5, 0.0, 0.7]), # On the table
                scale=np.array([0.04, 0.04, 0.04])
            )
            
            # Add a target location
            target_usd_path = self.nucleus_utils.get_assets_root_path() + "/Isaac/Props/UI/target.usd"
            self.stage_utils.add_reference_to_stage(target_usd_path, "/World/target")
            self.prims_utils.set_prim_transform(
                "/World/target", 
                translation=np.array([0.5, 0.3, 0.7]), # On the table, to the right
                scale=np.array([0.08, 0.08, 0.001])
            )
            
        elif self.task == "stacking":
            # Add several cubes for stacking
            colors = ["red", "blue", "green"]
            positions = [
                np.array([0.5, -0.2, 0.7]), 
                np.array([0.5, 0.0, 0.7]),
                np.array([0.5, 0.2, 0.7])
            ]
            
            for i, (color, pos) in enumerate(zip(colors, positions)):
                cube_usd_path = self.nucleus_utils.get_assets_root_path() + f"/Isaac/Props/Blocks/cube_{color}.usd"
                self.stage_utils.add_reference_to_stage(cube_usd_path, f"/World/cube_{i}")
                self.prims_utils.set_prim_transform(
                    f"/World/cube_{i}", 
                    translation=pos,
                    scale=np.array([0.04, 0.04, 0.04])
                )
                
        elif self.task == "pushing":
            # Add a ball for pushing
            ball_usd_path = self.nucleus_utils.get_assets_root_path() + "/Isaac/Props/Balls/ball_medium.usd"
            self.stage_utils.add_reference_to_stage(ball_usd_path, "/World/ball")
            self.prims_utils.set_prim_transform(
                "/World/ball", 
                translation=np.array([0.5, 0.0, 0.7]), # On the table
                scale=np.array([0.04, 0.04, 0.04])
            )
            
            # Add a target area
            target_usd_path = self.nucleus_utils.get_assets_root_path() + "/Isaac/Props/UI/target.usd"
            self.stage_utils.add_reference_to_stage(target_usd_path, "/World/target")
            self.prims_utils.set_prim_transform(
                "/World/target", 
                translation=np.array([0.7, 0.0, 0.7]), # On the table, further away
                scale=np.array([0.12, 0.12, 0.001])
            )
    
    def _setup_cameras(self):
        """Set up the cameras for the environment."""
        from pxr import UsdGeom, Gf
        from omni.isaac.core.objects import Camera
        
        # Set up base camera
        if self.use_base_camera:
            # Create the base camera
            self.base_cam = Camera(
                prim_path="/World/base_camera",
                position=np.array([0.8, -0.8, 1.2]),
                focal_length=24.0,
                resolution=(224, 224)
            )
            self.base_cam.set_target(np.array([0.5, 0.0, 0.7]))  # Look at the table
            self.world.scene.add(self.base_cam)
        
        # Set up wrist camera
        if self.use_wrist_camera:
            # Create the wrist camera - attached to the robot's end effector
            from omni.isaac.core.prims import XFormPrim
            
            # Create a mount point on the robot's end effector
            wrist_mount = XFormPrim(
                prim_path="/World/ur3e/base_link/shoulder_link/upper_arm_link/forearm_link/wrist_1_link/wrist_2_link/wrist_3_link/camera_mount",
                position=np.array([0.0, 0.0, 0.05]),  # Offset from wrist
                orientation=self.rotations_utils.euler_angles_to_quat(np.array([0, -np.pi/2, 0])) # Facing forward
            )
            self.world.scene.add(wrist_mount)
            
            # Add the camera to the mount
            self.wrist_cam = Camera(
                prim_path="/World/ur3e/base_link/shoulder_link/upper_arm_link/forearm_link/wrist_1_link/wrist_2_link/wrist_3_link/camera_mount/wrist_camera",
                position=np.array([0.0, 0.0, 0.0]),  # Relative to mount
                focal_length=12.0,
                resolution=(224, 224)
            )
            self.world.scene.add(self.wrist_cam)
    
    def reset(self):
        """Reset the environment."""
        # Reset simulation
        self.world.reset()
        
        # Set robot to home position
        home_joints = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.0])  # Example home position
        self.ur3e.set_joint_positions(home_joints)
        
        # Step simulation a few times to stabilize
        for _ in range(10):
            self.world.step(render=True)
        
        # Return the initial observation
        return self.get_observation()
    
    def get_observation(self):
        """Get the current observation from the environment."""
        # Get robot state
        joint_positions = self.ur3e.get_joint_positions()
        joint_velocities = self.ur3e.get_joint_velocities()
        
        # Create a dict for the observation (matching the policy's expected format)
        observation = {
            "joints": joint_positions[:6].tolist(),  # 6 joint positions
            "gripper": 0.0,  # No gripper in this simple example
            "prompt": "pick",  # Simple prompt to avoid tokenization issues
        }
        
        # Get camera images
        if self.use_base_camera:
            base_rgb = self.base_cam.get_rgba()[:, :, :3]  # Remove alpha channel
            observation["base_rgb"] = base_rgb
        
        if self.use_wrist_camera:
            wrist_rgb = self.wrist_cam.get_rgba()[:, :, :3]  # Remove alpha channel
            observation["wrist_rgb"] = wrist_rgb
        
        return observation
    
    def step(self, action):
        """Take a step in the environment with the given action.
        
        Args:
            action: Array of shape (7,) containing joint positions and gripper
        
        Returns:
            observation: The new observation after taking the action
            reward: The reward for the current state
            done: Whether the episode is done
            info: Additional information
        """
        # Unpack action
        joint_positions = action[:6]
        gripper_position = action[6]
        
        # Send joint positions to robot
        self.ur3e.set_joint_positions(joint_positions)
        
        # Step the simulation
        self.world.step(render=True)
        
        # Get new observation
        observation = self.get_observation()
        
        # Check if task is complete (simplified)
        done = False
        reward = 0.0
        
        # Return step information
        return observation, reward, done, {}
    
    def get_joint_positions(self):
        """Get the current joint positions of the robot."""
        return self.ur3e.get_joint_positions()
    
    def close(self):
        """Close the environment and clean up resources."""
        # Close the simulation app
        if hasattr(self, 'simulation_app'):
            self.simulation_app.close()


class VideoSaver:
    """Class to save videos of the simulation."""
    
    def __init__(self, output_dir: pathlib.Path, fps: int = 30):
        """Initialize the video saver.
        
        Args:
            output_dir: Directory to save videos
            fps: Frames per second for the video
        """
        self.output_dir = output_dir
        self.fps = fps
        self.frames = []
        self.started_saving = False
        
        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_reset(self, env, agent):
        """Called when the environment is reset."""
        if self.started_saving:
            self._save_video()
        self.frames = []
        self.started_saving = True
    
    def on_observation(self, env, agent):
        """Called when a new observation is available."""
        # Get the base camera image
        if hasattr(env, 'base_cam'):
            frame = env.base_cam.get_rgba()[:, :, :3]
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8
            self.frames.append(frame)
    
    def on_step(self, env, agent, action):
        """Called after each step."""
        pass
    
    def _save_video(self):
        """Save the collected frames as a video."""
        if not self.frames:
            return
        
        # Create a unique filename with timestamp
        timestamp = int(time.time())
        filename = self.output_dir / f"ur3e_sim_{timestamp}.mp4"
        
        # Save the video using OpenCV
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filename), fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            # Convert from RGB to BGR (OpenCV uses BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        logging.info(f"Saved video to {filename}")
        self.frames = []
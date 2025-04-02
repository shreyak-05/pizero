from dataclasses import dataclass
import numpy as np
import torch
import cv2
import numpy as np
import random

# class MockRobot:
#     @dataclass
#     class State:
#         joint_positions = np.zeros(7)
#         joint_velocities = np.zeros(7)

#     def __init__(self):
#         self.policy_running = False
#         self.ee_pos = torch.zeros(3)
#         self.ee_quat = torch.rand(4)
#         self.ee_quat /= (self.ee_quat**2).sum() ** 0.5
#         # Add joint positions storage
#         self._joint_positions = np.zeros(6)  # 6 joints for UR3

#     def set_home_pose(self, home_pose):
#         print(f"[mock robot]: set home_pose {home_pose}")

#     def update_desired_ee_pose(self, new_pos: torch.Tensor, new_quat: torch.Tensor):
#         assert self.policy_running

#         assert new_pos.shape == self.ee_pos.shape
#         assert new_quat.shape == self.ee_quat.shape

#         print(f"[mock]: robot update pos: {new_pos}, quat: {new_quat}")
#         self.ee_pos = new_pos
#         self.ee_quat = new_quat

#     def go_home(self, blocking):
#         print(f"[mock robot]: go home")

#         self.ee_pos = torch.zeros(3)
#         self.ee_quat = torch.rand(4)
#         self.ee_quat /= (self.ee_quat**2).sum() ** 0.5
#         self._joint_positions = np.zeros(6)  # Reset joint positions

#     def get_robot_state(self):
#         return MockRobot.State()

#     def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
#         return self.ee_pos, self.ee_quat

#     def is_running_policy(self):
#         return self.policy_running

#     def terminate_current_policy(self):
#         self.policy_running = False

#     def start_cartesian_impedance(self):
#         assert not self.is_running_policy()
#         self.policy_running = True

#     def get_joint_state(self):
#         """Return mock joint positions for compatibility with controller."""
#         # Return exactly 6 joint positions (UR3 has 6 joints)
#         return self._joint_positions  # Return the stored joint positions
        
#     def command_joint_state(self, joint_state):
#         """Mock method to receive joint position commands."""
#         print(f"[mock]: Received command_joint_state: {joint_state}")
#         # Store the commanded joint state
#         self._joint_positions = np.array(joint_state[:6])  # Ensure we store exactly 6 values
        
#     # Add these additional required methods
    
#     def get_joint_positions(self):
#         """Alternative method to get joint positions."""
#         return self._joint_positions
        
#     def move_joints(self, positions, *args, **kwargs):
#         """Move robot joints to specified positions."""
#         print(f"[mock]: move_joints {positions}")
#         # Simulate gradual movement
#         self._joint_positions = 0.8 * self._joint_positions + 0.2 * np.array(positions[:6])
#         return True
        
#     def reset(self):
#         """Reset robot state."""
#         print("[mock]: reset robot")
#         self.go_home(blocking=False)
#         return True
        
#     def get_ee_state(self):
#         """Get end effector state - alias for get_ee_pose."""
#         return self.get_ee_pose()
    
# Add these imports if not already present

class MockURTDEController:
    """Mock URTDE controller that returns consistent state data."""
    
    def __init__(self, config=None, task=None):
        """Initialize the mock controller.
        
        Args:
            config: Configuration object
            task: Task name
        """
        self.use_gripper = True
        self.desired_gripper_qpos = 0.5
        self._joint_positions = np.zeros(7)  # 7 values including gripper
        self._robot = MockRobot()  # Use the existing MockRobot class
        
        # Initialize the robot in a default position
        self._ee_pos = np.zeros(3)
        self._ee_quat = np.array([0, 0, 0, 1])
        
        print("[mock]: Created mock URTDE controller")
    
    def get_state(self):
        """Get current robot state.
        
        Returns:
            Tuple of (observations, in_good_range) where:
            - observations is a dict of robot state
            - in_good_range is a scalar boolean (not an array)
        """
        # Use the MockRobot's get_observations method
        obs = self._robot.get_observations()
        
        # Return scalar boolean for in_good_range, not an array
        in_good_range = True
        
        print(f"[mock]: get_state returned obs and {bool(in_good_range)}")
        
        return obs, in_good_range
    
    def move_to_eef_positions(self, positions, delta=False):
        """Mock moving end-effector to positions.
        
        Args:
            positions: Position and orientation [x, y, z, qx, qy, qz, qw]
            delta: Whether the positions are relative or absolute
            
        Returns:
            True for success
        """
        print(f"[mock]: move_to_eef_positions called with delta={delta}")
        
        # Extract position and orientation
        if len(positions) >= 7:
            pos = positions[:3]
            quat = positions[3:7]
        else:
            # Handle shorter positions arrays
            pos = positions[:min(3, len(positions))]
            quat = self._ee_quat  # Keep current orientation
        
        # Update internal state
        if delta:
            self._ee_pos = self._ee_pos + pos
            # Note: Quaternion addition isn't meaningful, but we're just mocking
            self._ee_quat = quat
        else:
            self._ee_pos = pos
            self._ee_quat = quat
            
        # Update robot's EE pose
        self._robot.command_eef_pose((self._ee_pos, self._ee_quat))
        
        return True
    
    def update_gripper(self, position, blocking=False):
        """Mock updating gripper position.
        
        Args:
            position: Gripper position [0-1]
            blocking: Whether to block until done
            
        Returns:
            True for success
        """
        print(f"[mock]: update_gripper called with position={position}")
        self.desired_gripper_qpos = float(position)  # Ensure it's a scalar
        self._robot._gripper_pos = np.array([self.desired_gripper_qpos])
        return True
    
    def update(self, action):
        """Mock update with an action vector.
        
        Args:
            action: Action vector [dx, dy, dz, drx, dry, drz, gripper]
            
        Returns:
            True for success
        """
        print(f"[mock]: Received action: {action}")
        
        # Convert action to list to ensure it's not a numpy array
        if isinstance(action, np.ndarray):
            action = action.tolist()
        
        # Extract position, orientation, and gripper
        if len(action) >= 7:
            pos_delta = action[:3]
            rot_delta = action[3:6]
            gripper = action[6]
        else:
            # Handle incomplete actions
            pos_delta = action[:min(3, len(action))]
            rot_delta = [0, 0, 0]
            gripper = 0.5
        
        # Update end effector position
        self.move_to_eef_positions(np.concatenate([pos_delta, rot_delta, [0]]), delta=True)
        
        # Update gripper
        self.update_gripper(gripper)
        
        return True
    
    def reset(self, randomize=False):
        """Mock reset the environment.
        
        Args:
            randomize: Whether to randomize the initial state
            
        Returns:
            True for success
        """
        print(f"[mock]: reset env")
        self._robot.reset()
        return True
    
    def get_ee_pose(self):
        """Mock get end-effector pose.
        
        Returns:
            array: Combined position and orientation [x, y, z, qx, qy, qz, qw]
        """
        pos, quat = self._robot.get_ee_pose()
        return np.concatenate([pos, quat])
    
    def command_joint_state(self, joint_positions):
        """Mock command joint positions.
        
        Args:
            joint_positions: Joint positions
            
        Returns:
            True for success
        """
        print(f"[mock]: Received command_joint_state: {joint_positions}")
        self._robot.command_joint_state(joint_positions)
        return True
    
    def move_to_joint_positions(self, joint_positions, steps=100):
        """Move to joint positions over a number of steps.
        
        Args:
            joint_positions: Target joint positions
            steps: Number of steps to take
            
        Returns:
            True for success
        """
        print(f"move_to_joint_positions")
        print(f"Steps to reach home pose {steps}")
        
        # Just call command_joint_state directly
        self.command_joint_state(joint_positions)
        return True
    
    def get_joint_positions(self):
        """Get the current joint positions.
        
        Returns:
            array: Joint positions
        """
        return self._robot.get_joint_positions()
    
    def get_gripper_state(self):
        """Get the gripper state.
        
        Returns:
            float: Gripper position [0-1]
        """

        return self._robot._get_gripper_pos()
    

class MockPolicy:
    """Mock policy for testing without a real model."""
    
    def __init__(self):
        """Initialize mock policy."""
        print("[mock]: Creating mock policy")
    
    def infer(self, example):
        """Return mock prediction results.
        
        Args:
            example: Dictionary containing model inputs
            
        Returns:
            Dictionary with mock actions
        """
        print("[mock]: Running mock inference")
        
        # Generate a small random action to simulate movement
        # Small values for safety
        pos_delta = np.random.uniform(-0.01, 0.01, size=3)  # x, y, z position
        rot_delta = np.random.uniform(-0.005, 0.005, size=3)  # rotation
        gripper = np.random.uniform(0.0, 1.0, size=1)  # gripper
        
        # Combine into 7D action vector
        actions = np.concatenate([pos_delta, rot_delta, gripper])
        
        return {"actions": actions}
    
    def _input_transform(self, example):
        """Mock input transformation function."""
        # Simply return the example
        return example
    
    def _sample_actions(self, *args, **kwargs):
        """Mock action sampling function."""
        # Return a mock batch of actions
        return np.zeros((1, 7))

class MockCamera:
    """Mock camera for testing without real cameras."""
    
    def __init__(self, camera_id, img_size=(224, 224)):
        """Initialize mock camera.
        
        Args:
            camera_id: Camera identifier (0=base, 1=wrist1, 2=wrist2)
            img_size: Image size (width, height)
        """
        self.camera_id = camera_id
        self.img_size = img_size
        self.is_open = True
        print(f"[mock]: Created mock camera {camera_id}")
        
        # Create distinctive mock images for each camera
        if camera_id == 0:  # Base camera - mostly red
            self.image = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            self.image[:, :, 0] = 200  # Red channel
            # Add a "button" in the center
            cv2.circle(self.image, (img_size[0]//2, img_size[1]//2), 30, (0, 0, 255), -1)
        elif camera_id == 1:  # Wrist1 camera - mostly green
            self.image = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            self.image[:, :, 1] = 200  # Green channel
            # Add some features
            cv2.rectangle(self.image, (50, 50), (150, 150), (255, 255, 255), 2)
        else:  # Wrist2 camera - mostly blue
            self.image = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            self.image[:, :, 2] = 200  # Blue channel
            # Add some features
            cv2.line(self.image, (0, 0), (img_size[0], img_size[1]), (255, 255, 255), 2)
    
    def isOpened(self):
        """Check if camera is opened."""
        return self.is_open
    
    def read(self):
        """Read a frame from the mock camera."""
        # Add some slight random noise to make each frame different
        noise = np.random.randint(-10, 10, self.image.shape, dtype=np.int16)
        noisy_image = np.clip(self.image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return True, noisy_image
    
    def release(self):
        """Release the mock camera."""

class MockRobot:
    @dataclass
    class State:
        joint_positions = np.zeros(7)
        joint_velocities = np.zeros(7)

    def __init__(self):
        self.policy_running = False
        self.ee_pos = torch.zeros(3)
        self.ee_quat = torch.rand(4)
        self.ee_quat /= (self.ee_quat**2).sum() ** 0.5
        # Add joint positions storage with correct length
        self._joint_positions = np.zeros(7)  # 7 values including gripper
        self._home_pose_length = 7  # Store expected home pose length
        self._gripper_pos = np.array([0.5])  # Default gripper position

    def set_home_pose(self, home_pose):
        print(f"[mock robot]: set home_pose {home_pose}")
        # Remember the length of home_pose to match it later
        self._home_pose_length = len(list(home_pose))
        # Also update _joint_positions to have the same length
        self._joint_positions = np.zeros(self._home_pose_length)

    def update_desired_ee_pose(self, new_pos: torch.Tensor, new_quat: torch.Tensor):
        assert self.policy_running

        assert new_pos.shape == self.ee_pos.shape
        assert new_quat.shape == self.ee_quat.shape

        print(f"[mock]: robot update pos: {new_pos}, quat: {new_quat}")
        self.ee_pos = new_pos
        self.ee_quat = new_quat

    def go_home(self, blocking):
        print(f"[mock robot]: go home")

        self.ee_pos = torch.zeros(3)
        self.ee_quat = torch.rand(4)
        self.ee_quat /= (self.ee_quat**2).sum() ** 0.5
        # Reset joint positions with correct length
        self._joint_positions = np.zeros(self._home_pose_length)

    def get_robot_state(self):
        return MockRobot.State()
    
    #     # Add to MockGripper class
    # def get_current_position(self):
    #     """Get the current gripper position."""
    #     return self.width

    def get_ee_pose(self):
        """Get the end effector position and orientation.
        
        Returns:
            Tuple of (position, orientation)
        """
        # Convert torch tensors to numpy arrays for compatibility
        if isinstance(self.ee_pos, torch.Tensor):
            pos = self.ee_pos.detach().cpu().numpy()
        else:
            pos = self.ee_pos
            
        if isinstance(self.ee_quat, torch.Tensor):
            quat = self.ee_quat.detach().cpu().numpy()
        else:
            quat = self.ee_quat
            
        return pos, quat
    def is_running_policy(self):
        """Check if the policy is running."""
        # Make sure we return a scalar boolean, not an array
        return bool(self.policy_running)

    def terminate_current_policy(self):
        self.policy_running = False

    def start_cartesian_impedance(self):
        assert not self.is_running_policy()
        self.policy_running = True

    def get_joint_state(self):
        """Return mock joint positions for compatibility with controller."""
        # This is what's checked in the assert statement
        return self._joint_positions

    def command_joint_state(self, joint_state):
        """Command the robot to move to the specified joint state.
        
        Args:
            joint_state: Array of joint angles
            
        Returns:
            True for success
        """
        print(f"[mock]: Received command_joint_state: {joint_state}")
        # Ensure joint_state and _joint_positions have the same length
        if len(joint_state) != len(self._joint_positions):
            # Resize _joint_positions to match
            self._joint_positions = np.zeros(len(joint_state))
        self._joint_positions = np.array(joint_state)
        return True

    def command_eef_pose(self, eef_pos):
        """Command the robot to move to the specified end effector pose.
        
        Args:
            eef_pos: End effector position and orientation
            
        Returns:
            True for success
        """
        print(f"[mock]: Received command_eef_pose: {eef_pos}")
        # Update the robot's end effector pose
        if isinstance(eef_pos, tuple) and len(eef_pos) == 2:
            pos, quat = eef_pos
            self.ee_pos = torch.tensor(pos) if not isinstance(pos, torch.Tensor) else pos
            self.ee_quat = torch.tensor(quat) if not isinstance(quat, torch.Tensor) else quat
        else:
            # If it's just a position, keep the current orientation
            self.ee_pos = torch.tensor(eef_pos) if not isinstance(eef_pos, torch.Tensor) else eef_pos
        return True

    def get_joint_positions(self):
        """Alternative method to get joint positions."""
        return self._joint_positions

    def move_joints(self, positions, *args, **kwargs):
        """Move robot joints to specified positions."""
        print(f"[mock]: move_joints {positions}")
        # Ensure _joint_positions and positions have the same length
        if len(positions) != len(self._joint_positions):
            # Resize _joint_positions to match
            self._joint_positions = np.zeros(len(positions))
        # Simulate gradual movement
        self._joint_positions = 0.8 * self._joint_positions + 0.2 * np.array(positions)
        return True

    def reset(self):
        """Reset robot state."""
        print("[mock]: reset robot")
        self.go_home(blocking=False)
        return True

    def get_ee_state(self):
        """Get end effector state - alias for get_ee_pose."""
        return self.get_ee_pose()
    
    def num_dofs(self):
        """Get the number of joints in the robot.
        
        Returns:
            int: The number of joints in the robot.
        """
        return len(self._joint_positions)
    
    def _get_gripper_pos(self):
        """Get the gripper position.
        
        Returns:
            numpy.ndarray: The gripper position.
        """
        return self._gripper_pos
    
    def get_observations(self):
        """Get all robot observations with all required fields."""
        pos, quat = self.get_ee_pose()
        
        # Convert to numpy arrays if needed
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        
        # Make sure gripper_pos is scalar if it's used in boolean contexts
        gripper_pos = float(self._gripper_pos[0]) if isinstance(self._gripper_pos, np.ndarray) and len(self._gripper_pos) > 0 else 0.5
        
        # Create observations with all required fields that might be used by TwoStage
        return {
            'joint_positions': self._joint_positions,
            'robot0_eef_pos': pos,
            'robot0_eef_quat': quat,
            'robot0_gripper_qpos': gripper_pos,
            'robot0_desired_gripper_qpos': gripper_pos,  # Add this field explicitly
        }

class MockGripper:
    @dataclass
    class Metadata:
        max_width = 0.08

    @dataclass
    class State:
        width: float

    def __init__(self):
        self.metadata = MockGripper.Metadata()
        self.width = self.metadata.max_width

    # Add to MockGripper class
    def get_current_position(self):
        """Get the current gripper position."""
        return self.width
    
    def goto(self, width, speed, force, blocking):
        assert width <= self.metadata.max_width, f"{width=}, {self.metadata.max_width=}"
        print(f"[mock]: gripper.goto {width:.4f}")
        self.width = width

    def get_state(self):
        return MockGripper.State(width=self.width)

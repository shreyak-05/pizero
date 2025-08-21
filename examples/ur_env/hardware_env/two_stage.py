import time
import numpy as np
from scipy.spatial.transform import Rotation


class TwoStageEEConfig:
    def __init__(self):
        self.init_ee_pos = [0.5, 0, 0.32]  # Initial end-effector position
        self.home = np.array(
            [
                np.deg2rad(-90.0),
                np.deg2rad(-90.0),
                np.deg2rad(-90.0),
                np.deg2rad(-90.0),
                np.deg2rad(90.0),
                np.deg2rad(90.0),
                # -1.275836,  # press toast
                # -1.6735962,  # 2nd joint, vertical, negative: go up, positive: go down
                # -0.73670348,  # 3rd joint, horizontal,
                # -2.301,
                # 1.5694001,
                # 1.88216307,  # 6th, smaller -> inward
                0.0,  # np.pi * 0.5,  # control the rotation of the gripper
            ],
            dtype=np.float32,
        )

        self.pos_low = np.array([0.45, -0.15, 0.19])
        self.pos_high = np.array([0.65, 0.15, 0.4])
        self.rot_abs_min = np.pi * np.array([0.75, 0, 0.0]).astype(np.float32)
        self.rot_abs_max = np.pi * np.array([1, 0.25, 1]).astype(np.float32)        
        self.ee_range_low = self.pos_low.tolist() + [-np.pi, -np.pi, -np.pi]
        self.ee_range_high = self.pos_high.tolist() + [np.pi, np.pi, np.pi]    
        
        
    def clip(self, pos: np.ndarray, rot: np.ndarray):
        pos = np.clip(pos, self.pos_low, self.pos_high)
        rot = np.sign(rot) * np.clip(np.abs(rot), self.rot_abs_min, self.rot_abs_max)
        return pos, rot    
    
    def ee_in_good_range(self, pos: np.ndarray, quat: np.ndarray, verbose):
        # rot = Rotation.from_quat(quat).as_euler("xyz")
        if np.any((pos < self.pos_low - 0.02) | (pos > self.pos_high + 0.02)):
            if verbose:
                print(f"bad pos: {pos}")
            return False        
        # rot_abs = np.abs(rot)
        # if any((rot_abs < self.rot_abs_min) | (rot_abs > self.rot_abs_max)):
        #     if verbose:
        #         print(f"bad rot: {rot}")
        #     return False        
        return True    
    
    def reset(self, robot):
        min_height = 0.4
        ee_pos, ee_quat = robot.get_ee_pose()
        if ee_pos[2] < min_height:
            print("fixing position")
            # robot.update_desired_ee_pose([ee_pos[0], ee_pos[1], min_height], ee_quat)






class TwoStage:
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.prev_gripper_qpos = 0
        self.hold_count = 0
        self.z_at_hold = 0
        self.placement_threshold = 0.02
        self.default_desired_gripper_qpos = 0.5

        # self.stage = 0
        # self.gripper_used = 0        # Define initial positions and goals for each stage
        # self.obj_positions = {
            # 0: np.array([0.0, 0.6, 0.02]),  # Initial position of the pot
            # 1: np.array([0.0, 0.6, 0.02]),  # Position of the broccoli
            # 2: np.array([0.0, 0.6, 0.02])   # Position of the lid
        # }
        # self.goal_positions = {
            # 0: np.array([0.1, 0.8, 0.2]),   # Place pot on the stove
            # 1: np.array([0.1, 0.8, 0.2]),   # Place broccoli in the pot
            # 2: np.array([0.1, 0.8, 0.2])    # Place lid on the pot
        # }    
        
    def reset(self):
        # self.stage = 0
        # self.gripper_used = 0
        # Reset positions or any other necessary state    

        self.prev_gripper_qpos = 0
        self.hold_count = 0
        self.z_at_hold = 0
    
    def reward(self, obs):
            """Calculate reward based on observation.
            
            Args:
                obs: Observation dictionary containing robot state
                
            Returns:
                float: Reward value [0-1]
            """
            try:
                # Check if obs is None or empty
                if obs is None or not isinstance(obs, dict):
                    if self.verbose:
                        print("Warning: Observation is None or not a dictionary")
                    return 0.0
                    
                # Create a working copy to avoid modifying the original
                obs_copy = dict(obs)
                
                # Add default values for missing keys
                required_keys = [
                    'robot0_desired_gripper_qpos',
                    'robot0_gripper_qpos',
                    'robot0_eef_pos',
                    'robot0_eef_quat'
                ]
                
                default_values = {
                    'robot0_desired_gripper_qpos': self.default_desired_gripper_qpos,
                    'robot0_gripper_qpos': 0.5,
                    'robot0_eef_pos': np.zeros(3),
                    'robot0_eef_quat': np.array([0, 0, 0, 1])
                }
                
                # Add missing keys with default values
                for key in required_keys:
                    if key not in obs_copy or obs_copy[key] is None:
                        obs_copy[key] = default_values[key]
                
                # Very simple placeholder reward function for mock mode - return small positive value
                return 0.1
                
            except Exception as e:
                if self.verbose:
                    print(f"Error in TwoStage.reward: {e}")
                return 0.0
        
    def is_done(self, obs):
        """Check if task is complete.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            bool: True if task is complete
        """
        try:
            # Check if obs is None or empty
            if obs is None or not isinstance(obs, dict):
                if self.verbose:
                    print("Warning: Observation is None or not a dictionary")
                return False
                
            # Create a working copy to avoid modifying the original
            obs_copy = dict(obs)
            
            # Add default values for missing keys
            required_keys = [
                'robot0_desired_gripper_qpos',
                'robot0_gripper_qpos',
                'robot0_eef_pos',
                'robot0_eef_quat'
            ]
            
            default_values = {
                'robot0_desired_gripper_qpos': self.default_desired_gripper_qpos,
                'robot0_gripper_qpos': 0.5,
                'robot0_eef_pos': np.zeros(3),
                'robot0_eef_quat': np.array([0, 0, 0, 1])
            }
            
            # Add missing keys with default values
            for key in required_keys:
                if key not in obs_copy or obs_copy[key] is None:
                    obs_copy[key] = default_values[key]
            
            # Simple placeholder completion check - always return False for mock mode
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"Error in TwoStage.is_done: {e}")
            return False
    

    # def reward(self, curr_prop):
    #     # if self.verbose:
    #         # print("---------------------------")
    #         # pprint.pprint(curr_prop)        
        
    #     desired_gripper_qpos = curr_prop["robot0_desired_gripper_qpos"].item()
    #     real_gripper_qpos = curr_prop["robot0_gripper_qpos"].item()        # Check if the gripper is opening or closing
    #     if desired_gripper_qpos > 0.8:
    #         if self.verbose:
    #             print(">>> Gripper is opening")
    #         return 0  # No reward if gripper is opening        
    #     diff = abs(self.prev_gripper_qpos - real_gripper_qpos)
    #     if diff >= 1e-2:
    #         if self.verbose:
    #             print(f">>> Gripper is still changing {diff}")
    #         return 0  # No reward if gripper is still moving        # Check if object has been placed
    #     box_height = 0.14
    #     object_pos = curr_prop["robot0_eef_pos"]  # End-effector position
    #     if abs(object_pos - box_height) < self.placement_threshold:
    #         if self.verbose:
    #             print(">>> Object placed on box")
    #         return 1  # Reward for successful placement        
    #     if self.verbose:
    #         print(">>> Object not yet placed")
    #     return 0  # No reward if object is not placed correctly





    def perform_stage(self, robot):
        # Perform actions based on the current stage
        if self.stage < 3:
            obj_pos = self.obj_positions[self.stage]
            goal_pos = self.goal_positions[self.stage]
            # This function needs to be implemented to handle robot-specific actions
            robot.move_to(obj_pos)
            robot.grasp()
            robot.move_to(goal_pos)
            robot.release()            
            if self.verbose:
                print(f"Completed stage {self.stage}: Moved from {obj_pos} to {goal_pos}")
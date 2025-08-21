#!/usr/bin/env python3
"""
A standalone script to visualize UR3e joint trajectories in PyBullet.
Replace the fetch_next_action() stub with your own inference loop
(or import & call your controller/query logic).
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
from ur_ikfast.ur_ikfast import ur_kinematics

# Initialize kinematics for UR5e (for FK) and UR3e (for IK)
ur5_arm = ur_kinematics.URKinematics('ur5e')
ur3_arm = ur_kinematics.URKinematics('ur3e')

def compute_ur3_command(action, last_seed):
    """
    Converts a 7-dim UR5e-style action into a UR3e joint+gripper command.
    action: [j1..j6, gripper]
    last_seed: previous 6-joint array for IK seeding
    Returns: 7-element array [ur3_j1..j6, gripper]
    """
    gripper = action[6]
    ur5_joints = np.array(action[:6], dtype=float)

    # 1) UR5e FK -> end-effector pose
    pose = ur5_arm.forward(ur5_joints)

    # 2) UR3e IK -> joint solution
    ur3_j = ur3_arm.inverse(pose, False, q_guess=last_seed)
    if ur3_j is None:
        # fallback: shrink pose into UR3e reach
        pose[:3] *= 0.6
        ur3_j = ur3_arm.inverse(pose, False, q_guess=last_seed)
        if ur3_j is None:
            return None, last_seed

    # 3) clip to ±pi
    ur3_j = np.clip(ur3_j, -np.pi, np.pi)

    # 4) append gripper
    ur3_full = np.concatenate([ur3_j, [gripper]])
    return ur3_full, ur3_j

def fetch_next_action(step):
    """
    Stub: replace this with your controller/query logic.
    Here we just generate a dummy sinusoidal joint motion.
    """
    j = np.sin(0.02 * step + np.arange(6) * 0.5)
    return np.concatenate([j, [0.5]])

def main():
    # 1) Start PyBullet GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # 2) Load ground plane and UR3e URDF
    plane = p.loadURDF("plane.urdf")
    ur3e_id = p.loadURDF("/home/shreya/Desktop/gendp/sapien_env/sapien_env/assets/robot/ur_description/ur3e.urdf", useFixedBase=True)

    # 3) Setup simulation parameters
    p.setGravity(0, 0, -9.81)
    sim_timestep = 1.0 / 240.0
    p.setTimeStep(sim_timestep)

    last_seed = np.zeros(6)
    try:
        step = 0
        while True:
            # 4) Get the next action (replace stub as needed)
            action = fetch_next_action(step)

            # 5) Compute the UR3e command
            ur3_full, last_seed = compute_ur3_command(action, last_seed)
            if ur3_full is not None:
                joint_angles = ur3_full[:6]
                # 6) Apply to PyBullet
                for i, q in enumerate(joint_angles):
                    p.resetJointState(ur3e_id, i, q)
                p.stepSimulation()

            # 7) Pause to match your real control loop
            time.sleep(0.05)
            step += 1

    except KeyboardInterrupt:
        print("Exiting simulation.")

    finally:
        p.disconnect()

if __name__ == "__main__":
    main()


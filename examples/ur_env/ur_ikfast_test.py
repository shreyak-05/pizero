from ur_ikfast.ur_ikfast import ur_kinematics
import numpy as np

ur3e_arm = ur_kinematics.URKinematics('ur3e')
ur5e_arm = ur_kinematics.URKinematics('ur5e')

# Inverse Kinematics
ee_pose_quaternion = [0.43093423, 0.2255771, 0.07461681, 0.01503962, 0.70575622, 0.70776805, 0.02731932]
joint_angles = ur3e_arm.inverse(ee_pose_quaternion, False, q_guess=np.zeros(6))
print("Joint Angles:", joint_angles)
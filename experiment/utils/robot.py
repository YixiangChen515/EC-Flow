import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

GeomTypeList = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
RobotGeomList = ['mesh', 'cylinder', 'mesh', 'mesh', 'box', 'mesh', 'mesh', 'cylinder', 'mesh', 'mesh', 'cylinder', 'mesh', 'mesh', 'cylinder', 'mesh', 'cylinder', 'box', 'box', 'box', 'box', 'box','cylinder']

Geom2RobotPart = [
    "right_arm_base_link",
    "right_arm_base_link",
    "right_l0",
    "head",
    "screen",
    "right_l1",
    "right_l2",
    "right_l2",
    "right_l3",
    "right_l4",
    "right_l4",
    "right_l5",
    "right_l6",
    "right_l6",
    "right_hand",
    "right_hand",
    "hand",
    "rightclaw",
    "rightpad",
    "leftclaw",
    "leftpad",
    "right_l1_2",
]

def find_sublist_index(full_list, sub_list):
    sub_len = len(sub_list)
    for i in range(len(full_list) - sub_len + 1):
        if full_list[i:i+sub_len] == sub_list:
            return i
    return -1



# class RobotParser:
#     def __init__(self, xml_file):
#         self.model = mujoco.MjModel.from_xml_path(xml_file)
#         self.data = mujoco.MjData(self.model)

class RobotParser:
    def __init__(self, env):
        self.env = env
        self.model = self.env.model
        self.data = self.env.data

        for i in range(self.model.nbody):
            joint_name = self.get_name(i)
            if joint_name == "right_arm_base_link":
                self.robot_start_idx = i
            if joint_name == "right_l1_2":
                self.robot_end_idx = i
                break
        
        self.robot_id_list = list(range(self.robot_start_idx, self.robot_end_idx+1))
        self.robot_name_list = [self.get_name(i) for i in self.robot_id_list]
        
        total_geom_list = [GeomTypeList[i] for i in self.model.geom_type]
        robot_geom_idx = find_sublist_index(total_geom_list, RobotGeomList)
        self.geom_id_list = list(range(robot_geom_idx, robot_geom_idx + len(RobotGeomList)))

    def get_name(self, idx, type="body"):
        if type=="body":
            mojoco_type = mujoco.mjtObj.mjOBJ_BODY
        elif type=="geom":
            mojoco_type = mujoco.mjtObj.mjOBJ_GEOM
        return mujoco.mj_id2name(self.model, mojoco_type, idx)
    
    def get_id(self, name, type="body"):
        if type=="body":
            mojoco_type = mujoco.mjtObj.mjOBJ_BODY
        elif type=="geom":
            mojoco_type = mujoco.mjtObj.mjOBJ_GEOM
        return mujoco.mj_name2id(self.model, mojoco_type, name)
    
    def get_relative_transform(self, child_id):

        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(self.model.body_quat[child_id], scalar_first=True).as_matrix()
        transform[:3, 3] = self.model.body_pos[child_id]

        return transform
    
    
    def get_joint_world_transforms(self, T_end, end_effector_name="hand"):
        joint_transforms = {}
        
        child_idx = self.get_id(end_effector_name)
        joint_transforms[end_effector_name] = T_end @ np.eye(4)
        parent_idx = self.model.body_parentid[child_idx]
        # From End-Effector to Base
        while parent_idx >= self.robot_start_idx and parent_idx <= self.robot_end_idx:
            parent_joint_name = self.get_name(parent_idx)
            child_joint_name = self.get_name(child_idx)
            joint_transforms[parent_joint_name] = joint_transforms[child_joint_name] \
                                                        @ np.linalg.inv(self.get_relative_transform(child_idx))
            
            child_idx = parent_idx
            parent_idx = self.model.body_parentid[parent_idx]
            
        
        # Calculate the rest of the joints
        for i in range(self.robot_start_idx, self.robot_end_idx+1):
            joint_name = self.get_name(i)  
            if joint_name not in joint_transforms:
                parent_joint_name = self.get_name(self.model.body_parentid[i])
                joint_transforms[joint_name] = joint_transforms[parent_joint_name] @ self.get_relative_transform(i)
                
        return joint_transforms
        
    def compute_joint_ranges(self):
        joint_bounds = {}
        total_geom_list = [GeomTypeList[i] for i in self.model.geom_type]
        robot_geom_idx = find_sublist_index(total_geom_list, RobotGeomList)
        
        for geom_id in self.geom_id_list:
            geom_name = Geom2RobotPart[geom_id-robot_geom_idx]
            # Extract geom_aabb
            geom_aabb = self.model.geom_aabb[geom_id]
            center_local = geom_aabb[:3]  # First 3 numbers (center in geom frame)
            half_sizes = geom_aabb[3:]  # Last 3 numbers (half-sizes)

            # Extract geom_xpos and geom_xmat
            geom_xpos = self.data.geom_xpos[geom_id]  # Position of the geom in world frame
            geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)  # Orientation matrix of the geom

            # Transform center from geom frame to world frame
            center_world = geom_xpos + geom_xmat @ center_local

            # Define local corner offsets for the AABB
            local_offsets = np.array([
                [-1, -1, -1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1,  1,  1],
                [ 1, -1, -1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [ 1,  1,  1]
            ]) * half_sizes

            # Compute world corners
            corners = np.array([center_world + geom_xmat @ offset for offset in local_offsets])
            
            
            
            # Return all the corners
            if geom_name in joint_bounds:
                joint_bounds[geom_name] = np.concatenate((joint_bounds[geom_name], corners), axis=0)
            else:
                joint_bounds[geom_name] = corners
        return joint_bounds
    
    def get_geom_center(self, geom_name):
        geom_id = self.get_id(geom_name, type="geom")
        geom_aabb = self.model.geom_aabb[geom_id]
        center_local = geom_aabb[:3]  # First 3 numbers (center in geom frame)

        # Extract geom_xpos and geom_xmat
        geom_xpos = self.data.geom_xpos[geom_id]  # Position of the geom in world frame
        geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)  # Orientation matrix of the geom

        # Transform center from geom frame to world frame
        center_world = geom_xpos + geom_xmat @ center_local
        
        return center_world


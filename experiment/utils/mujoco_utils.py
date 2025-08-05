from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco

GeomTypeList = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
RobotGeomList = ['mesh', 'cylinder', 'mesh', 'mesh', 'box', 'mesh', 'mesh', 'cylinder', 'mesh', 'mesh', 'cylinder', 'mesh', 'mesh', 'cylinder', 'mesh', 'cylinder', 'box', 'box', 'box', 'box', 'box','cylinder']

Geom2RobotPart = [
    "right_arm_base_link",
    "TRASPARENT",
    "right_l0",
    "head",
    "TRASPARENT",
    "right_l1",
    "right_l2",
    "TRASPARENT",
    "right_l3",
    "right_l4",
    "TRASPARENT",
    "right_l5",
    "right_l6",
    "TRASPARENT",
    "right_hand",
    "right_hand",
    "hand",
    "rightclaw",
    "rightpad",
    "leftclaw",
    "leftpad",
    "TRASPARENT"
]


def find_sublist_index(full_list, sub_list):
    sub_len = len(sub_list)
    for i in range(len(full_list) - sub_len + 1):
        if full_list[i:i+sub_len] == sub_list:
            return i
    return -1


def get_name(model, idx, type="body"):
    if type=="body":
        mojoco_type = mujoco.mjtObj.mjOBJ_BODY
    elif type=="geom":
        mojoco_type = mujoco.mjtObj.mjOBJ_GEOM
    return mujoco.mj_id2name(model, mojoco_type, idx)


def get_id(model, name, type="body"):
    if type=="body":
        mojoco_type = mujoco.mjtObj.mjOBJ_BODY
    elif type=="geom":
        mojoco_type = mujoco.mjtObj.mjOBJ_GEOM
    return mujoco.mj_name2id(model, mojoco_type, name)


def quat_to_rotation_matrix(quat):
        qw, qx, qy, qz = quat
        R_mat = np.stack([
            np.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], axis=-1),
            np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)], axis=-1),
            np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)], axis=-1)
        ], axis=-2)
        return R_mat
    
    
def get_relative_pose(pos, quat):

    transform = np.eye(4)
    rotation_matrix = quat_to_rotation_matrix(quat)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = pos
                                                                                                                                                                                                                                                
    return transform
    
    
def get_world_pose(pos, mat):
    
    transform = np.eye(4)
    transform[:3, :3] = mat
    transform[:3, 3] = pos
    
    return transform


def get_robot_geom_id(model):
    total_geom_list = [GeomTypeList[i] for i in model.geom_type]
    robot_geom_idx = find_sublist_index(total_geom_list, RobotGeomList)
    geom_id_list = list(range(robot_geom_idx, robot_geom_idx + len(RobotGeomList)))
    return geom_id_list

def get_mujoco_info(model, data, geom_info=True):
    for i in range(model.nbody):
        joint_name = get_name(model, i)
        if joint_name == "right_arm_base_link":
            robot_start_idx = i
        if joint_name == "right_l1_2":
            robot_end_idx = i
            break
        
    robot_id_list = list(range(robot_start_idx, robot_end_idx+1))    
    total_geom_list = [GeomTypeList[i] for i in model.geom_type]
    
    
    mujoco_info = {}
    mujoco_info["nbody"] = len(robot_id_list)
    
    
    mujoco_info["body"] = [
        {
            "name": get_name(model, i),
            "parent_id": model.body_parentid[i] - robot_start_idx,
            "world_pose": get_world_pose(data.xpos[i], data.xmat[i].reshape(3, 3)),
            # "relative_pose": get_relative_pose(   \
            #                             model.body_pos[i], model.body_quat[i]),
        }
        for i in robot_id_list
    ]
    mujoco_info["body_name_to_idx"] = {body_info["name"]: i for i, body_info in enumerate(mujoco_info["body"])}
    # mujoco_info["mocap_pos"] = data.body('mocap').xpos
    
    if geom_info:
        robot_geom_idx = find_sublist_index(total_geom_list, RobotGeomList)
        geom_id_list = list(range(robot_geom_idx, robot_geom_idx + len(RobotGeomList)))
        mujoco_info["ngeom"] = len(geom_id_list)
        mujoco_info["geom"] = [
            {
                    "geom_name": Geom2RobotPart[i-geom_id_list[0]],
                    "geom_aabb": model.geom_aabb[i],
                    "geom_xpos": data.geom_xpos[i],
                    "geom_xmat": data.geom_xmat[i],  
            }
            for i in geom_id_list
        ]
    return mujoco_info

    

def get_hand_bounds(env):
    hand_low, hand_high = env.hand_low, env.hand_high
    soft_bounds_margin = 0.05 # do not reach actual bounds
    
    return [np.array([low+soft_bounds_margin, high-soft_bounds_margin]) for low, high in zip(hand_low, hand_high)]

                                                                                                                                                                                                                 
def compute_joint_ranges(model, data):
    mujoco_info = get_mujoco_info(model, data)
    
    joint_bounds = {}
    geom_center = {}
    
    for geom_id in range(mujoco_info["ngeom"]):
        geom_name = mujoco_info["geom"][geom_id]["geom_name"]
        
        if geom_name == "TRASPARENT":
            continue
        
        else:
            # Extract geom_aabb
            geom_aabb = mujoco_info["geom"][geom_id]["geom_aabb"]
            center_local = geom_aabb[:3]  # First 3 numbers (center in geom frame)
            half_sizes = geom_aabb[3:]  # Last 3 numbers (half-sizes)

            # Extract geom_xpos and geom_xmat
            geom_xpos = mujoco_info["geom"][geom_id]["geom_xpos"]  # Position of the geom
            geom_xmat = mujoco_info["geom"][geom_id]["geom_xmat"].reshape(3, 3)  # Rotation matrix of the geom

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
                
            geom_center[geom_name] = center_world
    return geom_center, joint_bounds


class LevenbegMarquardtIK:
    
    def __init__(self, mujoco_model_data_ik=(), step_size=0.5, tol=0.03, alpha=0.5, damping=0.15):
        self.model, self.data = mujoco_model_data_ik
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.damping = damping
        
        self.init_q = self.data.qpos.copy()
        
        # FIXME: this only assumes one free joint
        free_joint = [i for i, jt in enumerate(self.model.jnt_type) if jt == 0]
        if free_joint:
            self.free_joint_adr = int(self.model.jnt_qposadr[free_joint])
        else:
            self.free_joint_adr = None
        
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, end_effctor="hand"):
        """Calculate the desire joints angles for goal"""
        
        body_id = self.model.body(end_effctor).id
        self.data.qpos = self.init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)

        iter = 0
        max_iter = 100
        while (np.linalg.norm(error) >= self.tol) and (iter < max_iter):
            #calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
        
            delta_q = j_inv @ error
            
            # FIXME: find a better way to remove free joint
            try:
                # Remove free joint
                if self.free_joint_adr:
                    #compute next step
                    self.data.qpos[:self.free_joint_adr] += self.step_size * delta_q[:self.free_joint_adr]
                    self.data.qpos[self.free_joint_adr+7:] += self.step_size * delta_q[self.free_joint_adr+6:]
                    self.check_joint_limits(self.data.qpos[:self.free_joint_adr])
                    self.check_joint_limits(self.data.qpos[self.free_joint_adr+7:])
                else:
                    self.data.qpos += self.step_size * delta_q
                    self.check_joint_limits(self.data.qpos)
            except:
                print(self.model.jnt_type)
            #check limits
            
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
            
            iter += 1
            
        if iter == max_iter:
            return False
        return True      
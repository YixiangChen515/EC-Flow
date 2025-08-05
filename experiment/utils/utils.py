
import numpy as np
from utils.camera import transform_world_to_pixels, transform_pixels_to_world
from utils.rigid_transform import *
from utils.mujoco_utils import compute_joint_ranges
import matplotlib.path as mpath
import copy
from metaworld.envs.mujoco.sawyer_xyz import v2
import cv2

def get_center_crop_img(original_img, crop_size=(240, 200)):
    h, w = original_img.shape[:2]
    
    center_h, center_w = h // 2, w // 2
    crop_size_half_h, crop_size_half_w = crop_size[0] // 2, crop_size[1] // 2

    crop_img = np.zeros_like(original_img)

    top = center_h - crop_size_half_h
    bottom = center_h + crop_size_half_h
    left = center_w - crop_size_half_w
    right = center_w + crop_size_half_w

    crop_img[top:bottom, left:right] = original_img[top:bottom, left:right]
    
    return crop_img

# sorting points for constructing polygon
def sort_points_by_angle(points):
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]
        

def sample_from_mask(mask, num_samples=100):
    on = np.array(mask.nonzero()[::-1]).T.astype(np.float64)  # Reverse the order to get (x, y)
    if len(on) == 0:
        on = np.array((mask == 0).nonzero()[::-1]).T.astype(np.float64)  # Same for the empty case
    sample_ind = np.random.choice(len(on), num_samples, replace=True)
    ### add +-0.5 uniform noises to the samples
    samples = on[sample_ind]
    samples += np.random.uniform(-0.5, 0.5, samples.shape)
    return samples


def sample_with_binear(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = max(0, int(kp[0])), max(0, int(kp[1]))
    x1, y1 = min(max_x, x0+1), min(max_y, y0+1)
    x, y = max(0, kp[0]-x0), max(0, kp[1]-y0)
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature


def get_grasp(seg, depth, camera_to_world_transform, r=5):
    samples = sample_from_mask(seg, 500)
    def loss(i):
        return np.linalg.norm(samples - samples[i], axis=1).sum()
    grasp_2d = samples[np.argmin([loss(i) for i in range(len(samples))])]
    neighbor_threshold = r
    neighbors = samples[np.linalg.norm(samples - grasp_2d, axis=1) < neighbor_threshold]
    # neighbors_d = np.array([[sample_with_binear(depth, kp)] for kp in neighbors])
    neighbors_d = np.array([[sample_with_binear(depth, kp)] for kp in neighbors])
    d = np.median(neighbors_d)

    return transform_pixels_to_world(grasp_2d, [d], camera_to_world_transform)



def assign_points_to_joints(joint_ranges, points1_uv, points1_3d, points2_uv):
    keys = ["2d_last_frame", "3d_last_frame", "2d_next_frame", "idx_list"]
    joint_dict = {joint: {key: [] for key in keys} for joint in joint_ranges}
    joint_paths = {joint: mpath.Path(joint_corners) for joint, joint_corners in joint_ranges.items()}
    
    for i, point1_uv in enumerate(points1_uv):
        assigned_joints = []

        # Check which joint ranges the point belongs to
        for joint, polygon_path in joint_paths.items():
            if polygon_path.contains_point(point1_uv):
                assigned_joints.append(joint)

        # If the point belongs to exactly one joint, add it to the corresponding joint
        if len(assigned_joints) == 1:
            joint_dict[assigned_joints[0]]["2d_last_frame"].append(point1_uv)
            joint_dict[assigned_joints[0]]["3d_last_frame"].append(points1_3d[i])
            joint_dict[assigned_joints[0]]["2d_next_frame"].append(points2_uv[i])

    # Remove joints with no points assigned
    result_dict = {
            joint: {
                "2d_last_frame": np.array(points["2d_last_frame"]),
                "3d_last_frame": np.array(points["3d_last_frame"]),
                "2d_next_frame": np.array(points["2d_next_frame"]),
            }
            for joint, points in joint_dict.items()
            if len(points["2d_last_frame"]) > 5
        }

    return result_dict


# Using Tracking
def get_transforms(points1_uv, points1_3d, points2_uv, camera_to_world_transform, initial_env_info, present_env_info, ransac_tries=50, ransac_threshold=0.5, rgd_tfm_tries=5):
    model, data, hand_bounds = initial_env_info
    world_to_camera_transform = np.linalg.inv(camera_to_world_transform)
    geom_center, robot_joint_corner_3d = compute_joint_ranges(model, data) 
    # print("geom_center", geom_center)
    
    robot_joint_range_2d = {}        
    for joint_name, joint_corner_3d in robot_joint_corner_3d.items():
        joint_corner_2d = transform_world_to_pixels(joint_corner_3d, world_to_camera_transform)
        robot_joint_range_2d[joint_name] = sort_points_by_angle(joint_corner_2d)
        
    # # For Debug joint bounding box
    # for joint_name, joint_range in robot_joint_range_2d.items():
    #     debut_img = cv2.cvtColor(seg_robot, cv2.COLOR_GRAY2BGR)
    #     sorted_points_int = joint_range.astype(np.int32)
    #     cv2.polylines(debut_img, [sorted_points_int], isClosed=True, color=(0, 255, 0), thickness=2)
    #     cv2.imwrite(f"seg_robot_box_{joint_name}.png", debut_img)
        
        
    # Step 1: Assign points to joints
    point_list = assign_points_to_joints(robot_joint_range_2d, points1_uv, points1_3d, points2_uv)

    # Step 2: Create point list for each joint
    for joint_name, joint_points in point_list.items():
        joint_points.update({"center_3d": np.array([geom_center[joint_name]])})
    
    # Find inliers
    for joint_name, point in point_list.items():
        center_uv = transform_world_to_pixels(point["center_3d"], world_to_camera_transform)[0]

        # Apply RANSAC to find inliers
        _, inliers = ransac(point["2d_last_frame"], center_uv, point["2d_next_frame"], ransac_tries, ransac_threshold)

        # Store results if inliers are found
        if len(inliers) > 0:
            point_list[joint_name] = {key: point[key][inliers] for key in ["2d_last_frame", "3d_last_frame", "2d_next_frame"]}

        
    # Step 4: Solve for 3D rigid transformation
    mujoco_model_data_original = (model, data)
    mujoco_model_data_ik = present_env_info
    
    solution = solve_3d_rigid_tfm(point_list, world_to_camera_transform, \
                                            mujoco_model_data_original, mujoco_model_data_ik, hand_bounds, rgd_tfm_tries)
    return np.array(solution.x), point_list
import numpy as np
import mujoco

def get_camera_extrinsic_matrix(env, camera_name):

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_pos = env.data.cam_xpos[cam_id]
    camera_rot = env.data.cam_xmat[cam_id].reshape(3, 3)
    
    R = np.zeros((4, 4))
    R[:3, :3] = camera_rot
    R[:3, 3] = camera_pos
    R[3, 3] = 1.0

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R

def get_camera_intrinsic_matrix(env, camera_name, camera_height, camera_width ):

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
    fovy = env.model.cam_fovy[cam_id]

    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    # FIXME: Change the second line f to -f to match the result, don't know why
    # K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    K = np.array([[f, 0, camera_width / 2], [0, -f, camera_height / 2], [0, 0, 1]])
    return K

def get_camera_transform_matrix(env, camera_name, resolution):
    camera_height, camera_width = resolution
    R = get_camera_extrinsic_matrix(env=env, camera_name=camera_name)
    K = get_camera_intrinsic_matrix(
        env=env, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
    )
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    inv_R = np.zeros((4, 4))
    inv_R[:3, :3] = R[:3, :3].T
    inv_R[:3, 3] = -inv_R[:3, :3].dot(R[:3, 3])
    inv_R[3, 3] = 1.0
    
    return K_exp @ inv_R

def transform_pixels_to_world(pixels, depth, camera_to_world_transform):
    # pixels in uv space (not row col)
    # sample from the depth map using the pixel locations with bilinear sampling
    pixels = pixels.astype(float)
    depth = np.array(depth)

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    # cam_pts = [pixels[..., 1:2] * depth, pixels[..., 0:1] * depth, depth, np.ones_like(depth)]
    cam_pts = [pixels[..., 0:1] * depth, pixels[..., 1:2] * depth, depth, np.ones_like(depth)]
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
    return points[..., :3]


def transform_world_to_pixels(points, world_to_camera_transform):
    # pixels in uv space (not row col)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(world_to_camera_transform, points.T).T
    pixels = points[:, 0:2] / points[:, 2:3]
    return pixels
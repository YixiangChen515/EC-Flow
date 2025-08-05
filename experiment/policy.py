from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.action import Action
import numpy as np

import json
import cv2
from utils.utils import get_grasp, get_transforms, sample_from_mask, sample_with_binear, get_center_crop_img
from utils.camera import get_camera_transform_matrix, transform_pixels_to_world
from utils.mujoco_utils import get_robot_geom_id, get_hand_bounds


import mujoco
import copy

# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)
with open("name2mode.json", "r") as f:
    name2mode = json.load(f)
with open("name2graspdev.json", "r") as f:
    name2graspdev = json.load(f)
    
    
class MyFlowPolicy(Policy):
    def __init__(self, env, task, camera, flow_pred_model, plan_timeout=15, max_replans=0, log=False, seed=0):
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.resolution = (240, 320)
        
        self.world2camera_mat = get_camera_transform_matrix(env, camera, self.resolution)
        self.camera2world_mat = np.linalg.inv(self.world2camera_mat)
        
        self.mode = name2mode[task]
        self.grasp_dev = name2graspdev[task]
        
        self.flow_pred_model = flow_pred_model
        
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log
        self.seed = seed    # For debug
        
        self.init_obs = self.env.reset()
        self.pred_flows = []    # For Visualization
        
        depth = self.env_render("depth")
        seg_img = self.env_render("seg")
        sed_obj = self.get_seg(seg_img, seg_ids=self.seg_ids)
        
        self.grasp = get_grasp(sed_obj, depth, self.camera2world_mat)
        # print("Grasp position:", self.grasp)
        
        self.init_grasp()  
        
        self.replan_countdown = self.plan_timeout
        self.images, self.episode_return = self.collect_video()
        

    def env_render(self, mode):
        
        if mode == "rgb":
            self.env.modify_render_mode('rgb')
            image = self.env.render()
            image = np.flipud(image)
            
            return image
        
        elif mode == "depth":
            self.env.modify_render_mode('depth')

            depth = self.env.render()
            def depthimg2meters(depth):
                extent = self.env.model.stat.extent
                near = self.env.model.vis.map.znear * extent
                far = self.env.model.vis.map.zfar * extent
                return near / (1.0 - depth * (1.0 - near / far))
            
            depth = np.flipud(depth)
            return depthimg2meters(depth)
        
        elif mode == "seg":
            self.env.modify_render_mode('rgb')
                
            seg = self.env.render(segmentation=True)
            seg = np.flipud(seg)
            
            return seg
        
        else:
            raise ValueError("Invalid mode")

    
    def get_seg(self, seg_img, seg_ids):
        img = np.zeros(seg_img.shape[:2], dtype=bool)
        types = seg_img[:, :, 0]
        ids = seg_img[:, :, 1]
        geoms = types == mujoco.mjtObj.mjOBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        # print(geoms_ids)

        for i in geoms_ids:
            if i in seg_ids:
                img[ids == i] = True
        img = img.astype('uint8') * 255
        return cv2.medianBlur(img, 3)
    
    def calc_grasp(self):
        depth = self.env_render("depth")
        seg_img = self.env_render("seg")
        sed_obj = self.get_seg(seg_img, seg_ids=self.seg_ids)
        grasp = get_grasp(sed_obj, depth, self.camera2world_mat)
        return grasp
    
    
    def recover_flow(self, flows, img_size):
        flows = np.clip(flows / 2 + 0.5, a_min=0, a_max=1)
        
        flows[..., :2] = np.clip(np.round(flows[..., :2] * [img_size[1], img_size[0]]).astype(np.int32),
                    [0, 0], [img_size[1] - 1, img_size[0] - 1])
        
        return flows
        
        
    def calc_flows(self):
        image = self.env_render("rgb")
        depth = self.env_render("depth")
        seg_img = self.env_render("seg")
        
        robot_geom_ids = get_robot_geom_id(self.env.model)
        seg_robot = self.get_seg(seg_img, seg_ids=robot_geom_ids)
        seg_robot = get_center_crop_img(seg_robot, crop_size=(240, 200))


        sample_pt = sample_from_mask(seg_robot, 400)
        flow_pred = self.flow_pred_model.flow_pred(image, self.task, sample_pt)
        flow_pred = self.recover_flow(flow_pred, self.resolution)
        
        # # For debug
        # sorted_indices = np.argsort(sample_pt[:, 0])   # Sorted by X 
        # sorted_indices = sorted_indices[np.argsort(sample_pt[sorted_indices, 1])]  # Sorted by Y 
        # sample_pt = sample_pt[sorted_indices]
        
        self.pred_flows.append({
            "flows": flow_pred,
            "image": image,
        })
        
        # Initial Points
        self.points1_uv = flow_pred[:, 0, :2]
        self.points1_3d = transform_pixels_to_world( self.points1_uv, \
                                                np.array([[sample_with_binear(depth, kp)] for kp in self.points1_uv]), \
                                                self.camera2world_mat)
        
        hand_bounds = get_hand_bounds(self.env)
        self.initial_env_info = (copy.deepcopy(self.env.model), copy.deepcopy(self.env.data), hand_bounds)
        
        # # For Debug
        # rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # for point in self.points1_uv:
        #     point = [int(p) for p in point]
        #     cv2.circle(rgb_img, point, radius=3, color=(255,0,0), thickness=-1)
        # cv2.imwrite("sample_points_all.png", rgb_img)

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        
        return flow_pred[:, 1:] # remove the first frame
    
    
    def calc_subgoal(self):
        points2_uv = self.flows[:, 0, :2]
        visible = self.flows[:, 0, -1] > 0.8
        
        threshold = 0
        
        distances = np.linalg.norm(points2_uv - self.points1_uv, axis=1)
        filtered_indices = (distances > threshold) & visible
        # print("filtered_indices:", np.sum(filtered_indices))
        
        present_env_info = (copy.deepcopy(self.env.model), copy.deepcopy(self.env.data))
        subgoal, points_list = get_transforms(self.points1_uv[filtered_indices], self.points1_3d[filtered_indices], \
                                                points2_uv[filtered_indices], self.camera2world_mat, self.initial_env_info, present_env_info)
        
        
        if self.mode == "push":
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoal -= np.array([0, 0, 0.03]) # 
        return subgoal
        
                
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            self.grasp[:2] = self.grasp[:2] + np.array(self.grasp_dev)
            self.grasp = self.grasp - np.array([0, 0, 0.03])

    def get_action(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        o_d = self._parse_obs(obs)
        
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped (all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            if not self.grasped:
                self.grasp = self.calc_grasp()
                self.init_grasp()
                return self.grasp
            else:
                self.flows = self.calc_flows()
                self.subgoal = self.calc_subgoal()
                return self.subgoal
        
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            self.flows = self.calc_flows()
            self.subgoal = self.calc_subgoal()
            return self.grasp
        
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoal) > move_precision:
            return self.subgoal
        # if close enough to the current subgoal, move to the next subgoal
        elif self.flows.shape[1] > 1:
            self.flows = self.flows[:, 1:]
            self.subgoal = self.calc_subgoal()
            return self.subgoal
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoal
    
        
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']
        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8
        
    def collect_video(self):
        images = []
        episode_return = 0
        done = False
        obs = self.init_obs 
        
        image = self.env_render("rgb")
        images += [image]
        dd = 10 ### collect a few more steps after done
        while dd:
            action = self.get_action(obs)
            try:
                obs, reward, _, _, info = self.env.step(action)
                done = info['success']
                dd -= done
                episode_return += reward
            except Exception as e:
                print(e)
                break
            if dd != 10 and not done:
                break
            if len(images) > 500:
                break

            image = self.env_render("rgb")
            images += [image]

        return images, episode_return
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import imageio

import argparse
import os
import clip

import os
import gc

from diffusion import create_diffusion
from PIL import Image

import argparse
import cv2
# from models import DiT_models
from model import DiT_models as DiT_models_track

from sam_and_track.utils import GoundedSam2
from torchvision import transforms
import colorsys
from matplotlib import cm
from sam_and_track.utils import PointTracker

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


def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def sample_from_mask(mask, num_samples=100):
    on = np.array(mask.nonzero()[::-1]).T.astype(np.float64)  # Reverse the order to get (x, y)
    if len(on) == 0:
        on = np.array((mask == 0).nonzero()[::-1]).T.astype(np.float64)  # Same for the empty case
    sample_ind = np.random.choice(len(on), num_samples, replace=True)
    samples = on[sample_ind]
    # samples += np.random.uniform(-0.5, 0.5, samples.shape)
    return samples


def process_image(image):
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        
        return image_transforms(image)
    
def normalize_flow(flow, H, W):
    flow[..., 0] /= W
    flow[..., 1] /= H
    return (flow - 0.5) / 0.5  # scale to [-1, 1]
    

def draw_point_tracking_sequence(
    image, sequence, draw_line=True, thickness=1, radius=3, add_alpha_channel=False
):
    # sequence: (num_points,T,2)
    frame = image.copy()

    def draw_point_flow(frame, tracking_data, color, draw_line):
        # tracking_data: (T,2)
        for i in range(len(tracking_data) - 1):
            start_point = (int(tracking_data[i][0]), int(tracking_data[i][1]))
            end_point = (int(tracking_data[i + 1][0]), int(tracking_data[i + 1][1]))
            if i == len(tracking_data) - 2:
                cv2.circle(frame, end_point, radius, color, -1, lineType=16)

    num_points = len(sequence)

    for i in range(num_points):
        color_map = cm.get_cmap("jet")
        color = np.array(color_map(i / max(1, float(num_points - 1)))[:3]) * 255
        color_alpha = 1
        hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])
        if add_alpha_channel:
            color = (color[0], color[1], color[2], 255.0)
        draw_point_flow(frame, sequence[i], color, draw_line)

    return frame


def viz_generated_flow(
    flows,
    initial_frame,
    normalize=True,
):
    # flows: (N,T,2)
    frame_shape = initial_frame.shape[:2]
    if normalize:
        # for t in range(flows.shape[1]):
        #     print(np.sum(flows[:, t, -1] > 0))
        flows = flows[..., :2]
        
        flows = np.clip(flows / 2 + 0.5, a_min=0, a_max=1)
        
        flows = np.clip(np.round(flows * [frame_shape[1], frame_shape[0]]).astype(np.int32),
                    [0, 0], [frame_shape[1] - 1, frame_shape[0] - 1])

    
    frames = []
    for j in range(flows.shape[1]):
        frame = draw_point_tracking_sequence(
            initial_frame.copy(),
            flows[:, :j],
        )
        frames.append(frame)
    return frames



def load_video(sequence_path):
    images = []
    for img_name in sorted(os.listdir(sequence_path)):
        if img_name.endswith('.png'):
            img_path = os.path.join(sequence_path, img_name)
            img = Image.open(img_path)
            images.append(np.array(img))
    return np.array(images)


class FlowPredModel:
    def __init__(
            self, 
            num_points = 400, 
            pred_horizon = 8, 
            num_sampling_steps = 250,
        ):
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        
        self.num_points = num_points
        self.pred_horizon = pred_horizon
        self.num_sampling_steps = num_sampling_steps

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ckpt_path = "../ckpt/flow.pt"
        
        self.flow_pred_model = DiT_models_track['DiT-XL-NoPosEmb-Lang']().to(self.device)
        
        
        state_dict = find_model(self.ckpt_path)
        self.flow_pred_model.load_state_dict(state_dict)
        self.flow_pred_model.eval()  # important!
        self.diffusion_flow = create_diffusion(str(self.num_sampling_steps))
        
        
        self.clip_model, _ = clip.load("RN50", device=self.device)
        self.clip_model.eval()
        
    def flow_pred(self, initial_image, text, samples_robot):
        
        H, W = initial_image.shape[:2]
        
        with torch.no_grad():
            # # Sorting by X, Y coordinates
            sorted_indices = np.argsort(samples_robot[:, 0])   # Sorted by X 
            sorted_indices = sorted_indices[np.argsort(samples_robot[sorted_indices, 1])]  # Sorted by Y 
            samples_robot = samples_robot[sorted_indices]
            
            samples_robot = normalize_flow(samples_robot, H, W)
            text_tokens = clip.tokenize([text]).to(self.device)
            lang =self.clip_model.encode_text(text_tokens)[0].float()
            
            # initialize action from Guassian noise
            # # [Batch, 2*T, num_points]
            z = torch.randn((1, 3*self.pred_horizon, self.num_points), device=self.device)
            
            # noise conditioned on initial points
            conditioned_points = np.ones((3, self.num_points))
            conditioned_points[:2, :] = samples_robot.T # x, y, visible
            z[0, :3, :] = torch.from_numpy(conditioned_points).to(self.device)
            
            rezize_img = cv2.resize(initial_image, (128, 128))
            init_img = np.asarray(rezize_img)
            init_img = process_image(init_img).to(self.device)

            model_kwargs = dict(y=init_img[None,:], lang=lang[None,:])
            

            naction = self.diffusion_flow.p_sample_loop(self.flow_pred_model.forward, z.shape, z, \
                                                            clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device, point_conditioned=True)

            #####
            naction = naction[0].reshape(self.pred_horizon, 3, self.num_points) # [T, 2, num_points]
            naction = naction.permute(2, 0, 1)  # [num_points, T, 2]
            naction = naction.detach().to('cpu').numpy()
            return naction

def main(args):
    # parameters
    pred_horizon = 8
    
    
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

     # Load model:
    device = 'cuda'
    flow_pred_model = DiT_models_track[args.model]().to(device)

    ckpt_path = args.ckpt 

    state_dict = find_model(ckpt_path)
    flow_pred_model.load_state_dict(state_dict)
    flow_pred_model.eval()  # important!
    diffusion_flow = create_diffusion(str(args.num_sampling_steps))
    
    img_pred_model = DiT_models_track[args.img_model](input_size=128, num_pt=400, pred_horizon=pred_horizon).to(device)
    ckpt_path = args.img_ckpt
    state_dict = find_model(ckpt_path)
    img_pred_model.load_state_dict(state_dict)
    img_pred_model.eval()  # important!
    diffusion_goal_img = create_diffusion(str(args.num_sampling_steps), goal="img_pred")
    
    # Load task data
    sequences = []
    evaluate_task = ["button-press", "button-press-topdown", "door-close", "door-open", "faucet-close", "faucet-open", "hammer", "handle-press", "shelf-place"]
    base_path = "data/metaworld_original"
    for task_name in sorted(os.listdir(base_path)):
        if task_name not in evaluate_task:
            continue
        task_path = os.path.join(base_path, task_name)
        task_name = task_name.replace("-", " ")
        if os.path.isdir(task_path):
            camera_name = "corner"
            camera_path = os.path.join(task_path, camera_name)
            if os.path.isdir(camera_path):
                for sequence_name in sorted(os.listdir(camera_path)):
                    sequence_path = os.path.join(camera_path, sequence_name)
                    if os.path.isdir(sequence_path):
                        sequences.append({
                            "path": sequence_path,
                            "task_name": task_name,
                        })
                        break


    
    prev_task_name = ""
    task_idx = 0
    result_save_path = "flow_prediction_results"
    os.makedirs(result_save_path, exist_ok=True)
    
    for sequence in sequences:
        sequence_path = sequence["path"]
        text = sequence["task_name"]
        
        if prev_task_name != text:
            task_idx = 0
            
        task_videos_rgb = load_video(sequence_path)
        task_videos = [np.transpose(image, (2, 0, 1)) for image in task_videos_rgb]

        initial_image = task_videos_rgb[0]
        
        
        H, W, C = initial_image.shape

        
        num_points = args.num_points
        
        with torch.no_grad():
            grounded_sam_model = GoundedSam2(device=device)
            tracker_model = PointTracker(device=device) # provide ground truth flow
            clip_model, _ = clip.load("RN50", device=device)
            clip_model.eval()
            
            robot_mask = grounded_sam_model.get_arm_segementation_mask(initial_image)
            robot_mask = get_center_crop_img(robot_mask, crop_size=(240, 200))
            samples_robot = sample_from_mask(robot_mask, num_points)
            
            # # Sorting by X, Y coordinates
            sorted_indices = np.argsort(samples_robot[:, 0])   # Sorted by X 
            sorted_indices = sorted_indices[np.argsort(samples_robot[sorted_indices, 1])]  # Sorted by Y 
            samples_robot = samples_robot[sorted_indices]
            
            # GT Flow
            points_queries = np.zeros((samples_robot.shape[0], 3))
            points_queries[:, 1:] = samples_robot

            
            gt_flow = tracker_model.track(task_videos, points_queries)
            gt_flow = gt_flow.squeeze(0).cpu().numpy()[..., :2]
            
            sample_indices = np.linspace(0, len(task_videos)-1, pred_horizon).astype(int)
            gt_flow_clip = gt_flow[sample_indices]
            
            samples_robot = normalize_flow(samples_robot, H, W)
            text_tokens = clip.tokenize([text]).to(device)
            lang = clip_model.encode_text(text_tokens)[0].float()

            del grounded_sam_model, clip_model, tracker_model
            gc.collect()
            torch.cuda.empty_cache()
            
        
            # initialize action from Guassian noise
            # # [Batch, 3*T, num_points]
            z = torch.randn((1, 3*pred_horizon, num_points), device=device)
            # noise conditioned on initial points
            conditioned_points = np.ones((3, num_points))
            conditioned_points[:2, :] = samples_robot.T # x, y, visible
            z[0, :3, :] = torch.from_numpy(conditioned_points).to(device)
            
            rezize_img = cv2.resize(initial_image, (128, 128))
            init_img = np.asarray(rezize_img)
            init_img = process_image(init_img).to(device)

            model_kwargs = dict(y=init_img[None,:], lang=lang[None,:])
            

            naction = diffusion_flow.p_sample_loop(flow_pred_model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, point_conditioned=True)

            #####
            naction = naction[0].reshape(pred_horizon, 3, num_points) # [T, 3, num_points]
            naction = naction.permute(2, 0, 1)  # [num_points, T, 3]
            naction = naction.detach().to('cpu').numpy()
            
            z_img = torch.randn_like(init_img)[None, :]
            diffusion_goal_img.noise_for_flow_pred = diffusion_flow.noise_for_flow_pred
            pred_goal_img = diffusion_goal_img.p_sample_loop(img_pred_model.forward, z_img.shape, z_img, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
            pred_goal_img = pred_goal_img[0][0].permute(1, 2, 0).detach().to('cpu').numpy()
            pred_goal_img = (255 * (pred_goal_img + 1) / 2).astype(np.uint8)
            
            del z, init_img, z_img
            gc.collect()
            torch.cuda.empty_cache()
            
            
            
        frames = viz_generated_flow(
                    flows=naction,
                    initial_frame=initial_image,
            )

        imageio.mimsave(os.path.join(result_save_path, f"{text}_{task_idx}_gen_flow.gif"), frames, duration=5)
        
        # gt_frames = viz_generated_flow(
        #             flows=np.transpose(gt_flow, (1, 0, 2)),
        #             initial_frame=initial_image,
        #             normalize=False,
        #     )
        
        # imageio.mimsave(os.path.join(result_save_path, f"{text}_{task_idx}_gt_flow_all.gif"), gt_frames, duration=5)
        
        gt_frames_clip = viz_generated_flow(
                    flows=np.transpose(gt_flow_clip, (1, 0, 2)),
                    initial_frame=initial_image,
                    normalize=False,
            )
        
        imageio.mimsave(os.path.join(result_save_path, f"{text}_{task_idx}_gt_flow.gif"), gt_frames_clip, duration=5)
        pred_goal_img = cv2.cvtColor(pred_goal_img, cv2.COLOR_RGB2BGR)
        pred_goal_img = cv2.resize(pred_goal_img, (320, 240))
        cv2.imwrite(os.path.join(result_save_path, f"{text}_{task_idx}_pred_goal_img.png"), pred_goal_img)
        
        task_idx += 1
        prev_task_name = text




if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=400)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--model", type=str,  default="DiT-XL-NoPosEmb-Lang") # DiT-XL-NoPosEmb-Lang, DiT-XL-NoPosEmb-Lang-Small, DiT-XL-NoPosEmb-Lang-Tiny
    parser.add_argument("--ckpt", type=str, default='ckpt/flow.pt',
                        help="path to trained checkpoint")
    parser.add_argument("--img-model", type=str,  default="DiT-S/8")
    parser.add_argument("--img-ckpt", type=str, default='ckpt/goal_img.pt',
                        help="path to trained checkpoint")

    args = parser.parse_args()
    
    main(args)







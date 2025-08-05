import torch
import clip
from torchvision import transforms
import zarr
import numpy as np
import os
import cv2
import gc
import random

# dataset
class TrackDataset(torch.utils.data.Dataset):
    def __init__(   
                self,
                args,
                n_frames: int,
                device,
                dataset = "metaworld"
            ):
        self.pred_horizon = n_frames
        self.device = device

        self.dataset = dataset
        
        self.data_path = args.data_path
        
        if self.dataset == "real_world":
            self.point_tracking_img_size = (480, 640)
        else:
            self.point_tracking_img_size = (240, 320)
            
        self.frame_resize_shape = (128, 128)
        self.sample_pt_num = 400
        
        print(f"Loading data from {self.data_path}")
        self.train_data = []
        
        
        
        self.clip_model, _ = clip.load("RN50", device=self.device)
        self.clip_model.eval()
        
        self.construct_dataset()
    
    def process_image(self, image):
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
        return image_transforms(image)
    
    
    def process_flow(self, flow):
        if isinstance(flow, np.ndarray):
            flow = torch.from_numpy(flow)
            
        flow = (flow - 0.5) / 0.5  # scale to [-1, 1]
        return flow
        
    def construct_dataset(self):
        for task in os.listdir(self.data_path):
            task_path = os.path.join(self.data_path, task)
            if os.path.isdir(task_path): 
                for episode_num in os.listdir(task_path):
                    episode_path = os.path.join(task_path, episode_num)
                    if os.path.isdir(episode_path):
                        episode = zarr.open(episode_path, mode="r")

                        point_tracking_sequence = episode["point_tracking_sequence"][:].copy()
                    
                        point_tracking_sequence = np.transpose(point_tracking_sequence, (0, 2, 1))  # (T, 3, num_pt)
                        
                        point_tracking_sequence[:, 1, :] = np.clip(point_tracking_sequence[:, 1, :], 0, self.point_tracking_img_size[0]-1)
                        point_tracking_sequence[:, 0, :] = np.clip(point_tracking_sequence[:, 0, :], 0, self.point_tracking_img_size[1]-1)
                        point_tracking_sequence[:, 1, :] = (
                            point_tracking_sequence[:, 1, :]
                            / self.point_tracking_img_size[0]
                        )
                        point_tracking_sequence[:, 0, :] = (
                            point_tracking_sequence[:, 0, :]
                            / self.point_tracking_img_size[1]
                        )

                        point_tracking_sequence = point_tracking_sequence.astype(np.float32)
                        init_img = episode["rgb_arr"][0].copy()
                        # resize the global image
                        init_img = cv2.resize(init_img, self.frame_resize_shape)
                        
                        goal_img = episode["rgb_arr"][-1].copy()
                        goal_img = cv2.resize(goal_img, self.frame_resize_shape)
                        
                        text = episode["task_description"][0]
                        text_tokens = clip.tokenize([text]).to(self.device)
                        with torch.no_grad():
                            text_embedding = self.clip_model.encode_text(text_tokens).cpu().numpy()[0].astype(float)
                        self.train_data.append(
                            {
                                "init_img": init_img,
                                "point_tracking_sequence": point_tracking_sequence,
                                "text_embedding": text_embedding,
                                "goal_img": goal_img
                            }
                        )
        del self.clip_model
        gc.collect()
        torch.cuda.empty_cache()
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        nsample = {}
        data = self.train_data[idx]

        video_length = data['point_tracking_sequence'].shape[0]
        video_sample_indices = np.linspace(0, video_length-1, self.pred_horizon).astype(int)
        
        actions = data['point_tracking_sequence'][video_sample_indices, ...].copy() # (T, 3, num_pt) 3 represents x, y, visible
        
        point_sample_indices = np.random.choice(actions.shape[-1], 
                                                        size=self.sample_pt_num, replace=False)
        point_sample_indices = np.sort(point_sample_indices)
        actions = actions[:, :, point_sample_indices]
         
        nsample['init_img'] = self.process_image(data['init_img'])
        nsample['flow'] = self.process_flow(actions)
        nsample['text'] = torch.from_numpy(data['text_embedding'])

        return nsample
    

class TrackConditionedDataset(TrackDataset):
    def __init__(self, args, n_frames: int, device, dataset = "metaworld"):
        super().__init__(args, n_frames, device, dataset)
    
    def __getitem__(self, idx):
        nsample = {}
        data = self.train_data[idx]

        video_length = data['point_tracking_sequence'].shape[0]
        if self.dataset == "real_world":
            start_idx = 0
        else:
            start_idx = random.randint(0, video_length-1)
            
        video_sample_indices = np.linspace(start_idx, video_length-1, self.pred_horizon).astype(int)
        # print(video_sample_indices)
        
        actions = data['point_tracking_sequence'][video_sample_indices, ...].copy() # (T, 3, num_pt) 3 represents x, y, visible
        
        point_sample_indices = np.random.choice(actions.shape[-1], 
                                                        size=self.sample_pt_num, replace=False)
        point_sample_indices = np.sort(point_sample_indices)
        actions = actions[:, :, point_sample_indices]
         
        nsample['init_img'], nsample['goal_img'] = self.process_image(data['init_img']), self.process_image(data['goal_img'])
        nsample['flow'] = self.process_flow(actions)
        nsample['text'] = torch.from_numpy(data['text_embedding'])

        return nsample
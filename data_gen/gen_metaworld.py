from PIL import Image
import numpy as np
import numcodecs

from sam_and_track.utils import GoundedSam2
from sam_and_track.utils import PointTracker

import zarr
from tqdm import tqdm
import torch
import os

def load_video(sequence_path):
    images = []
    for img_name in sorted(os.listdir(sequence_path)):
        if img_name.endswith('.png'):
            img_path = os.path.join(sequence_path, img_name)
            img = Image.open(img_path)
            images.append(np.array(img))
    return np.array(images)


def sample_from_mask(mask, num_samples=100):
    on = np.array(mask.nonzero()[::-1]).T.astype(np.float64)  # Reverse the order to get (x, y)
    if len(on) == 0:
        on = np.array((mask == 0).nonzero()[::-1]).T.astype(np.float64)  # Same for the empty case
    sample_ind = np.random.choice(len(on), num_samples, replace=True)
    samples = on[sample_ind]
    # samples += np.random.uniform(-0.5, 0.5, samples.shape)
    return samples


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


def max_distance_moved(points):

    T, N, _ = points.shape
    max_distances = np.zeros(N)

    for i in range(N):
        visible = np.sum(points[:, i, 2]) > 2
        if visible:
            max_dist = np.max(np.linalg.norm(points[1:, i, :2] - points[0, i, :2], axis=-1))
        else:
            max_dist = 0

        max_distances[i] = max_dist

    return max_distances

def filter_points(points_sequence, moving_threshold=10):
    # points_sequence: (T, np, 3)
    # Remove points that are not moving and not visible in all frames
    moving_mask = max_distance_moved(points_sequence) > moving_threshold
    return points_sequence[:, moving_mask, :]

def process_sequence(sequence_info):
    data_buffer_path, sequence_path, sample_pt_num, \
        num_samples, model_params, global_index_start, task_name = sequence_info
        
    device = "cuda"
    data_buffer = zarr.open(data_buffer_path, mode="a")
    
    with torch.no_grad():
        grounded_sam_model = GoundedSam2(device=device, **model_params['grounded_sam'])
        tracker_model = PointTracker(device=device, **model_params['tracker'])

        task_videos_rgb = load_video(sequence_path)
        task_videos = [np.transpose(image, (2, 0, 1)) for image in task_videos_rgb]

        initial_frame = task_videos_rgb[0]
        robot_mask = grounded_sam_model.get_arm_segementation_mask(initial_frame)
        robot_mask = get_center_crop_img(robot_mask, crop_size=(240, 200))
        # cv2.imwrite(f"{task_name}_mask_{global_index_start}.png", robot_mask.astype(np.uint8)*255)

        for sample_idx in tqdm(range(num_samples), desc=f"Processing samples for {task_name}", leave=False):
            global_sample_index = global_index_start + sample_idx
            
            point_tracking_sequence_list = []
            
            sample_per_time = 1000
            sample_times = int(sample_pt_num / sample_per_time)
            for _ in range(sample_times):
                samples_robot = sample_from_mask(robot_mask, int(sample_per_time))

                # (time, x, y)
                points_queries = np.zeros((samples_robot.shape[0], 3))
                points_queries[:, 1:] = samples_robot

                point_tracking_sequence = tracker_model.track(task_videos, points_queries)
                point_tracking_sequence = point_tracking_sequence.squeeze(0).cpu().numpy()
                point_tracking_sequence_list.append(point_tracking_sequence)
                
                
            point_tracking_sequence = np.concatenate(point_tracking_sequence_list, axis=1)
            
            # # Sorting by first frame X, Y coordinates
            first_frame_points = point_tracking_sequence[0, :, :2] 
            sorted_indices = np.argsort(first_frame_points[:, 0])   # Sorted by X 
            sorted_indices = sorted_indices[np.argsort(first_frame_points[sorted_indices, 1])]  # Sorted by Y 
            point_tracking_sequence = point_tracking_sequence[:, sorted_indices] # (T, N, 3)

            
            task_name_for_saving = task_name.replace(" ", "-")
            data_buffer[f"{task_name_for_saving}/episode_{global_sample_index}/point_tracking_sequence"] = point_tracking_sequence
            data_buffer[f"{task_name_for_saving}/episode_{global_sample_index}/rgb_arr"] = task_videos_rgb
            
            data_buffer.create_dataset(
                f"{task_name_for_saving}/episode_{global_sample_index}/task_description",
                shape=(1,), 
                dtype=object,
                object_codec=numcodecs.VLenUTF8(),
                overwrite=True
            )

            data_buffer[f"{task_name_for_saving}/episode_{global_sample_index}/task_description"][0] = task_name
            
        print(f"Processed samples for {task_name} finished.")
import torch
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path

import os

from PIL import Image
import numpy as np

class GoundedSam2:
    def __init__(self, device):
        
        self.text_prompt = "robotic arm."
        self.sam2_checkpoint = "sam_and_track/checkpoints/sam2.1_hiera_large.pt"
        self.sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.grounding_dino_config = "sam_and_track/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounding_dino_checkpoint = "sam_and_track/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        
        self.device = device
        self.sam2_model, self.grounding_model = self.load_sam_dino_model(device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def modify_text_prompt(self, text):
        self.text_prompt = text

    def load_sam_dino_model(self, device):
        sam2_checkpoint = self.sam2_checkpoint
        model_cfg = self.sam2_model_config
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        

        # build grounding dino model
        grounding_model = load_model(
            model_config_path=self.grounding_dino_config, 
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=device
        )
        return sam2_model, grounding_model

    def process_image(self, img):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(img).convert("RGB")
        image_transformed, _ = transform(image_source, None)
        return image_transformed
    
    def get_arm_segementation_mask(self, img):

        text = self.text_prompt
        h, w, _ = img.shape
        
        original_img = img
        
        img = self.process_image(img)

        self.sam2_predictor.set_image(original_img)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=img,
            caption=text,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold ,
        )

        # process the box prompt for SAM 2
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks[np.argmax(scores)]
    
    
    
class PointTracker:
    def __init__(self, path='sam_and_track/co-tracker/checkpoints/scaled_offline.pth', device='cuda'):
        self.model = CoTrackerPredictor(
        checkpoint=os.path.join(
            path
        )
    )
        self.model = self.model.to(device)
        self.device = device
        
        
    def track(self, frames, queries, save_track=False, task_name='', sequence_id=0):
        video = torch.tensor(frames)[None].float().to(self.device)
        queries = torch.tensor(queries).float().to(self.device)
        
        # pred_tracks (batch, time, points, 2), pred_visibility (batch, time, points)
        pred_tracks, pred_visibility = self.model(video, queries=queries[None])

        # Save tracks for debug
        if save_track:
            vis = Visualizer(
                save_dir=f'./tracks/{task_name}',
                linewidth=2,
                mode='cool',
                tracks_leave_trace=-1,
            )
            vis.visualize(
                video=video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename=str(sequence_id),)
        
        pred_tracks = torch.concatenate([pred_tracks, pred_visibility[..., None]], dim=-1)    
        return pred_tracks
        
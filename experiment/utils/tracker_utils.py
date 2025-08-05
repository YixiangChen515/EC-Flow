from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
import os
import torch

class PointTracker:
    def __init__(self, path='../sam_and_track/co-tracker/checkpoints/scaled_offline.pth', frames=[]):
        self.model = CoTrackerPredictor(
        checkpoint=os.path.join(
            path
        )
    )
        self.model.eval()
        self.model = self.model.cuda()
        self.device = 'cuda'
        self.video = torch.tensor(frames)[None].float().to(self.device)
        
    def track(self, queries):
        queries = torch.tensor(queries).float().to(self.device)
            
        pred_tracks, pred_visibility = self.model(self.video, queries=queries[None])
        
        # from cotracker.utils.visualizer import Visualizer

        # vis = Visualizer(
        #     save_dir='./',
        #     linewidth=2,
        #     mode='cool',
        #     tracks_leave_trace=-1
        # )
        # vis.visualize(
        #     video=self.video,
        #     tracks=pred_tracks,
        #     visibility=pred_visibility,
        #     filename='queries')
        
        # point of the next moment
        pred_tracks = pred_tracks.squeeze(0)[1].cpu().numpy()
        return pred_tracks

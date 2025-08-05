import torch
import torch.nn.functional as F
from unimatch.unimatch import UniMatch
from argparse import ArgumentParser


def get_flow_model():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='flow_pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    parser.add_argument('--feature_channels', type=int, default=128)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--upsample_factor', type=int, default=4)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--ffn_dim_expansion', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--reg_refine', type=bool, default=True)
    parser.add_argument('--task', type=str, default='flow')
    args = parser.parse_args(args=[])
    DEVICE = 'cuda:0'

    model = UniMatch(feature_channels=args.feature_channels,
                        num_scales=args.num_scales,
                        upsample_factor=args.upsample_factor,
                        num_head=args.num_head,
                        ffn_dim_expansion=args.ffn_dim_expansion,
                        num_transformer_layers=args.num_transformer_layers,
                        reg_refine=args.reg_refine,
                        task=args.task).to(DEVICE)

    checkpoint = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

    model.to(DEVICE)
    model.eval()
    model._requires_grad = False
    return model


### predict per frame flow   
def pred_flow_frame(model, frames, stride=1, device='cuda:0'):
    DEVICE = device 
    model = model.to(DEVICE)
    frames = torch.from_numpy(frames).float()
    images1 = frames[:-1]
    images2 = frames[1:]
    flows = []

    # print("starting prediction")
    # t0 = time.time()
    for image1, image2 in zip(images1, images2):
        image1, image2 = image1.unsqueeze(0).to(DEVICE), image2.unsqueeze(0).to(DEVICE)
    
        # nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
        #                     int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        ### dumb upsampling to (480, 640)
        nearest_size = [480, 640]
        inference_size = nearest_size
        ori_size = image1.shape[-2:]
        
        # print("inference_size", inference_size)
        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
        with torch.no_grad():
            results_dict = model(image1, image2,
                attn_type='swin',
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task='flow',
                pred_bidir_flow=True,
            )
        
        flow_pr = results_dict['flow_preds'][-1]
        
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
                
        flows += [flow_pr[0:1].permute(0, 2, 3, 1).cpu()]
        
    flows = torch.cat(flows, dim=0)
    flows = flows.numpy()

    return flows
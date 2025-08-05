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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp


import numpy as np
from collections import OrderedDict

from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random 
import cv2

import os
from model import DiT_models

from diffusion import create_diffusion

from dataset import TrackConditionedDataset


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def get_buffer_size(root):
    keys = list(root.group_keys())
    return len(keys)


#################################################################################
#                                  Training Loop                                #
#################################################################################



def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."


    # Setup DDP:
    dist.init_process_group("nccl")

    #dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        
        exp_details = 'trackexp' 
        
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}--{exp_details}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Flow model: {args.model}, Image model: {args.img_model}")
    else:
        logger = create_logger(None)

    # Create model:
    # parameters
    pred_horizon = 8
    
    flow_pred_model = DiT_models[args.model](horizon=pred_horizon)
    img_pred_model = DiT_models[args.img_model](input_size=128, num_pt=400, pred_horizon=pred_horizon)
    
    if args.resume:
        ckpt_path = args.ckpt 
        state_dict = find_model(ckpt_path)
        flow_pred_model.load_state_dict(state_dict)
        logger.info(f"resuming from ckpt {ckpt_path}")


    # Note that parameter initialization is done within the DiT constructor
    flow_ema = deepcopy(flow_pred_model).to(device)  # Create an EMA of the model for use after training
    goal_img_ema = deepcopy(img_pred_model).to(device)
    
    requires_grad(flow_ema, False)
    requires_grad(goal_img_ema, False)
    
    flow_pred_model = DDP(flow_pred_model.to(device), device_ids=[rank])
    img_pred_model = DDP(img_pred_model.to(device), device_ids=[rank])
    
    diffusion_flow = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    diffusion_goal_img = create_diffusion(timestep_respacing="", goal="img_pred")

    logger.info(f"DiT Flow Parameters: {sum(p.numel() for p in flow_pred_model.parameters()):,}")
    logger.info(f"DiT Goal Img Parameters: {sum(p.numel() for p in img_pred_model.parameters()):,}")

        
    opt_flow = torch.optim.AdamW(flow_pred_model.parameters(), lr=5e-5, weight_decay=0)
    opt_goal_img = torch.optim.AdamW(img_pred_model.parameters(), lr=1e-4, weight_decay=0)

    ############################## Setup data BEGIN ----------------------------

    # create dataset from file
    dataset = TrackConditionedDataset(
        args=args,
        n_frames=pred_horizon,
        device=device,
        # dataset="real_world",
    )

    ############################## Setup data END ----------------------------

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    logger.info(f"Dataset contains {len(dataset):,} examples")

    # Prepare models for training:
    update_ema(flow_ema, flow_pred_model.module, decay=0)  # Ensure EMA is initialized with synced weights
    flow_pred_model.train()  # important! This enables embedding dropout for classifier-free guidance
    flow_ema.eval()  # EMA model should always be in eval mode
    
    update_ema(goal_img_ema, img_pred_model.module, decay=0)  # Ensure EMA is initialized with synced weights
    img_pred_model.train()  # important! This enables embedding dropout for classifier-free guidance
    goal_img_ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss_flow = 0
    running_loss_goal_img = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Beginning epoch {epoch}...")

        for nbatch in loader:
            init_img = nbatch['init_img'].to(device)  # [Batch, T, 3, H, W]
            flow = nbatch['flow'].to(device) # [Batch, T, 3, num_points]
            goal_img = nbatch['goal_img'].to(device)  # [Batch, 3, H, W]
            
            lang = nbatch['text'].to(device)   # [Batch, 1024]
            num_points = flow.shape[-1]

            flow = flow.reshape(flow.shape[0], -1, num_points) # [Batch, 3*T, num_points]
            
            flow_noise = torch.randn_like(flow)
            flow_noise[:, :3, :] = 0    ## do not add noise to the first step
            
            t = torch.randint(0, diffusion_flow.num_timesteps, (flow.shape[0],), device=device)
            model_kwargs = dict(y=init_img, lang=lang)

            flow_loss_dict = diffusion_flow.training_losses(flow_pred_model, flow, t, model_kwargs, noise=flow_noise, point_conditioned=True)
            flow_loss = flow_loss_dict["loss"].mean()
            
            goal_img_noise = torch.randn_like(goal_img)
            model_kwargs = dict(y=init_img, lang=lang, pt_track_noise=diffusion_flow.noise_for_flow_pred)
            goal_img_loss_dict = diffusion_goal_img.training_losses(img_pred_model, goal_img, t, model_kwargs, noise=goal_img_noise)
            goal_img_loss = goal_img_loss_dict["loss"].mean()
            
            loss = flow_loss + goal_img_loss * 0.4
            
            opt_flow.zero_grad()
            opt_goal_img.zero_grad()
            
            loss.backward()
            
            opt_flow.step()
            opt_goal_img.step()
            
            update_ema(flow_ema, flow_pred_model.module)
            update_ema(goal_img_ema, img_pred_model.module)
            
            # Log loss values:
            running_loss_flow += flow_loss.detach().clone().item()
            running_loss_goal_img += goal_img_loss.detach().clone().item()
            
            log_steps += 1

            
        if epoch % args.log_every == 0 and epoch > 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            epoch_time = (end_time - start_time) / args.log_every
            # Reduce loss history over all processes:
            avg_loss_flow = torch.tensor(running_loss_flow / log_steps, device=device)
            dist.all_reduce(avg_loss_flow, op=dist.ReduceOp.SUM)
            avg_loss_flow = avg_loss_flow.detach().clone().item() / dist.get_world_size()
            
            avg_loss_goal_img = torch.tensor(running_loss_goal_img / log_steps, device=device)
            dist.all_reduce(avg_loss_goal_img, op=dist.ReduceOp.SUM)
            avg_loss_goal_img = avg_loss_goal_img.detach().clone().item() / dist.get_world_size()
            
            logger.info(f"(epoch={epoch:07d}) Train Loss Flow: {avg_loss_flow:.4f}, Train Loss Goal Img: {avg_loss_goal_img:.4f}, Epoch Sec: {epoch_time:.2f}")
            # Reset monitoring variables:
            running_loss_flow = 0
            running_loss_goal_img = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if epoch % args.ckpt_every == 0 and epoch > 0:
            if rank == 0:
                checkpoint = {
                    "model": flow_pred_model.module.state_dict(),
                    "ema": flow_ema.state_dict(),
                    "opt_flow": opt_flow.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/flow_{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                checkpoint = {
                    "model": img_pred_model.module.state_dict(),
                    "ema": goal_img_ema.state_dict(),
                    "opt_goal_img": opt_goal_img.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/goal_img_{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    flow_pred_model.eval()  # important! This disables randomized embedding dropout
    goal_img_ema.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="checkpoints")
    parser.add_argument("--model", type=str,  default="DiT-XL-NoPosEmb-Lang")
    parser.add_argument("--img-model", type=str,  default="DiT-S/8")
    parser.add_argument("--epochs", type=int, default=5001)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--resume", type=bool, default=False,help="whether to resume from a ckpt")
    parser.add_argument("--ckpt", type=str, help="Optional path to a DiT checkpoint  to resume trainining is needed")


    args = parser.parse_args()
    

    main(args)



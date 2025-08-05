#!/bin/bash

# Pip install required packages
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install imageio[ffmpeg]
pip install --upgrade diffusers[torch]
pip install matplotlib flow_vis tqdm tensorboard accelerate timm

cd sam_and_track/sam/
pip install -e .

cd ../grounding_dino/
pip install -r requirements.txt
pip install -e .

cd ../co-tracker/
pip install -e .

cd ../../
pip install -e ./metaworld

pip install einops einops_exts ema_pytorch pytorch_fid pynvml ai2thor cython zarr iopath hydra-core
# pip install hydra-core==1.3.2
pip install git+https://github.com/openai/CLIP.git
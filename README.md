<div align="center">
<img src=assets/logo.svg width="50%"/>
</div>

# EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Embodiment-Centric Flow


<p align="center"><strong>ICCV 2025</strong></p>

[\[🏠Project Page\]](https://ec-flow1.github.io/)  [\[📄Paper\]](https://arxiv.org/abs/2507.06224) [\[📊Dataset\]](https://huggingface.co/datasets/YixiangChen/EC-Flow-MetaWorld)  [\[🤗Checkpoints\]](https://huggingface.co/YixiangChen/EC-Flow)

**TL;DR**: A method for learning robotic manipulation policies solely from action-unlabeled videos, enabling versatile control over deformable objects, occluded environments, and non-object-displacement tasks.
<div align="center">
<img src=assets/network.png "/>
</div>





## Installation
```sh
# Clone the repository
git clone https://github.com/YixiangChen515/EC-Flow.git
cd EC-Flow

# Download pretrained checkpoints (SAM, GroundingDINO, Co-Tracker)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O sam_and_track/checkpoints/sam2.1_hiera_large.pt
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O sam_and_track/gdino_checkpoints/groundingdino_swint_ogc.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -O sam_and_track/co-tracker/checkpoints/scaled_offline.pth

# Create conda environment
conda create -n ecflow python=3.8
conda activate ecflow

# Install dependencies
bash install.sh
```

## Flow Prediction
### 1. Training
We provide the Meta-World dataset in our [Huggingface repo](https://huggingface.co/datasets/YixiangChen/EC-Flow-MetaWorld). Please download the dataset and place it under the `data` directory. 

There are two ways to prepare the training data:

1. Use the `metaworld.tar.gz` file, which contains the pre-processed dataset with ground-truth point tracking results. This version is ready for training out of the box.

2. Alternatively, you can start with the original Meta-World dataset by using `metaworld_original.tar.gz`. To generate the processed dataset from it, run:
```sh
python -m data_gen.gen_metaworld_all
```

Once the dataset is prepared, you can start training the flow prediction module by running:
```sh
# Note: The global batch size should be divisible by the number of devices. We trained on 8 NVIDIA RTX 4090 GPUs (24GB) with a batch size of 7 per GPU.
torchrun --nnodes=1 --nproc_per_node=8 train.py --results-dir ckpt --global-batch-size=56 --data-path=data/metaworld
```  

### 2. Inference
You can download the pretrained checkpoints from our [Huggingface repo](https://huggingface.co/YixiangChen/EC-Flow) and and place them in the `ckpt` directory. To evaluate both the flow prediction and goal image prediction results, run the following command:
```sh
python inference.py --ckpt ckpt/flow.pt --img-ckpt ckpt/goal_img.pt
```

## Evaluation in Meta-World 
To evaluate **EC-Flow** in the Meta-World environment, follow these steps:
1. Download the pretrained checkpoints as described above.

2. Apply the necessary environment modifications by following the instructions in [modify_env.md](./experiment/modify_env.md) (**IMPORTANT**).

Once the setup is complete, run the following command to start evaluation:
```sh
cd experiment
bash eval_policy.sh
```

**Note**:
To speed up the evaluation process, you can use multiple GPUs by specifying the device IDs:
```sh
# Example Usage
bash eval_policy.sh "0,1,2,3"
```

## License

This repository is released under the MIT license.

## Acknowledgement

We extend our deepest thanks to the creators of these remarkable projects:
- [Track-2-Act](https://github.com/homangab/Track-2-Act)
- [AVDC](https://github.com/flow-diffusion/AVDC)
- [Im2Flow2Act](https://github.com/real-stanford/im2Flow2Act)
- [DiT](https://github.com/facebookresearch/DiT)
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [SAM2](https://github.com/facebookresearch/sam2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Co-Tracker](https://github.com/facebookresearch/co-tracker)
- [Meta-World](https://github.com/Farama-Foundation/Metaworld)

## Contact
If you have any questions about the code, please contact `yixiang.chen [AT] cripac.ia.ac.cn`

## Citation

Please consider citing **EC-Flow** if it benefits your research:
```
@article{chen2025ec,
  title={EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Embodiment-Centric Flow},
  author={Chen, Yixiang and Li, Peiyan and Huang, Yan and Yang, Jiabing and Chen, Kehan and Wang, Liang},
  journal={arXiv preprint arXiv:2507.06224},
  year={2025}
}
```

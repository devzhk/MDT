#! /bin/bash
pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate blobfile mpi4py
pip install git+https://github.com/huggingface/pytorch-image-models.git
pip install git+https://github.com/huggingface/diffusers
export OPENAI_LOGDIR=output_mdt_xl2-256
torchrun --nproc_per_node=8 image_train.py --image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDT_XL_2 --diffusion_steps 1000 --batch_size 128 --microbatch 32 --log_interval 1000
"""
Train a diffusion model on images.
"""

import argparse

from masked_diffusion import dist_util, logger
from masked_diffusion.image_datasets import load_data
from masked_diffusion.resample import create_named_schedule_sampler
from masked_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from masked_diffusion.train_util import TrainLoop
from masked_diffusion import create_diffusion, model_and_diffusion_defaults, diffusion_defaults
import masked_diffusion.models as models_mdt

from masked_diffusion.latent_datasets import ImageNetLatentDataset

from torch.utils.data import DistributedSampler, DataLoader
from omegaconf import OmegaConf

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    args = create_argparser().parse_args()
    config = OmegaConf.load(args.config)

    dist_util.setup_dist_multinode(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    configs = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(configs)
    print(args)
    image_size = configs['image_size']
    latent_size = image_size // 8
    model = models_mdt.__dict__[args.model](input_size=latent_size, mask_ratio=args.mask_ratio, decode_layer=args.decode_layer)
    print(model)
    diffusion = create_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # Setup dataset
    dataset = ImageNetLatentDataset(
        config.data.root, resolution=config.data.resolution, 
        num_channels=config.data.num_channels)
    
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True,
        drop_last=True
    )
    data = sample_data(loader)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model="MDT_S_2",
        mask_ratio=None,
        decode_layer=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument(
        "--rank", default=0, type=int, help="""rank for distrbuted training."""
    )
    parser.add_argument("--config", type=str, help="Config file.", default="configs/imagenet.yaml")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

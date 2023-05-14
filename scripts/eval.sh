pip install torch torchvision torchaudio
pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate blobfile mpi4py
pip install git+https://github.com/huggingface/pytorch-image-models.git
pip install diffusers["torch"]

MODEL_FLAGS="--image_size 256 --model MDT_XL_2 --decode_layer 2"
DIFFUSION_FLAGS="--num_sampling_steps 250 --num_samples 50000 --batch_size 50"

NUM_GPUS=8

for ((i=20000; i< 230000; i+=20000))
do
   printf -v ckpt_id "%06d" $i
   MODEL_PATH=output_mdt_xl2/ema_0.9999_$ckpt_id.pt
   echo $MODEL_PATH
   export OPENAI_LOGDIR=output_mdt_xl2_eval/$ckpt_id
   echo $OPENAI_LOGDIR
   torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
   python3 evaluator.py ../assets/fid_stats/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz
done


for ((i=20000; i< 180000; i+=20000))
do
   printf -v ckpt_id "%06d" $i
   MODEL_PATH=output_mdt_xl2-256/ema_0.9999_$ckpt_id.pt
   echo $MODEL_PATH
   export OPENAI_LOGDIR=output_mdt_xl2_eval-256/$ckpt_id
   echo $OPENAI_LOGDIR
   torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
   python3 evaluator.py ../assets/fid_stats/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz
done

#!/bin/bash -x
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip
#SBATCH --partition scavenge
#SBATCH --constraint=volta32gb
#SBATCH --time=2-00:00:00 # Time limit in the form HH:MM:SS

eval "$(/path/to/conda/bin/conda shell.bash hook)" # init conda
conda activate pytorch
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd ~/workspace/github/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

srun --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --name "mobilenetv2_050" \
    --resume 'latest' \
    --train-data="/datasets01/metaclip_csam_filtered_and_face_blurred/*/*.tar" \
    --train-num-samples 6000000 \
    --dataset-type webdataset \
    --warmup 10000 \
    --batch-size=384 \
    --epochs=200 \
    --dataset-resampled \
    --grad-clip-norm 5.0 \
    --lr 1e-3 \
    --workers=6 \
    --model "mobilenetv2_050" \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --aug-cfg use_timm=True scale='(0.33, 1.0)' re_prob=0.35 \
    --report-to wandb --wandb-notes mobilenetv2_050 --wandb-project-name open_clip \
    --distill-model convnext_xxlarge --distill-pretrained laion2B-s34B-b82K-augreg-soup
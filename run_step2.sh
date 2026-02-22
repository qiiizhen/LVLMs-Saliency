#!/bin/bash
#SBATCH --job-name=llava_step2
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=rc_classwork17_pi
#SBATCH --qos=gpu_access
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err

module load cuda/12.4

cd /work/users/z/h/zhenqi/LVLMs-Saliency

export PYTHONPATH=/work/users/z/h/zhenqi/LVLMs-Saliency/src/transformers/src:/work/users/z/h/zhenqi/LVLMs-Saliency/src/LLaVA:$PYTHONPATH

/nas/longleaf/home/zhenqi/.conda/envs/lvlm-saliency/bin/python demo_step2_llava.py \
    --model-path /work/users/z/h/zhenqi/models/llava-v1.5-7b

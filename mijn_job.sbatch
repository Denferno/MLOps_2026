#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2   
#SBATCH --mem=16G           
#SBATCH --gpus=1           
#SBATCH --partition=gpu_course
#SBATCH --time=02:00:00
#SBATCH --output=traineroutput_%j.out

cd MLOps_2026
source my_venv/bin/activate
python experiments/train.py --config experiments/configs/train_config.yaml

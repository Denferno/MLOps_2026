#!/bin/bash
#SBATCH --job-name=benchmark_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=    2   
#SBATCH --mem=16G           
#SBATCH --gpus=1           
#SBATCH --partition=gpu_course
#SBATCH --time=02:00:00
#SBATCH --output=bench_gpu_%j.out

cd $SLURM_SUBMIT_DIR
source $HOME/my_venv/bin/activate
python scripts/benchmark.py
echo "nvidia-smi (before)"
nvidia-smi
PYTHONPATH=$SLURM_SUBMIT_DIR/src:$PYTHONPATH
python scripts/benchmark.py
echo "nvidia-smi (after)"
nvidia-smi
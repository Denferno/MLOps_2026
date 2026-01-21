# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

For the course ML programming and ML OPs, provided by UvA.

Contributors:
Dennis Chan 15833526
Lawrence Lam 15277844
Ngawang Tsarong 15157970
Michael Dong 15804232


Downloading the right data:
1. wget https://surfdrive.surf.nl/public.php/dav/files/wjRYtSborgbPF2P/?accept=zip -O pcam_data.tar src/ml_core/data
2. cd #path
3. unzip pcam_data.tar
4. cd mlops_2026_pcam_data
5. tar -xvf pcam_data.tar
6. cd surfdrive
7. mv camelyonpatch* ..
8. cd ..
9. mv mlops_2026_pcam_data camelyonpatch_level_2
10. Make sure that the data is inside src/ml_core/data/camelyonpatch_level_2


Instruction for running train.py.
1. module purge
2. module load 2025
3. module load Python/3.13.1-GCCcore-14.2.0
4. python -m venv ~/my_venv
5. source ~/my_venv/bin/activate
6. pip install --upgrade pip
7. pip install -r requirements.txt
8. pip install -e .
9. deactivate [optional]
9. sbatch mijn_job.sbatch
10. cat traineroutput_[numbers].out

Installation

 Local 
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
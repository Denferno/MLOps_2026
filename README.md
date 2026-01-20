# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

For the course ML programming and ML OPs, provided by UvA.

Contributors:
Dennis Chan 15833526
Lawrence Lam 15277844
Ngawang Tsarong 15157970
Michael Dong 15804232

Klaar zetten voor het runnen van train.py.
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
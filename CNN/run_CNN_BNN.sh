#!/bin/bash -l
#SBATCH --job-name=CNN+RayBNN   # Name of job
#SBATCH --account=def-taolu    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=1-16:00          # 11 hours
#SBATCH --nodes=1              # 1 node
#SBATCH --ntasks-per-node=6   # Cores per node
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=490G
#SBATCH --mail-user=xuanchen@uvic.ca
#SBATCH --mail-type=ALL

module load StdEnv/2020 gcc/9.3.0 cuda/12.2 openmpi/4.0.3 arrayfire/3.9.0 rust/1.70.0 python/3.11.5 scipy-stack 

# source /scratch/cxyycl/RayBNN_Python-main/venv/bin/activate

# pip install /scratch/cxyycl/raybnn_python-0.1.2-cp311-cp311-linux_x86_64.whl

python /home/cxyycl/scratch/RayBNN_Python-main/Rust_Code/run_network-CNN_BNN_backup.py

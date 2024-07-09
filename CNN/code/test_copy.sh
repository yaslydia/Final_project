#!/bin/bash -l
#SBATCH --job-name=MS-2_0   # Name of job
#SBATCH --account=def-taolu    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=2-21:00          # 11 hours
#SBATCH --nodes=1              # 1 node
#SBATCH --ntasks-per-node=6   # Cores per node
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=300G
#SBATCH --mail-user=xuanchen@uvic.ca
#SBATCH --mail-type=ALL

module load cuda cudnn scipy-stack 

source ~/tensor/bin/activate

pip install /scratch/cxyycl/raybnn_python-0.1.2-cp311-cp311-linux_x86_64.whl

# python /home/cxyycl/scratch/Microsleep-code/code/loadData.py
# python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/myModel.py
# python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/train_test.py
python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/train_feature_part1.py
python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/train_feature_part2.py
python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/train_feature_part3.py
python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/train_feature_part4.py
#python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/predict_val.py
#python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/predict_test.py
python /home/cxyycl/scratch/RayBNN_Python-main/Rust_Code/run_network_RayBNN_SLEEP_copy.py
python /home/cxyycl/scratch/RayBNN_Python-main/Rust_Code/run_network_RayBNN_SLEEP_copy_2-100.py

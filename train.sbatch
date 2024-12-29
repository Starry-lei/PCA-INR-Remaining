#!/bin/bash
#SBATCH --job-name="INR"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8,VRAM:16G
#SBATCH --mem=30G
#SBATCH --exclude=node[11,12]
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:TURING"
#SBATCH --time=36:30:00
#SBATCH --mail-user="ge23yat@mytum.de"
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/stud/chengl/storage/user/PCA-INR-Remaining/Training/slurm_output/slurm-%j.out
#SBATCH --error=/home/stud/chengl/storage/user/PCA-INR-Remaining/Training/slurm_output/slurm_error-%j.err
#SBATCH --partition="NORMAL"


# Start tmux
# tmux new -d -s "test_FM2" #SBATCH --exclude=node[]
# Run tmux commands to setup the session
# tmux split-window -h # this will setup two panes, splitting the screen horizontally

# Finally, make job wait for user to connected####SBATCH --comment="CVPR"
# sleep 24h


# Start tmux
tmux new -d -s "INR"
# Run tmux commands to setup the session
tmux split-window -h # this will setup two panes, splitting the screen horizontally
# srun --jobid 1171741 --pty tmux a -t backrest

nvidia-smi
# Finally, make job wait for user to connected
sleep 24h


# |GPU_GEN:TURING


# nvidia-smi
# srun python train.py --opt options/train/train.yaml 
# python train.py -c cfgs/partNetSSMLeg.yaml
# python train.py -c cfgs/partNetSSMSeat.yaml
# python train.py -c cfgs/partNetSSMVerticalBackPanel.yaml
# python train.py -c cfgs/partNetSSMArmRestBar.yaml
# python train.py -c cfgs/partNetSSMBackRestConnector.yaml
# python train.py -c cfgs/partNetSSMBackRestFrame.yaml
# python train.py -c cfgs/config.yaml


# Comment (Lukas K.)
# For a normal DL task, it is useful to use multiple CPUs per GPU for example --cpus-per-task=3. This depends on how many CPUs your dataloader (pytorch num_workers) can use effectively.
# For a normal DL task, one should use about 32G of memory. Of course, this depends on the task. Try not to take too much, but if the memory is insufficient the task will get killed.
#  sbatch findCorresp.sbatch 


# scancel 1273195
# sbatch train.sbatch
# srun --jobid 1310026 --pty tmux a -t INR
#  squeue -u chengl
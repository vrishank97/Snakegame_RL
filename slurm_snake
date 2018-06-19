#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=1:29:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

nproc
sleep 1
module load cuda75/toolkit/7.5.18
source /home/s.aakhil/snake_env/bin/activate
# python /home/s.aakhil/snakegame/run_DQN.py > /home/s.aakhil/snakegame/output.txt
# module unload cuda80/toolkit/8.0.44
ipython /home/s.aakhil/snakegame/ddqn.py


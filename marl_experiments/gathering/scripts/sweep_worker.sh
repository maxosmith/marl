#!/bin/bash

#SBATCH --output=/home/mxsmith/slurm/slurm-%x-%A_%a.out
#SBATCH --error=/home/mxsmith/slurm/slurm-%x-%A_%a.err
#SBATCH --mail-user=max.olan.smith@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10g
#SBATCH --time=0-05:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

source ~/.bashrc

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/pkgs/arc/python3.9-anaconda/2021.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh" ]; then
        . "/sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh"
    else
        export PATH="/sw/pkgs/arc/python3.9-anaconda/2021.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate model38

python sweep_worker.py --config_dir $1

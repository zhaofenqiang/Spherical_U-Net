#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=50g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 3-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fenqiang@email.unc.edu

module add python
python3 single_random_forest_for_prediction.py


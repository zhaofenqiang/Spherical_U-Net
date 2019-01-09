#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=50g
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 3-
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fenqiang@email.unc.edu

module add python
python3 support_vector_regression_for_prediction.py


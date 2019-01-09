#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=40g
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 5-

module add python
python3 linear_regression_for_prediction.py


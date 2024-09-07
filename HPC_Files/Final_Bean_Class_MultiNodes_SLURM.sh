#!/bin/bash
#SBATCH -N 2 #2 node
#SBATCH --ntasks-per-node=36
#SBATCH --time=30:00
#SBATCH --job-name=Beans_Classification
#SBATCH --error=%J.err_
#SBATCH --output=%J.out_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2 # GPU Cards


python  /home/gashugi/Th/multimirrored_FINAL.py


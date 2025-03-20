#!/bin/bash

#BSUB -J scotland	          # Job name
#BSUB -o output.%J                # Standard output file
#BSUB -e error.%J                 # Standard error file
#BSUB -W 48:00                    # Maximum time (hh:mm)
#BSUB -n 8                        # Number of MPI processes
#BSUB -R "span[hosts=1]"          # Use n cores on 1 node
#BSUB -R "rusage[mem=15GB]"       # Memory requirement (per core)
#BSUB -q cnr                      # The queue to join

# Activate conda environment
source activate /usr/local/usrapps/tcsi/LouisLANDIS/envs/louis_R_env

# Move to the work directory
cd /share/tcsi/lagoodal/Python/

# Run Python script
python scotland.py

# Deactivate conda environment
conda deactivate

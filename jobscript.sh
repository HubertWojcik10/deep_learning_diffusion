#!/bin/bash

### -------------specify job name ----------------
#BSUB -J diffusion          # Job name

### -------------specify output and error log ----------------
#BSUB -o training_output_100-10_epochs.log  # Standard output log
#BSUB -e training_error_100-10_epochs.log   # Standard error log

### -------------specify number of cores ----------------
#BSUB -n 4                    # Number of cores
#BSUB -R "span[hosts=1]"       # Use 1 host

### -------------specify GPU request ----------------
#BSUB -gpu "num=1:mode=exclusive_process"   # Request 1 GPU

### -------------specify memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"    # Memory request

### -------------specify wall-clock time ----------------
#BSUB -W 02:00                 # Wall time (2 hours)


### -------------create and activate virtual environment ----------------
# Create and activate the virtual environment

ENV_NAME=diffusion
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment "
    virtualenv $ENV_NAME
else
    echo "Activating existing virtual environment "
fi

# Activate the virtual environment
source $ENV_NAME/bin/activate

### -------------run the training script ----------------
# Run the training script
python train.py

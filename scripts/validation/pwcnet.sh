#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="saved_check_point/pwcnet"

# datasets
SINTEL_HOME=(YOUR PATH)/MPI-Sintel-complete/

# model and checkpoint
MODEL=PWCNet
CHECKPOINT="$EXPERIMENTS_HOME/PWCNet/checkpoint_best.ckpt"
EVAL_LOSS=MultiScaleEPE_PWC

SIZE_OF_BATCH=1

# validate clean configuration
SAVE_PATH="$EXPERIMENTS_HOME/eval_temp/$MODEL"
python ../../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
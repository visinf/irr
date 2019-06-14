#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="saved_check_point/pwcnet"

# datasets
KITTI_HOME=(YOUR PATH)/KITTI_flow/

# model and checkpoint
MODEL=IRR_PWC
CHECKPOINT="$EXPERIMENTS_HOME/IRR-PWC_kitti/checkpoint_latest.ckpt"
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI

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
--validation_dataset=KittiCombVal  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$KITTI_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
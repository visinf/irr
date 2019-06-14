#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
KITTI_HOME=(YOUR PATH)/KITTI_flow/

# model and checkpoint
MODEL=IRR_PWC
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI
CHECKPOINT="saved_check_point/IRR-PWC_things3d/checkpoint_latest.ckpt"
SIZE_OF_BATCH=4

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-$TIME"

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[616, 819, 1022, 1123, 1149, 1174, 1276, 1377, 1479, 1580]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=3e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=160 \
--total_epochs=710 \
--training_augmentation=RandomAffineFlowOccKITTI \
--training_augmentation_crop="[320,896]" \
--training_dataset=KittiCombFull \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$KITTI_HOME \
--training_dataset_preprocessing_crop=True \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=KittiCombVal  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$KITTI_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
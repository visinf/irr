#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
FLYINGTHINGS_HOME=(YOUR PATH)/things3d/FlyingThings3D_subset/
SINTEL_HOME=(YOUR PATH)/MPI-Sintel-complete/

# model and checkpoint
MODEL=IRR_PWC
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample
CHECKPOINT="saved_check_point/IRR-PWC_flyingchairsOcc/checkpoint_latest.ckpt"
SIZE_OF_BATCH=4

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-$TIME"

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[128, 139, 149]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-5 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=109 \
--total_epochs=159 \
--training_augmentation=RandomAffineFlowOcc \
--training_augmentation_crop="[384,768]" \
--training_dataset=FlyingThings3dCleanTrain \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGTHINGS_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
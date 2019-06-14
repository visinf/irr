#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
SINTEL_HOME=(YOUR PATH)/MPI-Sintel-complete/

# model and checkpoint
MODEL=IRR_PWC
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel
CHECKPOINT="saved_check_point/IRR-PWC_things3d/checkpoint_latest.ckpt"
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
--lr_scheduler_milestones="[245, 284, 322, 342, 346, 351, 370, 390, 409, 428]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1.5e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=160 \
--total_epochs=447 \
--training_augmentation=RandomAffineFlowOccSintel \
--training_augmentation_crop="[384,768]" \
--training_dataset=SintelTrainingCombFull \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$SINTEL_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelTrainingCombValid  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[620, 697, 774, 812, 822, 831, 870, 908, 947, 985]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-05 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=448 \
--total_epochs=591 \
--training_augmentation=RandomAffineFlowOccSintel \
--training_augmentation_crop="[384,768]" \
--training_dataset=SintelTrainingFinalFull \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$SINTEL_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelTrainingFinalValid  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
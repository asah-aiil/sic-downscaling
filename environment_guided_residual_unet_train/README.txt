Environment-guided Residual U-Net

full_10var contains the main ten-channel model and shared training modules.
The single runnable ablation example is ../physical_group_revised_ablation_train/group_HT_tos_t2m/.
All retained training configurations monitor validation loss with early stopping: patience 5, minimum improvement 0.0, maximum 40 epochs. The best checkpoint is saved.

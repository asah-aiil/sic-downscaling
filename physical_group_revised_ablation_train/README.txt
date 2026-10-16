Thermodynamic-group ablation example

Included: group_HT_tos_t2m (TOS + T2M + SIC + mask).
This is one representative runnable configuration, not an archive of every ablation experiment.
It uses the shared full_10var model, data loader, losses and training utilities via its local config.py.
Run from group_HT_tos_t2m with: python main.py
Configure local-path placeholders first and keep the repository structure intact.
Early stopping monitors val_loss: patience=5, min_delta=0.0; maximum 40 epochs.
Other variable groups can be configured by adapting INPUT_VARS and the corresponding channel/branch mappings; maintain consistent model and checkpoint settings.

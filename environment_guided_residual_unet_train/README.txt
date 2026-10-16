Environment-guided Residual U-Net training experiments

Architecture:
- SIC encoder input: coarse SIC + mask.
- Environment encoder input: the selected environmental group.
- Five environment gates: sigmoid(1x1 convolution of environment feature).
- Fusion: SIC feature * (1 + gate).
- Gate convolution starts at zero weights and bias -4, so fusion starts near identity.
- Existing PixelShuffle decoder is retained.
- Output: bilinear coarse SIC + tanh(delta_SIC).

Experiments:
- full_10var: eight environmental variables + SIC + mask.
- group_O_tos: TOS + SIC + mask.
- group_D_wind_msl: U10 + V10 + MSL + SIC + mask.
- group_T_t2m_ssr_tcc: T2M + SSR + TCC + SIC + mask.
- group_P_pr: PR + SIC + mask.

All use the existing T1 cache, fixed year split, RMSE and 40 epochs.
These checkpoints require this model.py for evaluation.

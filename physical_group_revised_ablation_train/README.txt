重新划分物理组的补充实验

新分组：
- 海洋—近地面热力组 HT = tos + t2m
- 辐射—云组 RC = ssr + tcc

训练模型：
- group_HT_tos_t2m：tos + t2m + sic + mask（完整HT组）
- group_HT_t2m_only：t2m + sic + mask（HT去掉tos）
- group_RC_ssr_tcc：ssr + tcc + sic + mask（完整RC组）
- group_RC_tcc_only：tcc + sic + mask（RC去掉ssr）
- group_RC_ssr_only：ssr + sic + mask（RC去掉tcc）

HT去掉t2m后就是已有的envgated_tos模型，不重复训练：
/home/ubuntu/sic_dsc/code/checkpoints/20260910_032335_envgated_tos_sic_mask_year_fixed_sic_residual_2014_2023_T1_d_rmse_fixed/best_model.pth

所有新模型使用2014--2023固定年份划分、T1、SIC残差、RMSE，最多40轮，
val_loss连续5轮不改善早停。每个实验都从现有多变量缓存按通道子集读取，
不需要重建缓存。

部署目录：
/home/ubuntu/sic_dsc/physical_group_revised_ablation_train

双卡启动：
cd /home/ubuntu/sic_dsc/physical_group_revised_ablation_train
chmod +x run_two_gpus.sh
nohup bash run_two_gpus.sh 0 1 > run_two_gpus.log 2>&1 &

GPU 0依次训练两个完整物理组；GPU 1依次训练三个单变量对照。
训练结束后，再将已有统一评估脚本的MODEL_VARS和checkpoint列表补上这些模型，
统一计算航道尺度的RMSE、MAE、Bias、SSIM和冰区分类指标。

四组 mask 消融实验

full10_mask     : 10变量 + SIC + mask
full10_nomask   : 10变量 + SIC，不输入 mask
sicmask_mask    : SIC + mask 基线
sicmask_nomask  : 仅 SIC 基线，不输入 mask

已有的 full10_mask 和 sicmask_mask 模型直接复用，不再重复训练。
本次只训练两个无 mask 模型：full10_nomask 和 sicmask_nomask。

两张 GPU 并行运行：
chmod +x run_two_gpus.sh
nohup bash run_two_gpus.sh 0 1 > run_two_gpus.log 2>&1 &
tail -f run_two_gpus.log

GPU0 训练 full10_nomask；
GPU1 训练 sicmask_nomask。
四组使用相同年份划分、缓存、RMSE、残差输出和早停设置。

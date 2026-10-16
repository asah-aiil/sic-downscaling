import os
import torch


class Config:
    # =========================================================
    # 运行环境
    # =========================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = True

    XR_ENGINE = None

    # =========================================================
    # 数据路径
    # =========================================================
    DATA_ROOT = "local-path/data/dataset"
    COARSE_DIR = os.path.join(DATA_ROOT, "input")
    HIGHRES_DIR = os.path.join(DATA_ROOT, "target")

    COARSE_FILE_TMPL = "input_25km_{year}_25km.nc"
    HR_FILE_TMPL = "Bremen_{year}.nc"

    # =========================================================
    # 年份范围：2014–2023 内部训练/验证/测试集
    # =========================================================
    YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    REQUIRE_ALL_YEARS = True

    # =========================================================
    # 输入变量：本物理组 + SIC；另添加 mask
    # =========================================================
    # 这里的顺序就是模型通道顺序，请勿在推理时改变。
    INPUT_VARS = ["tos","sic"]

    # 逻辑特征 -> NetCDF 中允许的源变量名。
    # 旧训练文件常用 tp/sst/sic，CMIP 文件常用 pr/tos/siconc。
    INPUT_VAR_ALIASES = {
        "t2m": ["t2m"],
        "u10": ["u10"],
        "v10": ["v10"],
        "msl": ["msl"],
        "pr": ["pr", "tp"],
        "ssr": ["ssr"],
        "tcc": ["tcc"],
        "tos": ["tos", "sst"],
        "sic": ["sic", "siconc"],
    }

    # 与现有十通道缓存的构建范围保持一致。
    FEATURE_NORM_RANGES = {
        "t2m": (220.0, 320.0),
        "u10": (-30.0, 30.0),
        "v10": (-30.0, 30.0),
        "msl": (97000.0, 105000.0),
        "pr": (0.0, 0.05),
        "ssr": (0.0, 1.0e7),
        "tcc": (0.0, 1.0),
        "tos": (270.0, 310.0),
        "sic": (0.0, 1.0),
    }
    # Bremen 高分辨率 SIC 变量名
    SIC_HR_VAR = "z"

    # SIC 值域
    SIC_DATA_RANGE = 1.0

    # =========================================================
    # 时间窗口配置
    # =========================================================
    # 单日多变量：只使用目标日 d
    TIME_OFFSETS = [0]
    TEMPORAL_OFFSETS = TIME_OFFSETS

    # 是否加入目标日 mask_d
    ADD_MASK_CHANNEL = True

    # 输入通道数由所选变量及 mask 自动计算
    INPUT_CHANNELS = (
        len(INPUT_VARS) * len(TIME_OFFSETS) + (1 if ADD_MASK_CHANNEL else 0)
    )
    OUTPUT_CHANNELS = 1

    # =========================================================
    # 直接复用现有 T1 十通道缓存，不需要重建。
    # =========================================================
    TEMPORAL_CACHE_DIR = "local-path/data/cache_T1_multivar_float16_2014_2023"

    # 只有需要从原始 NetCDF 重建独立四通道缓存时才使用这个目录。
    CACHE_BUILD_DIR = "local-path/data/cache_T1_pr_tos_sic"
    TEMPORAL_CACHE_DTYPE = "float16"

    # memmap 读取配置
    CACHE_RETURN_FLOAT32 = False
    CACHE_STRICT_META = True
    # 按变量别名从十通道缓存抽取所需通道。
    CACHE_ALLOW_CHANNEL_SUBSET = True
    # 旧缓存 meta 没有 complete/feature_norm_ranges 字段，但备份构建脚本已确认
    # tp、sst、sic 分别按 (0,0.05)、(270,310)、(0,1) 归一化。
    CACHE_ALLOW_LEGACY_META = True
    CACHE_TRAIN_SHUFFLE = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2

    # =========================================================
    # DataLoader 配置
    # =========================================================
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    PIN_MEMORY = True

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    # 按完整年份划分，避免相邻日期跨训练/验证/测试集。
    # 年份列表由 np.random.default_rng(42) 对 2014–2023 随机排列一次后固定：
    # [2019, 2020, 2014, 2021, 2017, 2016, 2018, 2023, 2015, 2022]
    SPLIT_MODE = "year_fixed"
    SPLIT_SEED = 42
    TRAIN_YEARS = [2014, 2016, 2017, 2019, 2020, 2021]
    VAL_YEARS = [2018, 2023]
    TEST_YEARS = [2015, 2022]

    ICE_MASK_THR = 0.15

    # =========================================================
    # 模型输出：显式 SIC 基线残差
    # =========================================================
    # sic_residual:
    #   prediction = bilinear_upsample(coarse_sic) + tanh(delta)
    # 网络最后一层零初始化，因此训练开始时输出严格等于 SIC 双线性基线。
    ARCHITECTURE_NAME = "environment_guided_residual_unet"
    ENV_VARS = INPUT_VARS[:-1]
    ENV_CHANNEL_INDICES = list(range(len(ENV_VARS)))
    MASK_CHANNEL_INDEX = INPUT_CHANNELS - 1
    ENV_GATE_INITIAL_BIAS = -4.0
    ENV_GATE_SCALE = 1.0

    MODEL_OUTPUT_MODE = "sic_residual"
    # SIC 在当前输入中的位置自动计算。
    SIC_CHANNEL_INDEX = INPUT_VARS.index("sic")
    SIC_BASELINE_INTERPOLATION = "bilinear"
    SIC_RESIDUAL_SCALE = 1.0
    SIC_RESIDUAL_ZERO_INIT = True

    # =========================================================
    # 训练超参
    # =========================================================
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0

    ACCUM_STEPS = 1
    WARMUP_EPOCHS = 0
    VIS_EPOCH_INTERVAL = 10

    # =========================================================
    # 损失配置
    # =========================================================
    LOSS_NAME = "rmse"
    LOSS_LEARNABLE_WEIGHTS = False

    LOSS_WEIGHT_INIT_BIAS_TO_RMSE = True

    SSIM_WIN_SIZE = 11
    SSIM_WIN_SIGMA = 1.5

    SSIM_WINDOW = SSIM_WIN_SIZE
    SSIM_SIGMA = SSIM_WIN_SIGMA

    LOSS_WEIGHT_LR_MULT = 0.1
    LOSS_WEIGHT_WD = 0.0
    LOSS_WEIGHT_FREEZE_EPOCHS = 0

    SMOOTHL1_BETA = 0.1

    # =========================================================
    # 验证指标配置
    # =========================================================
    FSS_THRESHOLD = 0.15
    FSS_RADII = [3, 5, 9, 17]

    ICE_EDGE_THRESHOLD = 0.15
    ICE_EDGE_THR = ICE_EDGE_THRESHOLD

    PIXEL_SIZE_KM = 6.25
    ICE_EDGE_PIXEL_KM = 6.25

    # =========================================================
    # 断点续训
    # =========================================================
    RESUME_TRAINING = False

    # =========================================================
    # 输出目录
    # =========================================================
    PROJECT_ROOT = "local-path/code"
    CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")

    EXP_TAG = "tenvar_mask_yearsplit_sicresidual_rmse"

    BEST_MODEL_PATH = os.path.join(CHECKPOINT_ROOT, EXP_TAG, "best_model.pth")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, EXP_TAG, "last_checkpoint.pth")

    LOG_DIR = os.path.join(CHECKPOINT_ROOT, EXP_TAG, "logs")
    RESULT_DIR = os.path.join(CHECKPOINT_ROOT, EXP_TAG, "results")

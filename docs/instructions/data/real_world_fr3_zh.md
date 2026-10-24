# FR3 真机数据接入 FastWAM-IDM

[English](real_world_fr3.md) | [数据索引](README_zh.md) | [FastWAM 配置](../config/fastwam_zh.md)

`real_world_fr3_fastwam_idm_wan22` 配方使用左相机、腕部相机、7 维关节位置和 1 维夹爪的 FR3 数据，训练 EasyWAM 中的 Wan2.2 FastWAM-IDM。默认保留 10 Hz 数据和 16 步动作 chunk。这是监督训练数据入口；真机部署、现有 LIBERO BC/RL 管线属于另外的接入工作。

以下命令从 `EasyWAM/` 目录执行。一键 shell 入口自动使用已准备好的 `.venv` 并设置源码导入路径，无需额外激活环境；单独运行其他 Python 脚本时仍应使用项目环境。

## 本机训练环境

本机使用 `EasyWAM/.venv`，基于已有 `conda/envs/troncamp_env` 的 Python 3.10.20 创建，`include-system-site-packages=false`，不会继承或修改该 Conda 环境的训练依赖。使用前激活：

```bash
cd /path/to/storage/Outer/EasyWAM
source .venv/bin/activate
bash scripts/run_real_world_fr3.sh config
```

依赖按 `pyproject.toml` 固定版本安装，包括 PyTorch `2.7.1+cu128`、Torchvision `0.22.1+cu128`、Accelerate `1.5.2`、DeepSpeed `0.16.9`、Transformers `4.57.0`、Datasets `3.6.0` 和 PyAV `16.0.1`；EasyWAM 以 editable 方式安装。另安装 `pytest` 运行 CPU 回归。FlashAttention 未安装，默认 `attention_backend=auto` 使用项目支持的 PyTorch SDPA 回退。

这套环境包含 CUDA 12.8 运行库；没有另行安装 CUDA 编译器或预编译 DeepSpeed 扩展。当前 ZeRO-1 配方使用 PyTorch fused AdamW，未启用 optimizer offload，不要求安装全部 DeepSpeed 自定义算子。今后若切换到需要编译扩展的功能，应另外配置匹配的 CUDA toolkit；本机 `/usr/local/cuda` 指向的 12.0 不能视为已配置好 12.8 编译环境。

复建环境时可使用已有的 `uv` 和 Python 3.10；先按本机集群约定配置网络，再在 EasyWAM 根目录运行：

```bash
uv venv --python /absolute/path/to/python3.10 .venv
uv pip install --python .venv/bin/python pip setuptools wheel ninja pytest
DS_BUILD_OPS=0 UV_LINK_MODE=copy uv pip install --python .venv/bin/python \
  --index https://pypi.org/simple \
  --default-index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -e .
.venv/bin/python -m pip check
```

上述索引顺序让同版本普通依赖优先来自 PyPI；CUDA 版 Torch/Torchvision 来自 PyTorch。首次使用相反优先级时，NCCL 被指向当前节点连接超时的 `pypi.nvidia.com`，失败日志予以保留。不要对已有 `.venv` 重复执行创建命令。环境安装不下载 Wan2.2 / ActionDiT 权重，不生成真实文本缓存，也不启动训练；这些步骤见下文。

2026-10-11 的独立环境验收已通过：`pip check` 无依赖冲突；FastWAM/Wan、训练器和 FR3 数据链导入成功；CPU SDPA 前后向、PyAV H.264 编码器与配置解析通过；FR3 回归 **16 项通过**（189.77 秒，1 条 torchvision 弃用警告）。完整命令、版本、安装重试和检查日志见[环境验收记录](../../../../test_artifacts/easywam_training_env_20261011_v1/RESULTS.json)，全部包版本另存于[版本清单](../../../../test_artifacts/easywam_training_env_20261011_v1/packages.json)。`python`、`accelerate`、`hf` 和 `ninja` 都位于 `.venv/bin`。

环境验收时当前节点未挂载 NVIDIA 设备，旧入口的默认 Wan2.2 与 ActionDiT 路径也不存在。后续新流程已接入父目录中现有 Robot DiT、UMT5、VAE 和 tokenizer，并成功在 CPU 上生成真实文本缓存，详见下文。CUDA/NCCL、显存、训练和真机执行仍为 **NOT-RUN**。CPU 检查临时使用的 `DS_ACCELERATOR=cpu` 和空 `CUDA_VISIBLE_DEVICES` 没有写入激活脚本，不会作为训练默认值。

如果从源码运行且尚未以 editable 方式安装 EasyWAM，先将 `src` 加入导入路径：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 选择输入数据

已有 OpenPI 转换结果满足以下字段时，可以直接使用其 LeRobot v2.1 数据集：

| 字段 | 磁盘表示 |
| --- | --- |
| `observation.images.left` | RGB 视频，640 × 480，10 FPS |
| `observation.images.wrist` | RGB 视频，640 × 480，10 FPS |
| `observation.state` | `float32[8]`：7 维绝对关节位置，随后是夹爪状态 |
| `action` | `float32[8]`：7 维绝对关节目标，随后是原始夹爪命令 |
| 任务 | 原始指令，通过 `task_index` 与 `meta/tasks.jsonl` 关联 |

参考 Parquet 文件携带的 Hugging Face `List` 特征元数据，不能被 EasyWAM 的 `datasets==3.6.0` 自动反序列化。v2.1 加载器现已在整库读取和指定 episode 读取时，使用 LeRobot `meta/info.json` 中的 features 构造读取 schema；源 Parquet 与 MP4 均不修改，v3 加载行为不变。两项元数据回归与参考数据 episode 0 直接加载均已通过。

本次提供的参考数据已经转换好：

```bash
OPENPI_REFERENCE=/path/to/storage/datasets/dataset_real_world/fr3_example_conversion_reference_20261011/converted_dataset/lsk/fr3_example
```

其元数据记录 100 条轨迹、52,529 帧、200 个视频、1 个任务，帧率为 10 Hz。它与相邻原始目录 `fr3_example` 是同一批演示，选择一种表示即可，同时加入会重复数据。不要对这份 10 Hz 数据再次降采样，也不要修改参考包。

新一键入口 `run_real_world_fr3.py` 与直接 Hydra 配方的默认数据目录均为 `./data/real_world_fr3_10hz`，用于接收原始数据转换结果；旧 `train_real_world_fr3.sh` 的默认值才是上述现有参考数据。可以用 `FR3_DATA_DIR` 或 `data.train.dataset_dirs` 指定其他位置。使用现有 OpenPI 结果时，应选择 `text-cache`、`train` 阶段；不要将它作为新流程 `all`/`prepare` 的转换输出。数据结构如下：

```text
<dataset>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/
│   ├── observation.images.left/episode_000000.mp4
│   └── observation.images.wrist/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

## 需要时从原始录制转换

`scripts/convert_real_world_fr3.py` 读取以下结构中的原始轨迹：

```text
<raw-root>/
├── data/episode_gello_000000.parquet
└── videos/
    ├── cam_000000000001_episode_gello_000000.mp4
    └── cam_000000000002_episode_gello_000000.mp4
```

相机映射固定为 `000000000001 → left`、`000000000002 → wrist`。Parquet 必须包含 `timestamp`、`qpos`、`action`、`language_instruction`；每行状态与动作均为 8 个有限数值。两路视频的帧数必须对应轨迹行数，源帧率必须一致。

```bash
python scripts/convert_real_world_fr3.py \
  --input-roots /path/to/raw/fr3_example \
  --output-root ./data/real_world_fr3_10hz \
  --fps 10
```

`--input-roots` 支持多个原始目录，输出目录必须尚不存在。默认转换全部轨迹；`--max-episodes 1` 可单独生成一个小转换结果供检查。脚本写入本地 LeRobot v2.1 数据，同时生成 `conversion_summary.json`，记录来源、相机映射、帧率和逐轨迹的原始/输出帧数；不会上传或覆盖源数据。

对于本次 30 FPS 原始数据，转换同步选择双相机、`qpos`、`action` 和指令的索引 `0, 3, 6, ...`。轨迹输出帧数为 `ceil(原始帧数 / 3)`，输出第 `k` 帧的时间戳是 `k / 10`。对齐依据是帧序号，没有对原始 timestamp 插值，因此不会校正采集时已经存在的延迟。源帧率与输出帧率必须对应整数 stride。

图像解码后由 BGR 转 RGB，通过原生 writer 写成 H.264 视频，保留原朝向与 640 × 480 完整视野；视频重新编码有损。状态与动作以绝对 `float32[8]` 落盘，转换阶段不减关节、不归一化、不二值化或反转夹爪，也不应用 LIBERO 的图像旋转。采集代码的前 7 维关节单位是弧度，末维约定为 `0=打开、1=闭合`，训练转换保留这些原值。

## 训练加载器实际构造什么

数据配方位于 `configs/data/real_world_fr3.yaml`，由 `configs/task/real_world_fr3_fastwam_idm_wan22.yaml` 选择。

| 配置 | 数值与作用 |
| --- | --- |
| `num_frames` | `17` 个观测时间步，包含当前观测 |
| `global_sample_stride` | `1`，保留磁盘上的 10 Hz 时间轴 |
| `action_video_freq_ratio` | `4`，解码视频偏移 `0, 4, 8, 12, 16` |
| 动作长度 | `16`，动作偏移 `0..15`，每 0.1 秒一条命令 |
| action / state 维度 | `8` / `8`，不补成 OpenPI 的 32 维 |
| 相机顺序 | `left` 在左，`wrist` 在右，横向拼接 |
| 最终视频尺寸 | 按高 × 宽为 `224 × 448` |

每路相机从 480 × 640 等比例缩放到 168 × 224，上下各补 28 像素成为 224 × 224，再横向拼成一张 224 × 448 RGB 图像。完整视野得以保留，不裁剪、不旋转，也不虚构第三路相机。模型视频 tensor 为 `[B, 3, 5, 224, 448]`，动作目标为 `[B, 16, 8]`。5 帧满足 Wan 的 `T % 4 == 1`，16 个动作可被 4 个视频转移整除。

归一化前执行 `RelativeJointTransform(keys=[default], joint_dim=7)`。从第 `t` 帧开始的 chunk 使用：

```python
relative_actions[:, :7] = absolute_actions[t : t + 16, :7] - state[t, :7]
relative_actions[:, 7] = absolute_actions[t : t + 16, 7]
```

每一步目标都减去同一个当前关节状态，不是相邻动作间的差，也不用未来实测 state 替换 action。state 保持绝对值。轨迹尾部由加载器重复末尾目标并提供 padding mask；`delta_action_dim_mask: null` 保证相对转换前保留这些绝对目标。padding 动作不计入动作损失。

以后接控制器时，应先反归一化，再给预测结果前 7 维加回原始当前关节状态，第 8 维保持原值。这说明表示的逆变换契约，当前没有实现真机执行客户端，现有 LIBERO/RoboTwin 部署接口也不是 FR3 执行入口。

参考包中当前使用的 `deploy_fr3_chunk-final.py` 把观测 state 的末维写成 `width > 0.05` 时为 `1`，即张开为 1，与采集时的闭合为 1 相反。后续接入时不能照抄这段 state 构造；必须先对齐采集约定。该客户端送出的每路图像也已做成带留边的 `224 × 224`，不能再套用面向原始 `640 × 480` 图像的 resize/pad，否则会重复添加留边。这些是现有客户端与训练数据之间的实际差异，本次没有修改参考包或实现真机部署。

上述依据位于 `dataset_real_world/fr3_example_deployment`：采集定义在 `gello_software/gello/agents/collect_fr3_gello_lerobot.py` 第 11–15、253–269 行；客户端 state 构造在 `irl_polymetis/polymetis/tests/deploy_fr3_chunk-final.py` 第 134–138 行，图像预处理在第 98–101 行。该客户端第 317–323 行的动作输出仍以 `action[7] > 0.5` 执行闭合，方向冲突具体发生在观测 state。

## 统计、模型资源与训练入口

配方采用相对关节转换之后的全局 z-score 归一化，EasyWAM 会将归一化数值裁剪到 `[-5, 5]`。`data.train.pretrained_norm_stats: null` 让训练初始化时的数据集计算 EasyWAM 统计，并将 `dataset_stats.json` 保存到训练输出目录。统计沿用 EasyWAM 按 episode 聚合、包含末尾重复样本的语义。正式训练应使用选定的完整训练数据计算统计，不能复用单轨迹 smoke 的统计。OpenPI 统计、LeRobot `episodes_stats.jsonl` 和 LIBERO 统计都不能替代这个文件；参考包原 OpenPI 检查使用的 `skip_norm_stats=True` 也没有验证本配方的统计。

[`scripts/run_real_world_fr3.sh`](../../../scripts/run_real_world_fr3.sh) 是新的一键入口，自动切换到 EasyWAM 根目录，使用 `.venv/bin/python` 并设置 `PATH`/`PYTHONPATH`。它调用 [`run_real_world_fr3.py`](../../../scripts/run_real_world_fr3.py) 组合一次最终 Hydra 配置，并将相同的数据、模型、缓存与用户覆盖传给各阶段；子进程失败会中止后续步骤。**不带阶段参数时默认执行 `all`，最后会启动训练。** 只准备数据和资源时选择 `prepare`。环境已激活时，`python scripts/run_real_world_fr3.py <stage>` 是等价方式。

| 阶段 | 执行内容 |
| --- | --- |
| `all`（默认） | `convert → init-action → text-cache → train` |
| `prepare` | `convert → init-action → text-cache`，完成后退出 |
| `config` | 只解析并打印最终配置，不构建模型、不访问 GPU |
| `convert` | 将所选原始目录中的全部轨迹转换为 10 Hz LeRobot v2.1 |
| `init-action` | 从所选 Robot 视频 DiT 生成独立 ActionDiT backbone 初始化文件 |
| `text-cache` | 用所选本地 UMT5/tokenizer 补齐真实 masked 文本缓存 |
| `train` | 使用已有数据、ActionDiT 和缓存，启动原生 Accelerate/DeepSpeed ZeRO-1 trainer |

旧 [`scripts/train_real_world_fr3.sh`](../../../scripts/train_real_world_fr3.sh) 保留 `config`、`text-cache`、`train` 三阶段，无参数时显示帮助。它默认使用现有 OpenPI 参考数据、`EasyWAM/checkpoints/Wan2.2-TI2V-5B` 和预先存在的 `ActionDiT_Wan22_5B_alphascale_1024hdim.pt`，不会转换数据或初始化 ActionDiT。以下命令和默认值均指新工作流入口。

## Robot 权重与独立 ActionDiT 初始化

本机 `FR3_CHECKPOINT_ROOT` 默认是 `/path/to/storage/checkpoints`，即 `EasyWAM/../../checkpoints`。Robot 目录中的视频 DiT 是 `checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors`；该目录不包含 T5/VAE，二者与 tokenizer 默认复用同一 checkpoint 根目录下的 `checkpoints/Wan2.2-TI2V-5B/`。

| 环境变量 | 新入口的默认值或作用 |
| --- | --- |
| `EASYWAM_PYTHON` | shell 入口使用的 Python，默认 `EasyWAM/.venv/bin/python` |
| `FR3_RAW_ROOT` | `EasyWAM/../../datasets/dataset_real_world/fr3_example` |
| `FR3_DATA_DIR` | `EasyWAM/data/real_world_fr3_10hz`，转换输出/训练输入 |
| `FR3_TEXT_CACHE` | `EasyWAM/data/text_embeds_cache/real_world_fr3/wan22` |
| `FR3_CHECKPOINT_ROOT` | 上述父目录中的 `checkpoints` 根目录 |
| `WAN22_MODEL_DIR` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-5B-Robot` |
| `WAN22_DIT_PATH` | `$WAN22_MODEL_DIR/checkpoint.safetensors` |
| `WAN22_T5_PATH` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth` |
| `WAN22_VAE_PATH` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |
| `WAN22_TOKENIZER_DIR` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl` |
| `FR3_ACTION_DIT_PATH` | `EasyWAM/checkpoints/ActionDiT_FR3_Wan22_Robot_alphascale_1024hdim.pt` |
| `FR3_PREPROCESS_DEVICE` | ActionDiT 初始化设备，默认 `cpu`，该阶段使用 bfloat16 |
| `FR3_TEXT_DEVICE` / `FR3_TEXT_DTYPE` | 文本计算默认 `cpu` / `bfloat16`；本次实测使用 `cpu` / `float32` |
| `RUN_ID` / `FR3_OUTPUT_DIR` | 默认时间戳；输出到 `EasyWAM/runs/real_world_fr3_fastwam_idm_wan22/$RUN_ID` |
| `NPROC_PER_NODE` | 每节点训练进程数；未设置时取 `PET_NPROC_PER_NODE`，否则为 `1` |

三个直接文件路径对应 `model.backbone.dit_path`、`text_encoder_path`、`vae_path`；tokenizer 对应 `model.backbone.tokenizer_model_id`。因此无需复制或改名用户提供的 Robot checkpoint。直接使用旧配置且三个文件字段为 `null` 时，原有按 `model_id` 目录查找标准文件名的行为不变。入口不下载模型资源。

`init-action` 调用原生 `preprocess_action_dit_backbone.py --task real_world_fr3_fastwam_idm_wan22`，传递同一套 Hydra 覆盖，从所选视频 DiT 插值到配置中的 1024 维 ActionDiT，并默认启用 alpha scaling。输出是独立的共享 backbone 初始化文件，不覆盖 Robot 权重，也不是训练完成的 FR3 checkpoint。输出文件已存在时直接复用；**更换源 DiT 或 ActionDiT 结构时，必须设置新的 `FR3_ACTION_DIT_PATH`**，脚本不会自动判断旧文件是否来自当前源权重。

## 一键运行与已有数据复用

直接检查配置，再选择完整流程或仅准备；shell 入口自动使用项目环境：

```bash
bash scripts/run_real_world_fr3.sh config

# 默认 all：转换、初始化、文本缓存，随后启动训练。
FR3_DATA_DIR="$PWD/data/real_world_fr3_10hz" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh

# 只准备，完成后不启动训练。
FR3_DATA_DIR="$PWD/data/real_world_fr3_10hz" \
  bash scripts/run_real_world_fr3.sh prepare
```

`convert` 阶段要求恰好一个输出数据集。输出不存在时转换原始目录中的全部 Parquet；已存在时，只有 `meta/info.json` 和完整 `conversion_summary.json` 存在，且摘要中的原始根目录、全部 episode 来源路径及 10 Hz 输出帧率与本次选择一致，才跳过转换。单轨迹 smoke 输出不能冒充全量转换。其他已有输出应另选新的 `FR3_DATA_DIR`；原始数据和已有参考包不被覆盖。

若使用上文的 `OPENPI_REFERENCE`，直接跳过转换，先补齐初始化和文本缓存，再训练：

```bash
FR3_DATA_DIR="$OPENPI_REFERENCE" bash scripts/run_real_world_fr3.sh init-action
FR3_DATA_DIR="$OPENPI_REFERENCE" bash scripts/run_real_world_fr3.sh text-cache
FR3_DATA_DIR="$OPENPI_REFERENCE" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh train
```

所有阶段接受尾部 Hydra 覆盖，后面的同名参数优先。分开执行时，数据、模型、缓存及结构覆盖应保持一致。若要固定训练输出名称，设置 `RUN_ID` 或 `FR3_OUTPUT_DIR`。多节点从原始数据开始时，应在一个进程上执行一次 `prepare`；使用已有 OpenPI 数据时则各执行一次 `init-action` 与 `text-cache`。共享数据、ActionDiT 和文本缓存准备好后，各节点仅执行 `train`，不要重复运行 `all`。多节点参数继续使用原生 launcher 的 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 及对应 `PET_*` 变量。入口保留 `CUDA_VISIBLE_DEVICES`，训练时由操作者选择已分配设备，并匹配训练进程数。

文本缓存默认 `++overwrite=false`，先按任务去重、分配 rank 并筛选已有文件；某个 rank 所需缓存全部存在时，跳过 T5/tokenizer 解析与加载，同时保留必要的分布式统计汇总。工作流入口仍检查配置中的 T5 文件与 tokenizer 目录存在。强制重建可追加 `++overwrite=true`。原生缓存脚本未指定设备时仍自动选择 CPU/GPU，并默认 bfloat16；新工作流的默认设备则明确为 CPU。CPU float32 的调用方式是：

```bash
FR3_TEXT_DEVICE=cpu FR3_TEXT_DTYPE=float32 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  bash scripts/run_real_world_fr3.sh text-cache
```

也可使用 `++text_encoder_device=cpu ++text_encoder_dtype=float32`。计算 dtype 不改变缓存格式：输出仍为 bfloat16 context、bool mask，padding 位置清零。训练默认 `load_text_encoder=false`，必须事先准备真实缓存；OpenPI token、历史 padding 可见的 FastWAM 缓存和测试用合成缓存不能替代它。

FR3 task 现在直接选择 `real_world_fr3` 数据和 `fastwam_idm_wan22` 模型，并显式配置训练参数，不再继承 LIBERO task。默认每进程 batch 16、梯度累积 1、学习率 `1e-4`、cosine schedule、20,000 个优化器更新，日志间隔 100、保存间隔 2,000、评估间隔 1,000。这些是初始设置，未做 FR3 调参或显存验证。两个 DiT 与线性状态投影训练，VAE 与文本编码器冻结；运行时在模型构建之前设置 `seed`，使新建动作/状态投影也受该种子控制。

按显存调整每进程 batch 与累积，例如：

```bash
TRAIN_ARGS=(batch_size=2 gradient_accumulation_steps=8 num_workers=4)
bash scripts/run_real_world_fr3.sh config "${TRAIN_ARGS[@]}"
NPROC_PER_NODE=1 bash scripts/run_real_world_fr3.sh train "${TRAIN_ARGS[@]}"
```

有效全局 batch 为 `batch_size × 总训练进程数 × gradient_accumulation_steps`；上例单进程时为 16。需要减少激活内存时，可以另行覆盖 `vae_micro_batch_size=1` 和 `model.backbone.use_gradient_checkpointing=true`。这些参数不是对特定 GPU 容量或吞吐的保证。

当前 `val_set_proportion: 0.0` 且没有独立 `val` 节点，trainer 的默认评估从训练数据中为每个 rank 随机取一个样本，执行生成、动作误差和 VAE 重建诊断。它不是 held-out 泛化、LIBERO 仿真或 FR3 真机成功率。默认 `eval_every=1000` 保留这一离线诊断；追加 `eval_every=0` 可关闭。`eval_save_video=false` 只关闭视频落盘，仍会运行推理。

输出包含 `config.yaml`、首次计算的 `dataset_stats.json`、`train.log`、`checkpoints/weights/step_XXXXXX.pt` 和 `checkpoints/state/step_XXXXXX/`。多节点日志名为 `train_node_<rank>.log`。最后一个 `output_dir=...` 覆盖同时决定训练输出与 launcher 日志目录；默认只保留最近 3 组权重/训练状态，可用 `checkpoint_save_limit` 调整。

## 继续训练与权重初始化

完整续训使用 `checkpoints/state/step_XXXXXX/` 目录，同时指定原运行的统计。以下示例假设原运行使用上面的 batch 2、累积 8：

```bash
PREVIOUS_RUN=/absolute/path/to/previous-run
export FR3_DATA_DIR=/absolute/path/to/original-training-dataset
FR3_OUTPUT_DIR="$PWD/runs/fr3_idm_resume01" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh train \
  "resume=$PREVIOUS_RUN/checkpoints/state/step_002000" \
  "data.train.pretrained_norm_stats=$PREVIOUS_RUN/dataset_stats.json" \
  batch_size=2 gradient_accumulation_steps=8 num_workers=4
```

目录恢复模型、优化器、调度器、训练步数和数据采样进度。保持原任务、数据、统计、seed、batch、进程数、梯度累积及训练调度一致；`max_steps` 是累计目标步数，不是追加步数。采样位置依赖 batch 与总进程数，改变它们不能宣称保持相同的数据续训轨迹。

若只想从一个兼容的 FR3 模型开始新的训练，使用权重文件，并仍携带它匹配的统计：

```bash
FR3_OUTPUT_DIR="$PWD/runs/fr3_idm_from_weights01" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh train \
  "resume=$PREVIOUS_RUN/checkpoints/weights/step_002000.pt" \
  "data.train.pretrained_norm_stats=$PREVIOUS_RUN/dataset_stats.json" \
  batch_size=2 gradient_accumulation_steps=8 num_workers=4
```

文件分支只恢复权重，不恢复优化器、调度器或步数。仅在随后确实恢复兼容完整模型时，才可用 `model.skip_dit_load_from_pretrain=true` 省去重复加载 DiT 预训练权重；VAE 仍需提供。

动作与状态均为 8 维的 FR3 模型，不能严格加载动作 7 维、状态 8 维的 LIBERO 完整 checkpoint，因为动作投影层形状不匹配。共享 Wan/ActionDiT backbone 不包含这些任务动作投影，可以用于新训练；OpenPI 模型权重也属于不同架构。更多权重语义见 [FastWAM 指南](../config/fastwam_zh.md)。

## 本次真实文本缓存与验证范围

2026-10-11 已在独立 `.venv` 中实际加载本地 UMT5 与 tokenizer，使用 **CPU、float32 计算、8 个线程**生成真实文本缓存。日志记录 `required=2 cached=0 to_encode=2`，完成后为 **`new=2 overwrite=0 skip=0`**，没有超长提示词。输出位于 `EasyWAM/data/text_embeds_cache/real_world_fr3/wan22`，使用训练所需的 bfloat16 context、bool mask 与零 padding。完整资产路径、实际子命令和结果见[真实文本缓存日志](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/precompute_real_text.log)。

这两条缓存分别覆盖真实指令 `Insert the tubes into the rack one by one.` 与转换器空层级占位任务经默认模板生成的提示词。任务文本来自完整 100 条 OpenPI 参考数据的 metadata，以及先前单轨迹真实转换结果 `/tmp/easywam-fr3-real-smoke-20261011` 的 metadata。两份输入只用于收集并去重任务文本；这不表示训练时同时加入重复的 episode 0，也没有执行 100 条轨迹的全量重新转换。

[真实缓存消费验证](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/real_text_cache.json)通过：两份 context 均为 bfloat16 `[128,4096]`，mask 为 bool `[128]`；真实指令有 28 个有效 token，空占位模板有 18 个，数值均有限且 padding 为零。配置后的 `RobotVideoDataset._get(0)` 实际读取 episode 0，得到 `[3,5,224,448]` 视频和 `[16,8]` 动作/proprio；返回的 context 与 mask 和磁盘真实缓存逐元素一致。

[真实缓存复用重跑](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/precompute_real_text_reuse.log)也已退出 0：`required=2 cached=2 to_encode=0`，最终 `new=0 overwrite=0 skip=2`，没有进入 `Preparing text encoder` 分支。因此跳过模型加载既有专项测试覆盖，也有真实缓存重跑证据。

本轮 CPU 测试分组累计覆盖通过 **45 个用例**：[既有回归 23 项](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_existing_regression.log)、资产/预处理 9 项、文本缓存 8 项和[一键流程 5 项](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_workflow.log)。

文本缓存专项 8 个用例均已覆盖通过：首轮 6 项通过，另外 2 项因分布式测试替身漏模拟 `get_rank()` 而失败，补齐替身后只重跑这 2 项并通过。[首轮日志](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_precompute_initial.log)与[复测日志](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_precompute_retry.log)均保留。其中完整缓存命中用例禁止调用模型路径解析、T5 加载和 tokenizer 构造，验证了提前跳过模型的行为；空 rank 的统计汇总也有覆盖。直接资产路径、任务配置与 ActionDiT 插值/alpha scaling 的[9 项 CPU 测试](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/asset_preprocess_cpu.json)使用小型替身验证，没有执行完整真实 ActionDiT 初始化。

本次已完成的是代码接入和真实 CPU 文本缓存。全量重新转换、完整真实 ActionDiT 初始化、GPU 模型运行、GPU/多节点训练、吞吐测量及真机执行仍为 **NOT-RUN**；文档中的 `all` 是下一步会启动训练的实际命令，不代表这些阶段已经执行。

## 历史 CPU 验收

独立训练环境的最新验收见上文；以下保留此前临时 CPU 环境的实现验收。该轮完整 CPU 回归 **16 项通过**，耗时 244.40 秒，另有 1 条 torchvision 弃用警告。新增 6 项覆盖真实配置子进程解析、文本缓存/训练命令替身、输出日志与 GPU 环境保留，以及模型构建前设置种子。种子测试中的模型、Accelerator 和 trainer 都使用替身，不加载真实模型。结果见[训练入口实现验收记录](../../../../test_artifacts/real_world_fr3_training_20261011_v1/RESULTS.json)与 [pytest 日志](../../../../test_artifacts/real_world_fr3_training_20261011_v1/pytest.log)。该轮命令从 Outer 根目录执行：

```bash
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /tmp/easywam-fr3-cpu-20261011/bin/python -m pytest -q \
  EasyWAM/tests/test_real_world_fr3.py EasyWAM/tests/test_convert_real_world_fr3.py \
  EasyWAM/tests/test_fr3_openpi_parquet_metadata.py EasyWAM/tests/test_training_entrypoint.py \
  EasyWAM/tests/test_fr3_training_launcher.py
```

两个 shell 入口通过 `bash -n`。转换器、关节变换与 5 个测试文件共 7 个 Python 文件通过 Ruff check；`runtime.py` 保留 17 条既有 lint，按 code/message 与 `git HEAD` 对比没有新增，LeRobot reader 的 11 条既有问题沿用前轮记录。本次没有将这些有既有问题的文件声称为全文件 lint 通过。

以下保留前一阶段的数据接入证据，原报告和日志未覆盖。

前一阶段的数据转换、变换、配置、processor 与 v2.1 元数据专项测试 **10 项通过**，耗时 12.79 秒，另有 1 条 torchvision 弃用警告。从 Outer 仓库根目录执行测试：

```bash
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /tmp/easywam-fr3-cpu-20261011/bin/python -m pytest -q \
  EasyWAM/tests/test_real_world_fr3.py EasyWAM/tests/test_convert_real_world_fr3.py \
  EasyWAM/tests/test_fr3_openpi_parquet_metadata.py
```

转换器、关节变换与 3 个测试文件通过 Ruff check。原生 LeRobot reader 保留 11 条既有 lint，按诊断 code/message 计数与 `git HEAD` 比较没有新增问题。临时 CPU 环境使用项目要求的 PyAV `16.0.1`；此前继承的 PyAV `12.3` 缺少 `codec.canonical_name`，对齐依赖版本后重跑通过。

[真实 episode 0 报告](../../../../test_artifacts/real_world_fr3_20261011_v1/real_episode0.json)记录 **PASS**：1,469 个原始帧转换为 490 帧；输出全部 state/action 行精确等于 `float32(raw[::3])`；两路视频均为 490 帧、10 FPS、640 × 480。两路首/中/末共 6 次图像对照的 RGB 平均绝对误差为 2.024–2.280（数值尺度 0–255），对应有损视频重编码。现有 OpenPI episode 直接读取通过；新转换结果通过包含统计、processor 和合成文本 cache 的完整 `RobotVideoDataset` 加载，输出 `[3, 5, 224, 448]` 视频、`[16, 8]` 动作，末尾 padding 也通过。[验证脚本](../../../../test_artifacts/real_world_fr3_20261011_v1/validate_real_episode0.py)及修复前后日志与报告一并保留。

该轮真实数据验证覆盖 1 条轨迹，当时仅使用合成文本 cache。后续真实 CPU UMT5 缓存已完成，见上文；全量转换、GPU 运行、训练和真机执行没有因这些历史检查而获得验证。

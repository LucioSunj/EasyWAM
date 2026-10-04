# 在 EasyWAM 中使用 FastWAM

[English](fastwam.md) | [模型配置](models_zh.md) | [配置索引](README_zh.md)

Wan2.2 FastWAM、FastWAM-Joint、FastWAM-IDM 保留原版的线性动作输入层、动作输出层和
proprioception 投影，使用 EasyWAM 的 trainer、数据管线、结构化注意力、批量 VAE、
cross-attention K/V 复用、编译目标及动态批量推理。运行时无需导入原 FastWAM 包。

**文本 padding 采用 EasyWAM 的语义：**训练缓存和在线 prompt 编码均只让有效文本
token 参与注意力。这是对原版 FastWAM padding 可见行为的明确修改。因此，旧权重
可以加载，但原先的预测结果和评测分数不能直接沿用。

## 配方

| 模型 | LIBERO task | RoboTwin task |
| --- | --- | --- |
| FastWAM / UNCOND | `libero_fastwam_wan22` | `robotwin_fastwam_wan22` |
| FastWAM-Joint | `libero_fastwam_joint_wan22` | `robotwin_fastwam_joint_wan22` |
| FastWAM-IDM | `libero_fastwam_idm_wan22` | `robotwin_fastwam_idm_wan22` |

这些 task 继承对应 EasyWAM 配方的数据与训练设置，包括训练步数；它们不会自动恢复
旧实验的科学设置。Video DiT、Action DiT 和线性状态投影参与训练，VAE 与文本编码器
冻结。状态 token 固定放在文本 context 中（`state_position: context`）；IDM 的条件
视频加噪概率仍为 `0.5`。

先查看组合后的配置，此命令不会加载模型或启动训练：

```bash
python scripts/train.py --cfg job task=libero_fastwam_idm_wan22
```

按 [Wan2.2 准备指南](../backbone/wan22.md)与对应[数据指南](../data/README_zh.md)
准备资源。插值得到的 ActionDiT backbone 只包含共享 Transformer 权重，可以用于
FastWAM 的线性投影结构。预处理脚本支持解析新模型的继承配置：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_wan22.yaml \
  --backbone wan22 \
  --output checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

使用 EasyWAM 为所选配方生成文本缓存：

```bash
python scripts/precompute_text_embeds.py task=libero_fastwam_idm_wan22
```

旧 FastWAM 缓存的文件名和 padding 可见 mask 不符合新缓存语义，应重新生成，不能只
重命名旧文件。沿用已有 FastWAM 数据时，通过 `data.train.dataset_dirs` 指定数据目录；
从已训练权重初始化时，还需用 `data.train.pretrained_norm_stats` 指定对应归一化统计。

选定运行资源后，使用标准 EasyWAM launcher：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_fastwam_idm_wan22
```

注意力后端、VAE batch、梯度 checkpoint 和数据加载配置见[效率配置](efficiency_zh.md)。
迁移模型共用这些控制项；具体工作负载上的速度需要实测。

## 从原版 checkpoint 初始化

原版完整模型 checkpoint 包含 `mot`，启用状态输入时还包含 `proprio_encoder`。
新 loader 严格加载全部 MoT 权重，并将线性 proprioception 权重映射为 EasyWAM 的
`state_encoder`。缺少模型权重时会报错，不会静默保留随机层。

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_fastwam_idm_wan22 \
  resume=/path/to/original_fastwam_idm.pt \
  model.skip_dit_load_from_pretrain=true \
  +data.train.pretrained_norm_stats=/path/to/dataset_stats.json
```

原版 checkpoint 不记录变种，必须选择相应 UNCOND、Joint 或 IDM task。
`resume` 指向 `.pt` 时只初始化权重，训练状态重新建立。恢复本实现产生的完整训练
状态时，应指向 EasyWAM 原生 `checkpoints/state/step_*` 目录，以恢复 optimizer、
scheduler、RNG 和数据加载位置。新权重文件记录变种、`state_position: context`
及 `text_padding: masked`。

## 推理

标准评测入口可以使用新的 task 名称：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_fastwam_idm_wan22 \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

`infer_action_batch` 与 `infer_joint_batch` 支持逐样本 seed，并使用原生批量推理和
缓存路径；编译使用继承的架构专用目标。标准 LIBERO、LIBERO-Plus 和 RoboTwin
评测配置均适用。多环境吞吐与单环境延迟需要分别测量。

本次配方覆盖原版三种监督训练模型。本项目特有的 Action-only／双分支 LoRA BC、
causal 实验和 Gate／Flow-PPO 仍需单独接入损失与参数归属；EasyWAM 的原生
Video-LoRA 配方不等同于这些方法。

CPU 测试使用小尺寸真实 Transformer 层和合成的 VAE／文本组件：

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_fastwam.py
```

测试覆盖严格权重加载、padding 屏蔽、训练与梯度、批量推理、配方组合和原生训练状态
恢复；它们不证明生产 GPU 性能或 benchmark 成功率。

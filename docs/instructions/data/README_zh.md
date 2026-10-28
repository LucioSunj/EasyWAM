# 训练数据文档索引

[English](README.md) | [文档总索引](../../README_zh.md) | [返回项目 README](../../../README_zh.md)

本分区说明数据集下载、预期的 LeRobot 目录结构、文本特征预处理、归一化统计和对应的训练入口。

`data.train.processor.action_clip_range` 控制动作归一化后的裁剪范围，
默认为 `[-5.0, 5.0]`。所选 checkpoint 使用未裁剪的动作标签训练时，
将其设为 `null`。状态归一化默认仍保留 `[-5.0, 5.0]` 范围；
对于使用未裁剪状态训练的配置，可将独立的
`data.train.processor.state_clip_range` 设为 `null`。
这两个选项都不会禁用仿射归一化。

| 数据集 | 内容 | 文档 |
| --- | --- | --- |
| LIBERO | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](libero_zh.md) |
| RoboTwin | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](robotwin_zh.md) |
| RoboDojo | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](robodojo_zh.md) |
| RoboCasa365 | Hub 下载、数据目录、统计、文本缓存与训练 | [数据与训练](robocasa_zh.md) |
| FR3 真机 | 原始/OpenPI 数据、相对关节目标、统计与 FastWAM-IDM 训练 | [数据与训练](real_world_fr3_zh.md) |

仿真环境安装和 rollout 评测统一放在 [Benchmark 文档](../benchmark/README_zh.md)中。

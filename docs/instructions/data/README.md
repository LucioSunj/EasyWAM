# Training Data Guides

[中文](README_zh.md) | [Documentation index](../../README.md) | [Back to project README](../../../README.md)

This section covers dataset downloads, the expected LeRobot layout, text-embedding preprocessing, normalization statistics, and benchmark-specific training entrypoints.

`data.train.processor.action_clip_range` controls clipping after action
normalization. It defaults to `[-5.0, 5.0]`; use `null` when the selected
checkpoint was trained with unclipped action targets. State normalization
retains its `[-5.0, 5.0]` range by default; the independent
`data.train.processor.state_clip_range` option accepts `null` for recipes
trained with unclipped states. Neither option disables affine normalization.

| Dataset | Contents | Guide |
| --- | --- | --- |
| LIBERO | Hub download, dataset layout, statistics, text cache, and training | [Data and training](libero.md) |
| RoboTwin | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robotwin.md) |
| RoboDojo | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robodojo.md) |
| RoboCasa365 | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robocasa.md) |
| FR3 real-world | Raw/OpenPI data, relative joint targets, statistics, and FastWAM-IDM training | [Data and training](real_world_fr3.md) |

Simulator setup and rollout evaluation are intentionally kept in the [benchmark guides](../benchmark/README.md).

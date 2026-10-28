# Training Data Guides

[中文](README_zh.md) | [Documentation index](../../README.md) | [Back to project README](../../../README.md)

This section covers dataset downloads, the expected LeRobot layout, text-embedding preprocessing, normalization statistics, and benchmark-specific training entrypoints.

`data.train.processor.action_clip_range` controls clipping after action
normalization. It defaults to `[-5.0, 5.0]`; use `null` when the selected
checkpoint was trained with unclipped action targets. State normalization
retains its `[-5.0, 5.0]` range.

| Dataset | Contents | Guide |
| --- | --- | --- |
| LIBERO | Hub download, dataset layout, statistics, text cache, and training | [Data and training](libero.md) |
| RoboTwin | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robotwin.md) |
| RoboDojo | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robodojo.md) |
| RoboCasa365 | Hub download, dataset layout, statistics, text cache, and training | [Data and training](robocasa.md) |

Simulator setup and rollout evaluation are intentionally kept in the [benchmark guides](../benchmark/README.md).

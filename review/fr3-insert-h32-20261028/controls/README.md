# Fresh Insert After Authorized Drawer Early Stop

The user explicitly requested stopping Drawer immediately and keeping only its
complete step 10000 training state. Drawer stopped at step 11158, not at 20000.
Its old failed/blocked status remains unchanged. Only the exact bound
EARLY_STOP_HANDOFF.json, stop authorization, process-release proof, full keeper
CRC identity evidence, and cleanup audit can release this new queue.
A generic failure or an old comparison/completion record cannot release it.

This root reuses the sealed source/data/stats/cache from
fr3-insert-tubes-fullfix-h32-20261027 by identity. It never copies those assets.
The old after-drawer-recovery queue is retired and never restarted.

The one-shot sequence remains:
1. Original WanRobot/ActionDiT fresh preflight through step 3.
2. Independent full-state restore from step 3 through step 4.
3. Original WanRobot/ActionDiT fresh formal training to step 20000, resume=null.

The unchanged recipe is 4 GPUs, B4/GAS4/global64, BF16/ZeRO1, AC off, H32
(33 observations, 9 video frames, 32 actions), stride1 and no hard clipping.
NCCL_CUMEM_HOST_ENABLE=0 remains mandatory; P2P/IB are not disabled.
Quota gates remain fresh405/resume314/formal223 GiB and 108 GiB before every save.
The formal run retains one rolling state and three weights, saving before pruning.

Formal output is fr3_insert_tubes_after_drawer10000_h32_4gpu_b4a4_z1_20261028.
Per-rank start/update records add observation only; they do not change training.
monitor.py --loop is an independent 120-second read-only process. It never repairs,
restarts, or launches training. Completion still requires 20000/exit0, all four
rank runtime audits, the final state/weights, and full integrity verification.

The preserved Drawer state is self-contained for a future full-state restore;
the old recovery controller is intentionally not restartable because its original
2000 and acceptance2003 payloads were deleted at the user's request. Any future
Drawer resume requires a new isolated config with resume=step_010000 and the
original absolute target20000. It is never an initialization source for Insert.

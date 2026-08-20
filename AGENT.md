# VIS_APP 27B GRPO — cursor-cli 交接

给新开的 Cursor CLI / Agent 用。先读完再改代码或占 GPU。

## 你是谁、要干什么

在本机把 **Qwen/Qwen3.8-27B** 的 **verl GRPO（LoRA）** 跑通：camel 风格多轮 tool loop 写 web app，序列级结构奖励，**训练和推理同时占卡、GPU 利用率尽量高**，同时把 **workspace 结构分** 拉上去。

成功标准（按顺序）：

1. 进程能完整跑完至少 1 个 GRPO step，不 OOM、不立刻崩。
2. `tool_calls/mean > 0`（模型真的调 `write_file` 等工具，而不是只吐 markdown）。
3. 组内 8 条 reward **不全相同** → `actor/pg_loss` 非零。
4. `critic/score/mean`（结构分）明显高于 0；再拉长 rollout 上下文，让 compose + frontend + backend 更完整。
5. GPU 4/5/6 在 generate 和 update 阶段都有实质利用率，而不是一张卡 100%、另外两张空转。

## 硬约束（违反即失败）

- **只占用 GPU 4、5、6。** 0–3 和 7 上是别人的 vLLM，不要杀、不要 `CUDA_VISIBLE_DEVICES` 扫到它们。
- **不要 `git commit` / `git push`，除非用户明确说。**
- **不要 docker / sudo。** 用户不在 docker 组。训练期不 `docker compose up`。官方 Playwright `combined_score_critical` 是换机器后的 Phase 2。
- **不要做 token-level PRM，不要用 LLM-as-judge 当主奖励。** 主信号是 workspace 结构分；若磁盘上已有 `eval_result.json` 的 `combined_score_critical`，才覆盖结构分。
- **不要 256k 训练上下文。** 256k 只是独立 serve 的推理余量。camel 真实峰值 prompt 大约 4k–43k。
- **TP=3 非法。** Qwen3.8-27B：hidden=5120，intermediate=17408，kv heads=4，GDN key heads=16，都不能被 3 整除。只用 **TP=2 或 TP=4**（或 TP=1）。
- 不要重启独立 vLLM `127.0.0.1:8201`，除非用户明确要。
- 不要实现 fully-async（staleness>1）。现在目标是 **one-step-off**：generate t+1 同时 train t。

## 机器与路径

- 机器：8× H100 80GB，Linux。用户：`yuhang.yao`。
- 本仓：`/shared_home/yuhang.yao/VIS_APP_Code`
- MERA / 训练代码：`/shared_home/yuhang.yao/MERA-Evolve`
- verl 0.8：`/share5/users/yuhang.yao/verl`
- Python：**必须** `$MERA/.venv_qwen35/bin/python`（不要用 MERA 的普通 `venv` 训 27B）
- 模型：`Qwen/Qwen3.8-27B`（HF，Apache-2.0）。架构是 `Qwen3_5ForConditionalGeneration`（hybrid GDN + 每 4 层 full attn）。训练 flag 按 Qwen3.5 处理。
- 独立 serve 脚本（当前应处于关闭）：`_serve_qwen38_27b.sh`，曾用 GPU 4+5、port 8201、`max_len=262144`。

Qwen3.8/3.5 训练必备环境（`rl/train_grpo.sh` 已设）：

- `QWEN35_ENABLE_VERL_PATCHES=1`
- `FLA_TILELANG=0`
- `attn_implementation=sdpa`
- `use_remove_padding=False`（GDN 不能 flash-attn unpad）
- `language_model_only=True`，`gdn_prefill_backend=triton`，`enable_thinking=false`
- Actor 需要 Triton **3.3 overlay**：`$MERA/.deps/qwen35-triton33`
- vLLM 需要 Triton **3.6**（venv 自带）。把 overlay 塞进全局 `PYTHONPATH` 会让 vLLM 报 `No module named 'triton.language.target_info'`。

## 代码布局（本仓 `rl/`）

| 文件 | 作用 |
|---|---|
| `rl/agent_loop.py` | `VisappAgentLoop`（verl `ToolAgentLoop`）：临时 workspace，compose 齐了可提前停 |
| `rl/tools.py` | `write_file` / `read_file` / `list_dir` / `shell_exec`；禁 docker/sudo/下运行时 |
| `rl/workspace.py` | 结构分：compose 名 0.4 + yaml services 0.2 + frontend 0.2 + backend 0.2；html/templates 算 frontend；另有 `min(0.15, 0.015*n_files)` 密度项，避免全 0 |
| `rl/reward.py` | 优先扫 `workspace_dir`；没有目录才 markdown fence 兜底 |
| `rl/prepare_visapp_data.py` | c4 十个任务 → hermes `<tool_call>` parquet，`agent_name=visapp_agent` |
| `rl/config/tools.yaml` | 工具 schema |
| `rl/config/agent_loop.yaml` | `_target_: rl.agent_loop.VisappAgentLoop` |
| `rl/train_grpo.sh` | 通用入口；`SEPARATE_ROLLOUT=1` 走 `verl.experimental.one_step_off_policy.main_ppo` |
| `rl/train_grpo_27b_async.sh` | **当前主脚本**：1 GPU actor + 2 GPU vLLM TP=2 |
| `rl/train_grpo_replay_27b.py` + `rl/train_grpo_27b_replay.sh` | 离线 LoRA 一步，已成功 |
| `rl/test_workspace_reward.py` | 结构分单测，9/9 已过 |
| `rl/train_grpo_smoke.sh` | 1.5B 冒烟（pipeline 通，但小模型不调工具） |

数据：`_rl_runs/data_27b/{train,val}.parquet`（10 个 c4 任务）。

Camel 干净跑：`_runs_camel_qwen38_27b_256k/`。只有 **`5_travel-booking`** 完整跑完：`20260818_070058_5_travel-booking_c4_camel`。Trace：`logs/camel_single_trace.jsonl`。Camel 已知坑：`message_window_size=48` 会丢掉 user message，vLLM 报 `No user query found in messages`。Harness：`CAMEL_SINGLE=1 ./run_eval.sh --task … --variant c4 --cli camel`，无 macOS sandbox 时需要 `ALLOW_UNSANDBOXED_EVAL=1`。

## 已经证明什么

**27B LoRA 能在一张 80GB 上 backward，不 OOM。**

离线 replay（跳过 vLLM）：把 `5_travel-booking` 轨迹复制 8 份、reward 全 1，GPU 4，seq 1536+512。峰值 **59GB**。centered GRPO `pg_loss=0`（组内 reward 全相同，这是算法，不是挂了）。uncentered 更新通了：NLL≈1.27，`grad_norm≈0.60`。Adapter：`_rl_runs/grpo_27b_replay/lora_adapter`。

1.5B 冒烟：pipeline 通，但 `tool_calls/mean=0`，score/`pg_loss` 全 0。不要再拿 1.5B 当 accuracy 信号。

## 当前主方案（用户已同意）

**SEPARATE_ROLLOUT / one-step-off async**，只占 4、5、6：

- 2 GPU vLLM，**TP=2**
- 1 GPU actor LoRA r=16，bf16 + param/optimizer offload
- n=8，T=1.0，8 steps
- 现在：`max_prompt=4096`，`max_response=4096`，`max_model_len=8192`，12 turns，8 agent workers
- 目标（第一步通了之后再加）：rollout 16k–48k。不要一上来 256k。Actor 单卡 80GB 在 8k+ 很容易 OOM，需要 offload 或 2-GPU FSDP。

启动：

```bash
cd /shared_home/yuhang.yao/VIS_APP_Code
CUDA_VISIBLE_DEVICES=4,5,6 bash rl/train_grpo_27b_async.sh
```

日志：`_rl_runs/grpo_27b_async/train.log`  
利用率采样：`_rl_runs/grpo_27b_async/gpu_util.csv`

看这些指标：`tool_calls/mean`、`num_turns/mean`、`critic/score/mean`、`actor/pg_loss`、`[27b-async] exit=`。

## 现场失败记录（按时间）

都发生在 `train_grpo_27b_async.sh`。修复已经写进脚本，不要回退。

| # | 现象 | 修复（已落地） |
|---|---|---|
| 1 | Qwen GDN + `use_remove_padding=True` | `train_grpo.sh` 对 Qwen3.5/3.8 **强制 False** |
| 2 | Actor FSDP init OOM：默认 `model_dtype=fp32`，~76.7GB | `FSDP_MODEL_DTYPE=bf16`，`param_offload=True`，`optimizer_offload=True`，`use_torch_compile=False`。Actor 约 53GB 能起来 |
| 3 | Actor OK；vLLM TP=2 崩：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 和 CuMemAllocator / `--enable_sleep_mode` 冲突 | `train_grpo_27b_async.sh` 里 **unset** 该 env |
| 4 | 2026-08-18 用户要求停掉时：Ray/`OneStepTaskRunner` 已起，数据 filter 完（10/10），正在建 colocated worker。GPU 4/5/6 仍空（权重还没 load）。日志有 Triton overlay 警告。被 SIGTERM 杀掉，core dumped | **不是**卡死在 hydra argv。下次启动会再走几分钟加载 |

Attempt 4 日志里的非致命但下次会咬人的问题：

```
Failed to import Triton kernels ... No module named 'triton.language.target_info'
```

原因：`train_grpo.sh` 把 Triton 3.3 overlay 放进了 **全局 PYTHONPATH**。MERA 的 `Qwen35TrainingRayWorkerGroup` 会把 overlay **只给 actor**；`one_step_off_policy.main_ppo` **不会**走那条 worker group。需要二选一：

- 让 Ray `worker_process_setup_hook` 只在 actor 进程插 overlay（脚本已默认 `tau2_evolve.qwen35_worker_setup.install_qwen35_worker_patches`，确认它真的生效）；或
- 不要把 overlay 放到启动进程的 PYTHONPATH，只在 actor worker 里加。

Hydra dump 里仍可能出现某处 `model_dtype: fp32`（critic 已 disable，但核对 actor/ref 必须是 bf16）。

## 停机状态（写本文档时）

- 27B async 训练、Ray、GPU util sampler、优化 loop（`AGENT_LOOP_WAKE_visapp27b` watcher + 8m heartbeat）**已全部杀掉**。不要重新 arm `/loop`，除非用户再要求。
- GPU：**4/5/6 = 0 MiB**。0–3、7 仍被别人占满。port **8201 down**。
- 不要 `ray stop --force` 除非确认是我们的 Ray；停之前看 `nvidia-smi` 和进程用户。

## 建议下一步（按优先级）

1. **再起一次** `CUDA_VISIBLE_DEVICES=4,5,6 bash rl/train_grpo_27b_async.sh`，盯 `train.log` 直到 vLLM sleep/load 和 actor FSDP init 结束。加载 27B 可能要好几分钟，GPU 空不等于挂了。
2. 若 vLLM 仍因 Triton overlay 挂：从启动环境去掉 overlay，只给 actor；或关 vLLM sleep mode（`free_cache_engine` / enable_sleep_mode）。
3. 若 actor 仍然紧：fallback **2-GPU FSDP actor + 1-GPU vLLM TP=1**。永远不要 TP=3。
4. 第一个成功 step 必须看到：非零 `tool_calls`、组内 reward 方差、非零 `actor/pg_loss`。
5. 再把 rollout 提到 16k，然后 32k。Actor backward 跟不上就保持短训序列 / 更大 offload，不要把训练 `max_model_len` 和推理余量绑死。
6. 结构分已经给了密度项；若 27B 仍不调工具，查 hermes 格式、`VisappAgentLoop` 的 sampling bound、以及 prompt 是否被 4096 截断（`truncation=left`）。

## 不要做的“优化”

- 不要上 token-level reward / LLM judge。
- 不要为了利用率把别人的卡抢过来。
- 不要在没通 8k step 之前把 context 拉到 128k/256k。
- 不要改 git config，不要 force push。
- 不要在本机用 docker 跑官方 VISTA 视觉评测。

## 相关对话

完整过程：[VIS_APP 27B GRPO](935624b5-ee0e-480a-a6cd-a5346a0b9a11)（Cursor agent transcript）。

"""Qwen3.8 one-step-off entry: patch actor worker group without touching Ray startup.

``python -m verl.experimental.one_step_off_policy.main_ppo`` hardcodes
``RayWorkerGroup``.  This wrapper runs the same Hydra config but patches the
mapping inside the TaskRunner process, so Triton 3.3 overlay stays off vLLM.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import ray

from verl.experimental.one_step_off_policy import main_ppo as osp_main
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def _unwrap_ray_actor(cls):
    meta = getattr(cls, "__ray_metadata__", None) or getattr(cls, "_ray_metadata", None)
    if meta is not None and getattr(meta, "modified_class", None) is not None:
        return meta.modified_class
    for attr in ("__ray_actor_class__", "_ray_actor_class"):
        inner = getattr(cls, attr, None)
        if inner is not None:
            return inner
    raise TypeError(f"cannot unwrap Ray actor class {cls!r}")


_OrigTaskRunner = _unwrap_ray_actor(osp_main.OneStepTaskRunner)


class _VisappOneStepTaskRunner(_OrigTaskRunner):
    def run(self, config):
        from rl.qwen35_worker_setup import install_visapp_qwen35_worker_patches

        install_visapp_qwen35_worker_patches()
        return super().run(config)


VisappOneStepTaskRunner = ray.remote(num_cpus=10, max_concurrency=100)(_VisappOneStepTaskRunner)

_CONFIG_DIR = str(Path(osp_main.__file__).resolve().parent / "config")


@hydra.main(config_path=_CONFIG_DIR, config_name="one_step_off_ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=VisappOneStepTaskRunner)


if __name__ == "__main__":
    main()

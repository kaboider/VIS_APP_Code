"""Ray worker setup for Qwen3.8/3.5 one-step-off GRPO.

``one_step_off_policy.main_ppo`` hardcodes ``RayWorkerGroup`` and never goes
through MERA's ``Qwen35TrainingRayWorkerGroup``.  Without this hook the Triton
3.3 overlay has to sit on the driver PYTHONPATH, which breaks vLLM 0.18
(``triton.language.target_info``).  Patch the mapping so only actor/ref
workers get the overlay; rollout keeps the venv Triton 3.6.

Disaggregated ``update_weights`` calls ``get_per_tensor_param()`` which would
reload the 27B shard onto GPU.  Inject ``rl/actor_site`` sitecustomize so that
path skips the send (vLLM already loaded safetensors).  Keep the real
``load_fsdp_model_to_gpu`` for actor log_prob / update.

``worker_process_setup_hook`` is not enough: ray.init already sets a global
tau2 hook, and LoRA flag defaults still run ``load_fsdp_model_to_gpu``.
"""
from __future__ import annotations

import os
from pathlib import Path

from tau2_evolve.qwen35_ray_setup import Qwen35TrainingRayWorkerGroup

_ACTOR_SITE = str(Path(__file__).resolve().parent / "actor_site")


class VisappTrainingRayWorkerGroup(Qwen35TrainingRayWorkerGroup):
    """Actor workers: Triton overlay + actor_site LoRA weight-sync sitecustomize."""

    def __init__(self, *args, **kwargs):
        # Qwen35TrainingRayWorkerGroup overwrites PYTHONPATH with
        # overlay + os.environ["PYTHONPATH"]. Prepend actor_site there so
        # sitecustomize runs in WorkerDict (workers are created in super()).
        orig_pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (_ACTOR_SITE, orig_pp) if p)
        try:
            super().__init__(*args, **kwargs)
        finally:
            if orig_pp:
                os.environ["PYTHONPATH"] = orig_pp
            else:
                os.environ.pop("PYTHONPATH", None)
        actor_pp = self.customized_worker_env.get("PYTHONPATH", "")
        print(
            f"[visapp] actor worker_env PYTHONPATH has actor_site="
            f"{_ACTOR_SITE in actor_pp} overlay_first={actor_pp.split(os.pathsep)[0]!r}",
            flush=True,
        )

    def _create_worker(self, *args, **kwargs):
        ray_cls_with_init = kwargs["ray_cls_with_init"]
        update_options = ray_cls_with_init.update_options

        def update_options_with_setup_hook(options):
            runtime_env = options.get("runtime_env")
            if runtime_env is not None:
                options = dict(options)
                runtime_env = dict(runtime_env)
                runtime_env["worker_process_setup_hook"] = (
                    "rl.qwen35_worker_setup.install_visapp_qwen35_worker_patches"
                )
                options["runtime_env"] = runtime_env
            update_options(options)

        ray_cls_with_init.update_options = update_options_with_setup_hook
        try:
            return super(Qwen35TrainingRayWorkerGroup, self)._create_worker(*args, **kwargs)
        finally:
            ray_cls_with_init.update_options = update_options


def _patch_one_step_off_worker_group() -> None:
    try:
        import verl.experimental.one_step_off_policy.main_ppo as osp
        import verl.experimental.separation.utils as sep
    except ImportError:
        return
    current = osp.create_role_worker_mapping
    if getattr(current, "_visapp_qwen35_patched", False):
        return

    def create_role_worker_mapping(config):
        mapping, _ = current(config)
        print("[visapp] create_role_worker_mapping -> VisappTrainingRayWorkerGroup", flush=True)
        return mapping, VisappTrainingRayWorkerGroup

    create_role_worker_mapping._visapp_qwen35_patched = True  # type: ignore[attr-defined]
    osp.create_role_worker_mapping = create_role_worker_mapping
    sep.create_role_worker_mapping = create_role_worker_mapping


def _patch_ray_worker_actor_site() -> None:
    """Prepend actor_site onto every RayWorkerGroup worker PYTHONPATH.

    Mapping-class patches are easy to miss under Ray's actor wrapper. Injecting
    here still reaches WorkerDict even if the default RayWorkerGroup is used.
    """
    from verl.single_controller.ray.base import RayWorkerGroup

    orig = RayWorkerGroup._create_worker
    if getattr(orig, "_visapp_actor_site", False):
        return

    def _create_worker(self, *args, **kwargs):
        worker_env = dict(kwargs.get("worker_env") or {})
        pp = worker_env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        parts = [p for p in pp.split(os.pathsep) if p]
        if _ACTOR_SITE not in parts:
            worker_env["PYTHONPATH"] = os.pathsep.join([_ACTOR_SITE, *parts])
            kwargs["worker_env"] = worker_env
            print("[visapp] inject actor_site into worker PYTHONPATH", flush=True)
        return orig(self, *args, **kwargs)

    _create_worker._visapp_actor_site = True  # type: ignore[attr-defined]
    RayWorkerGroup._create_worker = _create_worker


def _patch_lora_weight_sync() -> None:
    """No-op here: actor_site sitecustomize owns CPU LoRA collect.

    WorkerDict often only gets the global tau2 setup hook, so the real patch
    must live in sitecustomize on the actor PYTHONPATH.
    """
    return


def install_visapp_qwen35_worker_patches() -> None:
    try:
        from tau2_evolve.qwen35_worker_setup import install_qwen35_worker_patches

        install_qwen35_worker_patches()
    except Exception:
        try:
            from sitecustomize import install_attention_patch

            install_attention_patch()
        except Exception:
            pass
    _patch_lora_weight_sync()
    _patch_one_step_off_worker_group()
    _patch_ray_worker_actor_site()

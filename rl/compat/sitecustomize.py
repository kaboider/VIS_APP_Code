"""Interpreter startup for Qwen3.8 one-step-off GRPO.

Do not import verl/torch here. Ray dashboard / metrics agents inherit this
PYTHONPATH; a heavy import delays their port handshake and raylet dies
waiting for metrics_agent_port.

Lazy-patch ``one_step_off_policy.main_ppo.create_role_worker_mapping`` only
after that module has actually been imported (OneStepTaskRunner / driver).
"""
from __future__ import annotations

import builtins
import os
import sys

_MERA = os.environ.get("MERA_ROOT", "/shared_home/yuhang.yao/MERA-Evolve")
_MERA_SITE = os.path.join(
    _MERA, "experiments/tau-2/compat/qwen35_torch_fallback/sitecustomize.py"
)
if os.path.isfile(_MERA_SITE):
    import runpy

    runpy.run_path(_MERA_SITE, run_name="sitecustomize")

_orig_import = builtins.__import__
_patching = False


def _try_patch() -> None:
    mod = sys.modules.get("verl.experimental.one_step_off_policy.main_ppo")
    if mod is None or not hasattr(mod, "create_role_worker_mapping"):
        return
    if getattr(mod.create_role_worker_mapping, "_visapp_qwen35_patched", False):
        return
    from rl.qwen35_worker_setup import _patch_one_step_off_worker_group

    _patch_one_step_off_worker_group()


def __import__(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A001
    global _patching
    module = _orig_import(name, globals, locals, fromlist, level)
    if _patching or os.environ.get("QWEN35_ENABLE_VERL_PATCHES") != "1":
        return module
    if name == "verl.experimental.one_step_off_policy.main_ppo" or (
        name == "verl.experimental.one_step_off_policy" and fromlist and "main_ppo" in fromlist
    ):
        _patching = True
        try:
            _try_patch()
        except Exception:
            pass
        finally:
            _patching = False
    return module


builtins.__import__ = __import__

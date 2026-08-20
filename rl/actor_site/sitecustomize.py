"""Actor-worker interpreter startup for 27B LoRA weight sync.

This directory is prepended onto actor PYTHONPATH only. Re-export
torch_fallback's ``install_attention_patch`` (tau2's worker hook imports it
from sitecustomize).

Skip FSDP reload / LoRA send during rollout weight sync (vLLM already has
base safetensors). Do NOT no-op ``load_fsdp_model_to_gpu`` globally: log_prob
and the actor update need that reload after param offload.
"""
from __future__ import annotations

import builtins
import os
import runpy
import sys

_FALLBACK = os.path.join(
    os.environ.get("MERA_ROOT", "/shared_home/yuhang.yao/MERA-Evolve"),
    "experiments/tau-2/compat/qwen35_torch_fallback/sitecustomize.py",
)
_fallback_ns = {}
if os.path.isfile(_FALLBACK):
    # Do not use run_name="sitecustomize"; that would replace this module.
    _fallback_ns = runpy.run_path(_FALLBACK, run_name="_qwen35_torch_fallback_site")

install_attention_patch = _fallback_ns.get("install_attention_patch")
if install_attention_patch is None:

    def install_attention_patch():
        return None

_patched = False
_orig_import = builtins.__import__


def _cpu_tensor(param):
    import torch
    from torch.distributed.tensor import DTensor

    if isinstance(param, DTensor):
        param = param.full_tensor()
    if hasattr(param, "detach"):
        param = param.detach()
    if hasattr(param, "to"):
        param = param.to("cpu")
    return param


def _remap_lora_key(name: str) -> str:
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace(".default._flat_param", ".weight")
    name = name.replace(".lora_A.default.", ".lora_A.")
    name = name.replace(".lora_B.default.", ".lora_B.")
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name


def _peft_json_path() -> str:
    return os.environ.get("VISAPP_LORA_PEFT_JSON", "/tmp/visapp27b_ray/lora_peft_config.json")


def _peft_config_json(peft_config) -> dict | None:
    import json

    if peft_config is None:
        return None
    raw = peft_config.to_dict() if hasattr(peft_config, "to_dict") else dict(peft_config)
    out = {}
    for key, value in raw.items():
        if hasattr(value, "value"):
            value = value.value
        elif isinstance(value, set):
            value = list(value)
        try:
            json.dumps(value)
        except TypeError:
            value = str(value)
        out[key] = value
    return out


def _write_peft_json(peft_config_dict) -> None:
    import json
    from pathlib import Path

    if not peft_config_dict:
        return
    path = Path(_peft_json_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(peft_config_dict), encoding="utf-8")


def _read_peft_json() -> dict | None:
    import json
    from pathlib import Path

    path = Path(_peft_json_path())
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _peft_from_env() -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(os.environ.get("LORA_RANK", "16")),
        "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
    }


def _collect_lora_cpu(module):
    from peft.utils.save_and_load import get_peft_model_state_dict

    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    raw = {}
    try:
        from verl.utils.fsdp_utils import collect_lora_params

        raw = collect_lora_params(module, layered_summon=True, base_sync_done=True)
    except Exception as exc:
        print(f"[visapp] collect_lora_params failed: {exc}", flush=True)
        raw = {}
    if not raw:
        try:
            raw = get_peft_model_state_dict(peft_model)
        except Exception as exc:
            print(f"[visapp] get_peft_model_state_dict failed: {exc}", flush=True)
            raw = {
                name.replace("_fsdp_wrapped_module.", ""): param
                for name, param in peft_model.named_parameters()
                if "lora_" in name and "_flat_param" not in name
            }
    out = {}
    for name, param in raw.items():
        if "_flat_param" in name:
            continue
        out[_remap_lora_key(name)] = _cpu_tensor(param)
    return out


def _install_lora_sync_skip() -> None:
    global _patched
    if _patched or os.environ.get("VISAPP_LORA_WEIGHT_SYNC", "1") != "1":
        return
    impl = sys.modules.get("verl.workers.engine.fsdp.transformer_impl")
    if impl is None or not hasattr(impl, "FSDPEngine"):
        return
    orig_load = impl.load_fsdp_model_to_gpu

    def load_fsdp_model_to_gpu(model):
        import gc

        import torch

        try:
            param = next(model.parameters())
        except StopIteration:
            param = None
        if param is not None and getattr(param, "device", None) is not None and param.device.type == "cuda":
            print("[visapp] FSDP already on GPU, skip reload", flush=True)
            return None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print(f"[visapp] load_fsdp_model_to_gpu alloc={alloc:.1f}GiB", flush=True)
        return orig_load(model)

    impl.load_fsdp_model_to_gpu = load_fsdp_model_to_gpu
    orig_init = getattr(impl.FSDPEngine, "initialize", None)
    if orig_init is not None and not getattr(orig_init, "_visapp_empty_cache", False):

        def initialize(self, *args, **kwargs):
            import gc

            import torch

            out = orig_init(self, *args, **kwargs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"[visapp] after FSDP empty_cache alloc={alloc:.1f}GiB reserved={reserved:.1f}GiB", flush=True)
            return out

        initialize._visapp_empty_cache = True  # type: ignore[attr-defined]
        impl.FSDPEngine.initialize = initialize
    orig = impl.FSDPEngine.get_per_tensor_param
    if getattr(orig, "_visapp_skip_fsdp_reload", False):
        _patched = True
        return

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(peft_model, "peft_config"):
            return orig(self, layered_summon=False, base_sync_done=True, **kwargs)
        # NIXL send_weights loads tensors as full vLLM params. Init LoRA keys are
        # PEFT `base_model.*` and crash Qwen3.5 (`language_model.*`). vLLM already
        # has base safetensors; skip the payload so generate can start.
        # Do not call state_dict()/collect here: FSDP param-offload asserts on CPU.
        if os.environ.get("VISAPP_SKIP_ROLLOUT_LORA", "1") == "1":
            print("[visapp] skip rollout LoRA send (vLLM already has base)", flush=True)
            return iter(()), None
        peft_config = peft_model.peft_config.get("default", None)
        params = _collect_lora_cpu(self.module)
        peft_config_dict = _peft_config_json(peft_config)
        _write_peft_json(peft_config_dict)
        print(
            f"[visapp] CPU LoRA sync n_tensors={len(params)} sample={list(params)[:3]}",
            flush=True,
        )
        return params.items(), peft_config_dict

    get_per_tensor_param._visapp_skip_fsdp_reload = True  # type: ignore[attr-defined]
    impl.FSDPEngine.get_per_tensor_param = get_per_tensor_param
    _patched = True
    print("[visapp] actor sitecustomize: skip rollout LoRA send; skip FSDP reload if already on GPU", flush=True)


_skip_send_patched = False


def _rebuild_nested_position_ids(data, pid):
    """Rebuild Qwen mrope [4, S] nested position_ids after pickle loses jagged offsets."""
    import torch

    from verl.utils.tensordict_utils import nested_tensor_from_tensor_list

    mask = None
    if hasattr(data, "get"):
        mask = data.get("attention_mask")
    elif hasattr(data, "keys") and "attention_mask" in data.keys():
        mask = data["attention_mask"]
    try:
        rows = list(pid.unbind())
        if rows:
            return nested_tensor_from_tensor_list(rows, ragged_idx=rows[0].dim())
    except RuntimeError:
        pass
    if mask is None:
        return None
    if mask.is_nested:
        try:
            lengths = [int(row.shape[-1]) for row in mask.unbind()]
        except RuntimeError:
            padded = torch.nested.to_padded_tensor(mask, 0)
            lengths = [int(row.sum().item()) for row in padded]
    else:
        lengths = [int(x) for x in mask.sum(dim=-1).tolist()]
    values = pid.values()
    total = int(sum(lengths))
    if values.dim() == 2 and int(values.shape[-1]) == total:
        parts = []
        off = 0
        for length in lengths:
            parts.append(values[:, off : off + length].contiguous())
            off += length
        return nested_tensor_from_tensor_list(parts, ragged_idx=2)
    if values.dim() == 1 and int(values.shape[0]) == total:
        parts = []
        off = 0
        for length in lengths:
            parts.append(values[off : off + length].contiguous())
            off += length
        return nested_tensor_from_tensor_list(parts, ragged_idx=1)
    return None


def _install_position_ids_fix() -> None:
    tu = sys.modules.get("verl.utils.tensordict_utils")
    if tu is None or getattr(tu.maybe_fix_3d_position_ids, "_visapp_fix", False):
        return

    orig = tu.maybe_fix_3d_position_ids
    orig_index = tu.index_select_tensor_dict

    def maybe_fix_3d_position_ids(data):
        import torch

        pid = None
        if hasattr(data, "get"):
            pid = data.get("position_ids")
        elif hasattr(data, "keys") and "position_ids" in data.keys():
            pid = data["position_ids"]
        if isinstance(pid, torch.Tensor) and pid.is_nested:
            rebuilt = _rebuild_nested_position_ids(data, pid)
            if rebuilt is not None:
                data["position_ids"] = rebuilt
                print("[visapp] rebuilt nested position_ids", flush=True)
                return
        return orig(data)

    def index_select_tensor_dict(batch, indices):
        maybe_fix_3d_position_ids(batch)
        try:
            return orig_index(batch, indices)
        except RuntimeError as exc:
            if "split_with_sizes" not in str(exc):
                raise
            maybe_fix_3d_position_ids(batch)
            print("[visapp] retry index_select after position_ids rebuild", flush=True)
            return orig_index(batch, indices)

    maybe_fix_3d_position_ids._visapp_fix = True  # type: ignore[attr-defined]
    tu.maybe_fix_3d_position_ids = maybe_fix_3d_position_ids
    tu.index_select_tensor_dict = index_select_tensor_dict
    print("[visapp] actor sitecustomize: fix nested 3D position_ids unbind", flush=True)


def _copy_verl_register(src, dst) -> None:
    attr = "attrs_3141562937"
    if hasattr(src, attr):
        setattr(dst, attr, getattr(src, attr))
    dst.__name__ = getattr(src, "__name__", dst.__name__)
    dst.__qualname__ = getattr(src, "__qualname__", dst.__qualname__)


def _install_skip_weight_send() -> None:
    """NIXL async path drops peft_config, so vLLM would load LoRA tensors as full params.

    SKIP=1: no-op both sides (proven 32-step path).
    SKIP=0: send remapped LoRA; receive via vLLM add_lora (peft json sidecar).
    """
    skip = os.environ.get("VISAPP_SKIP_ROLLOUT_LORA", "1") == "1"

    ew = sys.modules.get("verl.workers.engine_workers")
    if skip and ew is not None and hasattr(ew, "ActorRolloutRefWorker"):
        orig = ew.ActorRolloutRefWorker.update_weights
        if not getattr(orig, "_visapp_skip_send", False):

            async def update_weights(self, global_steps=None, mode="auto"):
                print("[visapp] skip actor send_weights (vLLM already has base)", flush=True)
                return None

            _copy_verl_register(orig, update_weights)
            update_weights._visapp_skip_send = True  # type: ignore[attr-defined]
            ew.ActorRolloutRefWorker.update_weights = update_weights

    base = sys.modules.get("verl.checkpoint_engine.base")
    if base is None or not hasattr(base, "CheckpointEngineWorker"):
        return
    orig_ce = base.CheckpointEngineWorker.update_weights
    if getattr(orig_ce, "_visapp_weight_send", False):
        return

    if skip:

        async def ce_update_weights(self, global_steps=None):
            print("[visapp] skip checkpoint update_weights (no LoRA payload)", flush=True)
            return None

        _copy_verl_register(orig_ce, ce_update_weights)
        ce_update_weights._visapp_weight_send = True  # type: ignore[attr-defined]
        base.CheckpointEngineWorker.update_weights = ce_update_weights
        print("[visapp] skip empty NIXL/vLLM weight bucket", flush=True)
        return

    async def ce_update_weights(self, global_steps=None):
        peft = _read_peft_json() or _peft_from_env()
        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
        if not peft:
            print("[visapp] checkpoint update_weights skip (no peft config)", flush=True)
            if weights is not None and hasattr(weights, "__aiter__"):
                async for _ in weights:
                    pass
            elif weights is not None:
                for _ in weights:
                    pass
            return None
        print("[visapp] checkpoint update_weights LoRA add_lora", flush=True)
        await self.server_adapter.update_weights(
            weights,
            global_steps=global_steps,
            peft_config=peft,
            base_sync_done=True,
        )

    _copy_verl_register(orig_ce, ce_update_weights)
    ce_update_weights._visapp_weight_send = True  # type: ignore[attr-defined]
    base.CheckpointEngineWorker.update_weights = ce_update_weights
    print("[visapp] NIXL receive -> vLLM add_lora", flush=True)


def __import__(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A001
    module = _orig_import(name, globals, locals, fromlist, level)
    if not _patched:
        _install_lora_sync_skip()
    _install_skip_weight_send()
    _install_position_ids_fix()
    return module


builtins.__import__ = __import__

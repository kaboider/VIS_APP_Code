#!/usr/bin/env python3
"""Offline GRPO-shaped LoRA step on Qwen3.8-27B.

Assumes rollouts already exist: copy the 5_travel-booking camel trace 8 times,
assign reward=1, skip vLLM generate, and run one actor update. Accuracy is
irrelevant — this is a 27B fit / GDN-backward smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


def _read_jsonl_start(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "start":
                return event
    raise SystemExit(f"no start event in {path}")


def _build_ids(tokenizer, system: str, user: str, response: str, max_prompt: int, max_response: int):
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    if len(resp_ids) > max_response:
        resp_ids = resp_ids[:max_response]
    if not resp_ids:
        raise SystemExit("empty response ids after truncation")
    return prompt_ids, resp_ids


def _token_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean NLL over non-ignored tokens, plus per-token logprob mean."""
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    gather = logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(-100)
    n_tok = mask.sum().clamp_min(1)
    seq_logp = (gather * mask).sum() / n_tok
    nll = -seq_logp
    return nll, seq_logp


def _gpu_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {
        "alloc_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> int:
    vis = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=vis / "_runs_camel_qwen38_27b_256k" / "20260818_070058_5_travel-booking_c4_camel",
    )
    ap.add_argument("--out-dir", type=Path, default=vis / "_rl_runs" / "grpo_27b_replay")
    ap.add_argument("--n-copies", type=int, default=8)
    ap.add_argument("--reward", type=float, default=1.0)
    ap.add_argument("--max-prompt", type=int, default=1536)
    ap.add_argument("--max-response", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    run_dir = args.run_dir
    start = _read_jsonl_start(run_dir / "logs" / "camel_single_trace.jsonl")
    result = (run_dir / "logs" / "camel_single_result.txt").read_text(encoding="utf-8", errors="replace")
    compose = (run_dir / "workspace" / "docker-compose.yml").read_text(encoding="utf-8", errors="replace")
    response = (
        "Workspace is complete. Writing docker-compose.yml and stopping.\n\n"
        f"```yaml\n{compose.strip()}\n```\n\n{result.strip()}\n"
    )

    print(f"[replay] model={args.model}", flush=True)
    print(f"[replay] source={run_dir}", flush=True)
    print(f"[replay] n={args.n_copies} reward={args.reward} prompt<={args.max_prompt} resp<={args.max_response}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_ids, resp_ids = _build_ids(
        tokenizer, start["system_message"], start["prompt"], response, args.max_prompt, args.max_response
    )
    input_ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids
    print(
        f"[replay] tokens prompt={len(prompt_ids)} response={len(resp_ids)} total={len(input_ids)}",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": "5_travel-booking",
        "run_dir": str(run_dir),
        "n_copies": args.n_copies,
        "reward": args.reward,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(resp_ids),
        "model": args.model,
    }
    (args.out_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    t0 = time.time()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "text_config") and config.text_config is not None:
        config.text_config._attn_implementation = "sdpa"
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        for param in model.model.visual.parameters():
            param.requires_grad = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.train()
    model.print_trainable_parameters()
    print(f"[replay] loaded in {time.time() - t0:.1f}s mem={_gpu_mem()}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    opt.zero_grad(set_to_none=True)

    rewards = torch.full((args.n_copies,), float(args.reward), device="cuda")
    adv_centered = rewards - rewards.mean()
    print(
        f"[replay] rewards={rewards.tolist()} centered_adv={adv_centered.tolist()} "
        "(GRPO group std is 0 when all rewards match)",
        flush=True,
    )

    ids = torch.tensor([input_ids], device="cuda")
    lab = torch.tensor([labels], device="cuda")
    attn = torch.ones_like(ids)

    t1 = time.time()
    nlls = []
    logps = []
    for i in range(args.n_copies):
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        nll, seq_logp = _token_nll(out.logits, lab)
        # Centered GRPO advantage is 0 for identical rewards. Use uncentered
        # R=1 (REINFORCE / BC on the successful trace) so backward actually runs.
        loss = nll * rewards[i] / args.n_copies
        loss.backward()
        nlls.append(float(nll.detach()))
        logps.append(float(seq_logp.detach()))
        print(
            f"[replay] micro {i + 1}/{args.n_copies} nll={nlls[-1]:.4f} "
            f"logp={logps[-1]:.4f} mem={_gpu_mem()}",
            flush=True,
        )
        del out, nll, seq_logp, loss
        torch.cuda.empty_cache()

    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_s = time.time() - t1

    pg_centered = -float(adv_centered[0].item()) * (sum(logps) / len(logps))
    pg_uncentered = -float(args.reward) * (sum(logps) / len(logps))
    metrics = {
        "n_copies": args.n_copies,
        "reward": args.reward,
        "nll_mean": sum(nlls) / len(nlls),
        "logp_mean": sum(logps) / len(logps),
        "actor/pg_loss_centered": pg_centered,
        "actor/pg_loss_uncentered_r1": pg_uncentered,
        "grad_norm": float(grad_norm) if not isinstance(grad_norm, torch.Tensor) else float(grad_norm.item()),
        "step_s": step_s,
        "mem": _gpu_mem(),
        "ok": True,
    }
    print("[replay] metrics " + json.dumps(metrics, default=float), flush=True)

    adapter_dir = args.out_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(f"[replay] saved adapter {adapter_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
train/lora_qwen8b.py -- Unsloth LoRA fine-tune for a FIXED wall clock, logging tokens/s.

Defaults (all overridable by env or CLI):
  model      Qwen/Qwen3-8B        (Qwen/Qwen3.8-8B does not exist on the Hub as of 2026-09-02)
  dataset    yahma/alpaca-cleaned (public, no token)
  LoRA       r=16, alpha=32, dropout=0, all attention+MLP projections
  precision  bf16 weights, bf16 compute, NO 4-bit  (load_in_4bit=False)
  batch 8 x seq 2048, grad-accum 1
  TRAIN_MINUTES=15 wall clock -> a TrainerCallback stops training at the deadline.

Run under CUDA_VISIBLE_DEVICES=3 from train/lora_cotenant.sh (GPUs 0-2 keep serving).

API notes (verified 2026-09):
  * `import unsloth` MUST precede transformers/trl imports (it patches them).
  * FastLanguageModel.from_pretrained(model_name, max_seq_length, dtype, load_in_4bit)
    FastLanguageModel.get_peft_model(model, r, target_modules, lora_alpha, lora_dropout,
        bias, use_gradient_checkpointing, random_state)
  * trl >= 0.20 renamed SFTConfig.max_seq_length -> max_length and SFTTrainer(tokenizer=)
    -> processing_class=; both spellings are handled below by introspection.
  * Blackwell/sm_120: unsloth#2686 reports a CUDA "unknown error" inside Unsloth's
    offloaded gradient checkpointing on RTX 6000 Pro with some torch/triton combos.
    TRAIN_GRAD_CKPT=unsloth (default) | true (HF checkpointing) | false (none; 96 GB
    is enough for r=16 bf16 LoRA at 8x2048 on an 8B model).

Outputs:
  --out       results/train_lora.json        summary (tokens/s, steps, loss, peak mem, versions)
  --step-log  results/train_lora_steps.jsonl per-step rows
  --started-flag <file>                      touched after the first optimizer step
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import subprocess
import sys
import time


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=env("TRAIN_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--dataset", default=env("TRAIN_DATASET", "yahma/alpaca-cleaned"))
    ap.add_argument("--dataset-split", default=env("TRAIN_SPLIT", "train"))
    ap.add_argument("--max-samples", type=int, default=int(env("TRAIN_MAX_SAMPLES", "30000")))
    ap.add_argument("--minutes", type=float, default=float(env("TRAIN_MINUTES", "15")))
    ap.add_argument("--warmup-seconds", type=float, default=float(env("TRAIN_WARMUP_SECONDS", "60")),
                    help="excluded from the steady-state tokens/s figure")
    ap.add_argument("--seq-len", type=int, default=int(env("TRAIN_SEQ_LEN", "2048")))
    ap.add_argument("--batch", type=int, default=int(env("TRAIN_BATCH", "8")))
    ap.add_argument("--grad-accum", type=int, default=int(env("TRAIN_GRAD_ACCUM", "1")))
    ap.add_argument("--lora-r", type=int, default=int(env("TRAIN_LORA_R", "16")))
    ap.add_argument("--lora-alpha", type=int, default=int(env("TRAIN_LORA_ALPHA", "32")))
    ap.add_argument("--lr", type=float, default=float(env("TRAIN_LR", "2e-4")))
    ap.add_argument("--optim", default=env("TRAIN_OPTIM", "adamw_torch_fused"),
                    help="adamw_torch_fused avoids a bitsandbytes dependency on sm_120; adamw_8bit is the Unsloth default")
    ap.add_argument("--grad-ckpt", default=env("TRAIN_GRAD_CKPT", "unsloth"), choices=["unsloth", "true", "false"])
    ap.add_argument("--packing", action="store_true", default=env("TRAIN_PACKING", "0") == "1")
    ap.add_argument("--seed", type=int, default=int(env("TRAIN_SEED", "3407")))
    ap.add_argument("--out", default=env("TRAIN_OUT", "results/train_lora.json"))
    ap.add_argument("--step-log", default=env("TRAIN_STEP_LOG", "results/train_lora_steps.jsonl"))
    ap.add_argument("--started-flag", default=env("TRAIN_STARTED_FLAG", ""))
    ap.add_argument("--save-adapter", default=env("TRAIN_SAVE_ADAPTER", ""), help="directory; empty = do not save")
    ap.add_argument("--output-dir", default=env("TRAIN_OUTPUT_DIR", "/tmp/lora_out"))
    ap.add_argument("--label", default=env("TRAIN_LABEL", "cotenant"))
    return ap.parse_args()


def nvidia_smi_snapshot() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        rows = []
        for line in out.strip().splitlines():
            idx, name, used, total, util, power = [x.strip() for x in line.split(",")]
            rows.append({"gpu": int(idx), "name": name, "mem_used_mib": int(float(used)),
                         "mem_total_mib": int(float(total)), "util_pct": float(util), "power_w": float(power)})
        return rows
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]


def touch(path: str) -> None:
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(dt.datetime.now(dt.timezone.utc).isoformat())


def resolve_model(name: str) -> str:
    """Prefer weights already on disk from onstart's download set:
    $MODELS_DIR/models.json maps key -> {repo, path} (or key -> path); else $MODELS_DIR/<basename>; else the HF id."""
    if os.path.isdir(name):
        return name
    mdir = os.environ.get("MODELS_DIR", "/workspace/models")
    try:
        with open(os.path.join(mdir, "models.json"), encoding="utf-8") as f:
            m = json.load(f)
        entries = list(m.values()) if isinstance(m, dict) else list(m)
        for e in entries:
            if isinstance(e, dict) and e.get("repo") == name:
                p = e.get("path") or ""
                if p and os.path.exists(os.path.join(p, "config.json")):
                    return p
    except Exception:  # noqa: BLE001
        pass
    cand = os.path.join(mdir, name.split("/")[-1])
    if os.path.exists(os.path.join(cand, "config.json")):
        return cand
    return name


def main() -> int:
    args = parse_args()
    for p in (args.out, args.step_log):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    if os.path.exists(args.step_log):
        os.remove(args.step_log)

    t_import = time.time()
    import unsloth  # noqa: F401  -- must be first
    from unsloth import FastLanguageModel
    import torch
    import transformers
    import trl
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        sys.exit("[lora] CUDA not available in this interpreter (CUDA_VISIBLE_DEVICES=%s)" % os.environ.get("CUDA_VISIBLE_DEVICES"))
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"[lora] device={dev} sm_{cap[0]}{cap[1]} torch={torch.__version__} cuda={torch.version.cuda} arch_list={arch_list} "
          f"unsloth={getattr(unsloth, '__version__', '?')} trl={trl.__version__} transformers={transformers.__version__}",
          file=sys.stderr)
    if cap[0] >= 12 and not any(a in ("sm_120", "compute_120", "sm_121") for a in arch_list):
        # Blackwell needs cu128+ wheels; a cu126 torch would JIT from PTX at best and crash at worst.
        sys.exit("[lora] torch build has no sm_120 kernels (get_arch_list=%s); install with train/install_unsloth.sh TORCH_BACKEND=cu128" % arch_list)

    # ------------------------------------------------------------------ model
    t0 = time.time()
    model_path = resolve_model(args.model)
    if model_path != args.model:
        print(f"[lora] using local weights {model_path} for {args.model}", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    use_gc = {"unsloth": "unsloth", "true": True, "false": False}[args.grad_ckpt]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=use_gc,
        random_state=args.seed,
    )
    load_s = time.time() - t0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] model loaded in {load_s:.1f}s; trainable {trainable/1e6:.1f}M / {total/1e9:.2f}B params", file=sys.stderr)

    # ---------------------------------------------------------------- dataset
    ds = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))
    eos = tokenizer.eos_token or ""

    def to_text(ex):
        instr = ex.get("instruction", "") or ""
        inp = ex.get("input", "") or ""
        out = ex.get("output", "") or ""
        user = instr if not inp else f"{instr}\n\n{inp}"
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}, {"role": "assistant", "content": out}],
                tokenize=False, add_generation_prompt=False)
        except Exception:  # noqa: BLE001  -- no chat template: Alpaca layout
            text = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{user}\n\n### Response:\n{out}{eos}")
        return {"text": text}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"], num_proc=4, desc="format")

    # ------------------------------------------------------------- SFTConfig
    cfg_fields = set(getattr(SFTConfig, "__dataclass_fields__", {}).keys())
    cfg_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=10_000_000,                 # the wall-clock callback stops training
        num_train_epochs=1000,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        optim=args.optim,
        weight_decay=0.0,
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        packing=args.packing,
        dataloader_num_workers=2,
        gradient_checkpointing=(use_gc is True),   # Unsloth's own mode handles itself
        remove_unused_columns=True,
    )
    if "max_length" in cfg_fields:
        cfg_kwargs["max_length"] = args.seq_len
    else:
        cfg_kwargs["max_seq_length"] = args.seq_len
    if "dataset_num_proc" in cfg_fields:
        cfg_kwargs["dataset_num_proc"] = 4
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if (k in cfg_fields or not cfg_fields)}
    training_args = SFTConfig(**cfg_kwargs)

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tk = {"processing_class": tokenizer} if "processing_class" in trainer_params else {"tokenizer": tokenizer}
    trainer = SFTTrainer(model=model, train_dataset=ds, args=training_args, **tk)

    # ------------------------------------------------ token counting collator
    class CountingCollator:
        def __init__(self, inner):
            self.inner = inner
            self.tokens = 0          # non-padding tokens fed to the model
            self.padded = 0          # tokens including padding

        def __call__(self, features):
            batch = self.inner(features)
            am = batch.get("attention_mask") if isinstance(batch, dict) else None
            ids = batch.get("input_ids") if isinstance(batch, dict) else None
            if am is not None:
                self.tokens += int(am.sum())
                self.padded += int(am.numel())
            elif ids is not None:
                self.tokens += int(ids.numel())
                self.padded += int(ids.numel())
            return batch

    counter = CountingCollator(trainer.data_collator)
    trainer.data_collator = counter

    # ---------------------------------------------------- wall-clock callback
    class WallClock(TrainerCallback):
        def __init__(self):
            self.t_start = None
            self.last_t = None
            self.last_tokens = 0
            self.steady_t = None
            self.steady_tokens = None
            self.last_logs = {}
            self.rows = 0
            self.first_step_t = None

        def on_train_begin(self, args_, state, control, **kw):
            self.t_start = self.last_t = time.time()

        def on_log(self, args_, state, control, logs=None, **kw):
            if logs:
                self.last_logs = {k: logs[k] for k in ("loss", "grad_norm", "learning_rate", "mean_token_accuracy") if k in logs}

        def on_step_end(self, args_, state, control, **kw):
            now = time.time()
            elapsed = now - self.t_start
            if state.global_step == 1:
                self.first_step_t = now
                touch(args.started_flag)
            toks = counter.tokens
            row = {"step": state.global_step, "elapsed_s": round(elapsed, 2),
                   "step_time_s": round(now - self.last_t, 3),
                   "tokens_step": toks - self.last_tokens, "tokens_total": toks,
                   "tok_s_step": round((toks - self.last_tokens) / max(now - self.last_t, 1e-6), 1),
                   "tok_s_cum": round(toks / max(elapsed, 1e-6), 1),
                   "mem_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                   "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   **self.last_logs}
            with open(args.step_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.rows += 1
            if self.steady_t is None and elapsed >= args.warmup_seconds:
                self.steady_t, self.steady_tokens = now, toks
            self.last_t, self.last_tokens = now, toks
            if elapsed >= args.minutes * 60:
                control.should_training_stop = True
            return control

    wc = WallClock()
    trainer.add_callback(wc)

    # ------------------------------------------------------------------ train
    smi_before = nvidia_smi_snapshot()
    started = dt.datetime.now(dt.timezone.utc)
    t_train0 = time.time()
    status, error = "ok", None
    try:
        train_out = trainer.train()
        train_loss = getattr(train_out, "training_loss", None)
    except Exception as e:  # noqa: BLE001
        status, error, train_loss = "error", f"{type(e).__name__}: {e}", None
        import traceback
        traceback.print_exc()
    t_train1 = time.time()
    smi_after = nvidia_smi_snapshot()
    train_s = t_train1 - t_train0

    if args.save_adapter and status == "ok":
        model.save_pretrained(args.save_adapter)
        tokenizer.save_pretrained(args.save_adapter)

    steps = trainer.state.global_step
    toks = counter.tokens
    steady = None
    if wc.steady_t is not None and t_train1 - wc.steady_t > 30:
        steady = {"window_s": round(t_train1 - wc.steady_t, 1),
                  "tokens": toks - wc.steady_tokens,
                  "tok_s": round((toks - wc.steady_tokens) / (t_train1 - wc.steady_t), 1)}
    first_last_loss = None
    try:
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            first_last_loss = {"first": losses[0], "last": losses[-1], "n_logged": len(losses)}
    except Exception:  # noqa: BLE001
        pass

    summary = {
        "kind": "lora_train",
        "label": args.label,
        "status": status,
        "error": error,
        "model": args.model,
        "model_path": model_path,
        "dataset": args.dataset,
        "dataset_samples_available": len(ds),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": 0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "precision": "bf16 (load_in_4bit=False)",
        "seq_len": args.seq_len,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "effective_batch_tokens_max": args.batch * args.grad_accum * args.seq_len,
        "packing": args.packing,
        "optim": args.optim,
        "grad_ckpt": args.grad_ckpt,
        "lr": args.lr,
        "requested_minutes": args.minutes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": dev,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "unsloth": getattr(unsloth, "__version__", None), "trl": trl.__version__,
                     "transformers": transformers.__version__, "python": sys.version.split()[0]},
        "trainable_params": trainable,
        "total_params": total,
        "model_load_s": round(load_s, 1),
        "import_s": round(t0 - t_import, 1),
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_wall_s": round(train_s, 1),
        "time_to_first_step_s": round(wc.first_step_t - t_train0, 1) if wc.first_step_t else None,
        "steps": steps,
        "tokens_total": toks,
        "tokens_incl_padding": counter.padded,
        "padding_fraction": round(1 - toks / counter.padded, 4) if counter.padded else None,
        "tok_s_overall": round(toks / train_s, 1) if train_s > 0 else None,
        "tok_s_steady_state": steady,
        "sec_per_step_mean": round(train_s / steps, 3) if steps else None,
        "loss": first_last_loss,
        "training_loss_mean": train_loss,
        "peak_mem_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "nvidia_smi_before": smi_before,
        "nvidia_smi_after": smi_after,
        "step_log": os.path.abspath(args.step_log),
        "adapter_dir": args.save_adapter or None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("status", "steps", "tokens_total", "tok_s_overall",
                                               "tok_s_steady_state", "peak_mem_alloc_gb", "train_wall_s")}, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

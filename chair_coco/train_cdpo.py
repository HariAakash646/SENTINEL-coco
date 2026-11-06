"""Step 3: C-DPO (context-aware DPO, paper Eq. 2) with LoRA, any HF image-text-to-text model.

Reproduces the release's LLaMA-Factory recipe (llamafactory/examples/SENTINEL/Qwen_2_5_VL/qwen_2_5_vl_7B_dpo.sh):
--stage dpo --pref_loss sigmoid --pref_beta 0.1, LoRA r128 on every LM linear layer (alpha = 2r and dropout 0,
LLaMA-Factory's defaults), vision tower and projector frozen, lr 3e-6 cosine, warmup 0, 1 epoch, global batch
64, AdamW, weight decay 0, grad-norm clip 1.0, bf16. The reference policy is the base model (adapter
disabled), as in LLaMA-Factory LoRA DPO. Log-probs are summed over the label tokens.

Each pair is  prompt + context + " " + y  for y in (y_win, y_lose); only the tokens of y carry labels, so
the gradient comes from y_win vs y_lose alone (paper Eq. 2). The release reaches this through
llamafactory/src/llamafactory/data/processor/pairwise.py, which truncates each response after its last "."
token and then masks the context by searching for its token ids. With Qwen's tokenizer a sentence ending
in ".\\n\\n" or ".\\n" has no standalone "." token, so the cut lands inside the context, the search fails, and
the pair trains on context text instead (39% of the released Qwen2.5-VL-7B pairs; see README). Here the
label span is taken from character offsets, which gives the paper's masking for every pair.

FLOPs (flops.py convention) are counted every update and written to train/<out_name>/flops.json (and into
each checkpoint): reference forward, policy forward, LoRA-policy backward, checkpoint recompute, AdamW. At
the end, flops_pipeline.json adds the data generation's count (<run_dir>/flops/generate.json).

    accelerate launch --num_processes 1 chair_coco/train_cdpo.py --config chair_coco/configs/qwen2_5_vl_3b.yaml
"""
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from config import parse_with_config
from flops import (ADAMW_FLOPS_PER_PARAM, FlopLedger, image_token_id, lm, lm_body, load_profile, lora_flops_per_token,
                   profile_meta, vis)
from generate import finished_images
from vlm_common import LORA_TARGETS, build_batch, check_lora_targets, load_image, prompt_text, read_jsonl, response_logps

FLOP_COMPONENTS = ["reference", "policy_forward", "policy_backward", "checkpoint_recompute", "optimizer"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--run-dir")
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-file", default="train_pairs.jsonl", help="under <run_dir>/data")
    p.add_argument("--out-name", default="cdpo")
    p.add_argument("--lr", type=float, default=3.0e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--total-batch-size", type=int, default=64)
    p.add_argument("--per-device-batch-size", type=int, default=1, help="pairs per micro-step")
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--cutoff-len", type=int, default=4096)
    p.add_argument("--save-steps", type=int, default=0, help="0 = final adapter only (the release saves every 1000)")
    p.add_argument("--max-steps", type=int, default=0, help="debug; >0 caps the run")
    p.add_argument("--limit", type=int, default=0, help="debug: first N pairs")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--flop-profile", help="flop_profile.py output for --model")
    return parse_with_config(p, "train")


def text_input_require_grads(model):
    """Call after gradient_checkpointing_enable(). In transformers 5 that calls enable_input_require_grads(), which
    hooks the input embeddings of every sub-model, the vision tower's patch embedding included, so each update
    would also backpropagate through (and recompute) the frozen ViT: ~60% more backward FLOPs on Qwen2.5-VL-3B
    and no LoRA gradient changes. Keep only the language model's hook."""
    model.disable_input_require_grads()
    model.get_input_embeddings().register_forward_hook(lambda module, inputs, output: output.requires_grad_(True))


def sequence_flops(p, seqs, lora_tok):
    """Every pass one micro-step makes over its sequences [(tokens T, image tokens n)], by component.
    Logits are computed at every position (response_logps); the base model and ViT are frozen, LoRA is not."""
    out = dict.fromkeys(FLOP_COMPONENTS, 0.0)
    for T, n in seqs:
        fwd = lm(p, T, T) + vis(p, n)
        out["reference"] += fwd
        out["policy_forward"] += fwd + lora_tok * T
        out["policy_backward"] += lm(p, T, T) + p["b"] * T ** 2 + 2 * lora_tok * T  # attention products: 2 x
        out["checkpoint_recompute"] += lm_body(p, T) + lora_tok * T
    return out


def encode_response(tok, context, sentence):
    """Token ids of (context + " " + sentence) and the index of the first token that holds a character of sentence."""
    text = f"{context} {sentence}" if context else sentence
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start = len(context) + 1 if context else 0
    first = next(i for i, (_, end) in enumerate(enc["offset_mapping"]) if end > start)
    return enc["input_ids"], first


def main():
    args = parse_args()
    profile = load_profile(args.flop_profile)
    acc = Accelerator()
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = read_jsonl(os.path.join(args.run_dir, "data", args.train_file))
    if args.limit:
        rows = rows[: args.limit]
    data = []
    for r in rows:
        c_ids, c_first = encode_response(tok, r["context"], r["chosen"])
        r_ids, r_first = encode_response(tok, r["context"], r["rejected"])
        data.append({"path": r["image_path"], "question": r["question"], "chosen": (c_ids, c_first),
                     "rejected": (r_ids, r_first)})

    world = acc.num_processes
    per_step = args.per_device_batch_size * world
    assert args.total_batch_size % per_step == 0, f"total batch {args.total_batch_size} vs {per_step} per micro-step"
    grad_accum = args.total_batch_size // per_step
    steps_per_epoch = max(1, len(data) // args.total_batch_size)
    total = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    out_dir = os.path.join(args.run_dir, "train", args.out_name)
    if acc.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        print(f"{len(data)} pairs, {total} steps (batch {args.total_batch_size} = {world} x "
              f"{args.per_device_batch_size} x {grad_accum}), lr {args.lr}, beta {args.beta}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        attn_implementation=args.attn_implementation)
    n_targets = check_lora_targets(model)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    text_input_require_grads(model)
    if acc.is_main_process:
        print(f"LoRA on {n_targets} language-model linear layers")
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total), total)
    model, optimizer = acc.prepare(model, optimizer)
    model.train()
    base = acc.unwrap_model(model)
    image_token = image_token_id(base.config)
    lora_tok = lora_flops_per_token(base)
    n_trainable = sum(p.numel() for p in params)
    flops = FlopLedger(os.path.join(out_dir, "flops.json"), resume=False,
                       meta={**profile_meta(profile), "lora_flops_per_token": lora_tok, "n_trainable_params": n_trainable,
                             "n_processes": world})

    def micro_batch(idx):
        items, starts, n_image = [], [], []
        for i in idx:
            ex = data[i]
            enc = processor(text=[prompt_text(processor, ex["question"])], images=[load_image(ex["path"], args.max_pixels)],
                            return_tensors="pt")
            for key in ("chosen", "rejected"):
                ids, first = ex[key]
                assert enc["input_ids"].shape[1] + len(ids) <= args.cutoff_len, "sequence longer than cutoff_len"
                items.append((enc, ids))
                starts.append(first)
                n_image.append(int((enc["input_ids"] == image_token).sum()))
        batch, spans = build_batch(items, pad)
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        seqs = [(end, n) for (_, end), n in zip(spans, n_image)]  # real tokens only, padding excluded
        return {k: v.to(acc.device) for k, v in batch.items()}, spans, starts, seqs

    def shard(step):
        epoch, within = divmod(step, steps_per_epoch)
        perm = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(perm)
        block = perm[within * args.total_batch_size : (within + 1) * args.total_batch_size]
        r, b = acc.process_index, args.per_device_batch_size
        return [block[(k * world + r) * b : (k * world + r + 1) * b] for k in range(grad_accum)]

    def save(name, extra):
        acc.wait_for_everyone()
        if acc.is_main_process:
            path = os.path.join(out_dir, name)
            base.save_pretrained(path)
            json.dump({**vars(args), **extra}, open(os.path.join(path, "train_meta.json"), "w"), indent=1)
            flops.write(os.path.join(path, "flops.json"))
            print(f"saved {path}", flush=True)
        acc.wait_for_everyone()

    log_path = os.path.join(out_dir, "log.jsonl")
    t0 = time.time()
    for step in range(total):
        stats, step_flops, n_seq = [], dict.fromkeys(FLOP_COMPONENTS, 0.0), 0
        for k, idx in enumerate(shard(step)):
            batch, spans, starts, seqs = micro_batch(idx)
            n_seq += len(seqs)
            for key, v in sequence_flops(profile, seqs, lora_tok).items():
                step_flops[key] += v
            with torch.no_grad(), base.disable_adapter():
                ref = response_logps(base, batch, spans, starts)
            with acc.no_sync(model) if k < grad_accum - 1 else nullcontext():
                pol = response_logps(model, batch, spans, starts)
                ratio = pol - ref
                logits = args.beta * (ratio[0::2] - ratio[1::2])
                loss = -F.logsigmoid(logits).mean()
                acc.backward(loss / grad_accum)
            stats.append(torch.stack([loss.detach(), (args.beta * ratio[0::2]).mean().detach(),
                                      (args.beta * ratio[1::2]).mean().detach(), (logits > 0).float().mean(),
                                      pol[0::2].mean().detach(), pol[1::2].mean().detach()]))
        grad_norm = acc.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        s = acc.reduce(torch.stack(stats).mean(0), "mean").tolist()
        *summed, n_seq = acc.reduce(torch.tensor([step_flops[c] for c in FLOP_COMPONENTS[:-1]] + [n_seq],
                                                 dtype=torch.float64, device=acc.device), "sum").tolist()
        step_flops = dict(zip(FLOP_COMPONENTS, summed + [ADAMW_FLOPS_PER_PARAM * n_trainable]))
        for c, v in step_flops.items():
            flops.add(c, v)
        flops.count("updates")
        flops.count("sequences", int(n_seq))
        if acc.is_main_process:
            flops.write()
            rec = {"step": step + 1, "loss": s[0], "rewards/chosen": s[1], "rewards/rejected": s[2],
                   "rewards/margin": s[1] - s[2], "rewards/accuracy": s[3], "logps/chosen": s[4], "logps/rejected": s[5],
                   "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                   "flops/update": sum(step_flops.values()), "flops/cumulative": flops.total(),
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "elapsed_min": (time.time() - t0) / 60}
            print(json.dumps({k: float(f"{v:.5g}") if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if args.save_steps and (step + 1) % args.save_steps == 0 and step + 1 < total:
            save(f"checkpoint-{step + 1}", {"step": step + 1})
    save("checkpoint-final", {"steps": total})
    if acc.is_main_process:
        write_pipeline_flops(args, out_dir, flops)


def write_pipeline_flops(args, out_dir, train_flops):
    """train/<out_name>/flops_pipeline.json: data generation (every image generate.py processed, summed over its
    shards, including images beyond the ones this run's pairs come from) + this training run. `extra` gives the
    generation pro-rated to the images the training pairs used (build_pairs' images_used / images finished)."""
    import glob

    stages = {"train": {"total": train_flops.total(), "by_component": train_flops.by_component,
                        "source": os.path.join(out_dir, "flops.json")}}
    gen_paths = sorted(glob.glob(os.path.join(args.run_dir, "flops", "generate*.json")))
    extra = {}
    if gen_paths:
        gens = [json.load(open(p)) for p in gen_paths]
        by_component, counters = {}, {}
        for g in gens:
            for k, v in g["by_component"].items():
                by_component[k] = by_component.get(k, 0.0) + v
            for k, v in g["counters"].items():
                counters[k] = counters.get(k, 0) + v
        stages["generate"] = {"total": sum(g["total"] for g in gens), "by_component": by_component,
                              "counters": counters, "sources": gen_paths}
        meta_path = os.path.join(args.run_dir, "data", args.train_file.removesuffix(".jsonl") + "_meta.json")
        finished = len(finished_images(args.run_dir, os.path.basename(args.model.rstrip("/"))))
        if os.path.exists(meta_path) and finished:
            used = json.load(open(meta_path))["images_used"]
            share = min(1.0, used / finished)
            extra = {"images_used": used, "images_finished": finished,
                     "generate_prorated_to_images_used": stages["generate"]["total"] * share,
                     "total_with_prorated_generate": stages["generate"]["total"] * share + train_flops.total()}
    else:
        print("note: no flops/generate*.json; flops_pipeline.json has the training stage only", flush=True)
    out = {"total": sum(v["total"] for v in stages.values()), "stages": stages, "extra": extra,
           "convention": train_flops.summary()["convention"], "profile": train_flops.meta["profile"]}
    json.dump(out, open(os.path.join(out_dir, "flops_pipeline.json"), "w"), indent=1)
    print(json.dumps({k: f"{v['total']:.4e}" for k, v in stages.items()} | {"total": f"{out['total']:.4e}"}), flush=True)


if __name__ == "__main__":
    main()

import argparse
import fcntl
import hashlib
import json
import os
import tempfile

import torch
from PIL import Image
from safetensors.torch import save_file
from safetensors import safe_open, SafetensorError
from transformers import AutoModelForImageTextToText, AutoProcessor


def image_state(rundir, image_id, acts):
    """"done", "todo", or "captions": the .json is intact but its activations are not.

    "captions" is what deleting the .safetensors to save space leaves behind. Sampling such
    an image again would overwrite captions the judge has already labelled, and the judge
    stage would not notice (it checks rollout indices, not text), so resume rebuilds the
    activations from the saved token ids instead.
    """
    try:
        doc = json.load(open(os.path.join(rundir, f"{image_id}.json")))
        if doc.get("image_id") != image_id or not isinstance(doc.get("rollouts"), list):
            return "todo"
    except (OSError, ValueError, AttributeError):
        return "todo"
    if not acts:
        return "done"
    try:
        with safe_open(os.path.join(rundir, f"{image_id}.safetensors"), framework="pt") as f:
            return "done" if {"h", "qt"}.issubset(f.keys()) else "captions"
    except (OSError, ValueError, AttributeError, SafetensorError):
        return "captions"


def image_complete(rundir, image_id, acts):
    """Only skip a resumable image when all of its artifacts are readable."""
    return image_state(rundir, image_id, acts) == "done"


def plan_images(images, rundir, acts, overwrite):
    """(to_sample, to_rebuild): images to caption from scratch, and images whose saved
    captions only need their activations teacher-forced again."""
    if overwrite:
        return list(images), []
    to_sample, to_rebuild = [], []
    for m in images:
        state = image_state(rundir, m["image_id"], acts)
        if state == "todo":
            to_sample.append(m)
        elif state == "captions":
            to_rebuild.append(m)
    return to_sample, to_rebuild


def saved_caption_ids(doc):
    """The token ids an image's captions were teacher-forced with, checked against the
    offsets the .json records, since h is those captions' rows concatenated in order."""
    caps, off = [], 0
    for row in doc["rollouts"]:
        ids = row["token_ids"]
        if row["offset"] != off or row["n_tokens"] != len(ids):
            raise ValueError(f"image {doc['image_id']} rollout {row['rollout']}: offset "
                             f"{row['offset']} / n_tokens {row['n_tokens']} disagree with "
                             f"{len(ids)} token ids at offset {off}")
        caps.append(ids)
        off += len(ids)
    if not caps or off != doc["n_tokens"]:
        raise ValueError(f"image {doc['image_id']}: {len(caps)} captions, {off} tokens, "
                         f"but the .json records n_tokens={doc.get('n_tokens')}")
    return caps


def extract_layers(n_layers, have=None):
    """The hidden_states layers to store: 3L/4 only, the layer every reward probe reads.

    Five layers [1, L/4, L/2, 3L/4, L] used to be stored, at 5x the disk. A dump whose
    meta.json already records layers for this depth keeps them, so a resume or a rebuild
    never mixes two layouts in a directory that rollout_data.py reads with one meta.
    """
    if have and have.get("layers") and have.get("n_hidden_layers") == n_layers:
        return list(have["layers"])
    return [(3 * n_layers) // 4]


def meta_disagreements(have, want):
    """Keys where an existing meta.json differs from the representation about to be written."""
    keys = ("model", "adapter", "max_pixels", "layers", "hidden_size", "n_hidden_layers")
    return [f"{k}: meta.json has {have.get(k)!r}, this run would use {want[k]!r}"
            for k in keys if have.get(k) != want[k]]


def completion_ids(ids, stop_ids):
    """One sampled caption's tokens, cut at its first stop token. False => cut off by the budget.

    With num_return_sequences > 1, generate pads a sequence that finished early with
    pad_token_id out to the longest one. Qwen pads with <|endoftext|>, itself a stop id, so
    stripping trailing stop ids was enough there; LLaVA (</s> then <pad> 32001) and Gemma 4
    (<end_of_turn> then <pad> 0) pad with a token that is NOT a stop id, which left
    "</s><pad><pad>..." in the caption text, the token count and the cached hidden states.
    """
    for i, token in enumerate(ids):
        if token in stop_ids:
            return ids[:i], True
    return ids, False


def repeat_vision(inp, n):
    """Tile one image's vision tensors for n captions, whatever layout the processor used.

    Qwen packs patches as [num_patches, dim] and carries image_grid_thw; LLaVA passes a
    batched [1, 3, H, W] tensor and no grid at all. Gemma 4 passes [1, num_patches, dim]
    plus image_position_ids, the (x, y) patch coordinates its vision tower requires --
    drop it and the tower reads the arg's default and dies on `'bool' has no attribute
    'all'`. Repeating along dim 0 is correct for all three, so the caller never has to
    know which family it loaded.

    NOT tiled: Gemma 4's mm_token_type_ids. It is text-aligned rather than patch-aligned,
    and it only feeds the bidirectional vision mask that text_config.use_bidirectional_-
    attention == "vision" selects. E4B/E2B leave that None (causal), so omitting it is a
    no-op. A larger Gemma 4 that sets it WOULD need the types padded to the teacher-forced
    length, the way trl_compat.py does for Qwen M-RoPE -- check before changing family.
    """
    out = {}
    for key in ("pixel_values", "image_grid_thw", "image_sizes", "pixel_attention_mask",
                "image_position_ids"):
        if key in inp:
            tensor = inp[key]
            out[key] = tensor.repeat(n, *([1] * (tensor.dim() - 1)))
    return out

MODEL = "llava-hf/llava-1.5-7b-hf"
PROMPT = "Describe this image in detail."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--adapter", default="",
                   help="optional PEFT adapter to apply to --model for generation")
    p.add_argument("--manifest", default="data/coco_train.json")
    p.add_argument("--rundir", default="runs/rollouts_train")
    p.add_argument("--num_rollouts", type=int, default=8, help="rollouts per image")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N images (0 = all); for pilots")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-pixels", type=int, default=0,
                   help="cap processor pixels (Qwen); 0 leaves the processor default, "
                        "which is what fixed-resolution models such as LLaVA need")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true",
                   help="use deterministic greedy decoding (requires --n 1); "
                        "temperature and top-p are then inactive")
    p.add_argument("--no-acts", action="store_true",
                   help="sample text only: skip the teacher-forcing pass and write no "
                        "safetensors. Answers budget/label questions at zero storage cost.")
    p.add_argument("--defer-acts", action="store_true",
                   help="sample text only, but for a dump that WILL carry activations: "
                        "meta.json says acts=true, and a later plain resume teacher-forces "
                        "them from the saved token ids, as after deleting the .safetensors. "
                        "--no-acts instead marks the dump text-only for good")
    p.add_argument("--overwrite", action="store_true",
                   help="redo images that already have a .json (default: skip, i.e. resume)")
    p.add_argument("--acts-only", action="store_true",
                   help="only rebuild activations from saved captions; refuse to sample. "
                        "A plain resume does this too, but also samples images it has no "
                        ".json for")
    return p.parse_args()


def teacher_force(model, inp, cap_ids, layers, pad_id):
    """One batched forward for all of an image's captions.

    Right-padding is safe here: attention is causal, so a position never attends to anything
    to its right, and every state we read lies at index < Tp + len(caption). The attention
    mask still marks the pad so nothing downstream counts it.

    Returns ([n_tokens, n_layers, d] fp16 CPU per caption, qt [n_layers, d] fp16 CPU).
    """
    dev = inp["input_ids"].device
    N, Tp = len(cap_ids), inp["input_ids"].shape[1]
    Amax = max(len(c) for c in cap_ids)

    ids = torch.full((N, Tp + Amax), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros((N, Tp + Amax), dtype=torch.long, device=dev)
    ids[:, :Tp] = inp["input_ids"]
    for i, c in enumerate(cap_ids):
        ids[i, Tp:Tp + len(c)] = torch.tensor(c, device=dev)
        mask[i, :Tp + len(c)] = 1

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    **repeat_vision(inp, N),
                    output_hidden_states=True, logits_to_keep=1)
    hs = out.hidden_states

    parts = [torch.stack([hs[l][i, Tp:Tp + len(c)] for l in layers], dim=1).to(torch.float16).cpu()
             for i, c in enumerate(cap_ids)]
    qt = torch.stack([hs[l][0, Tp - 1] for l in layers]).to(torch.float16).cpu()
    return parts, qt


def prompt_inputs(processor, path, device):
    """The prompt every caption in a rollout dump was sampled from, for one image."""
    image = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=[image], return_tensors="pt").to(device)


def main():
    args = parse_args()
    if args.greedy and args.num_rollouts != 1:
        raise SystemExit("--greedy requires --num_rollouts 1")
    if args.acts_only and (args.no_acts or args.overwrite):
        raise SystemExit("--acts-only rebuilds activations from saved captions; it cannot be "
                         "combined with --no-acts or --overwrite")
    if args.defer_acts and (args.no_acts or args.acts_only):
        raise SystemExit("--defer-acts cannot be combined with --no-acts or --acts-only")
    # Deferring writes captions exactly as --no-acts does; only meta.json's acts differs.
    acts_now = not (args.no_acts or args.defer_acts)
    os.makedirs(args.rundir, exist_ok=True)
    torch.manual_seed(1234 + args.shard)

    images = json.load(open(args.manifest))
    if args.limit:
        images = images[:args.limit]
    images = [m for i, m in enumerate(images) if i % args.num_shards == args.shard]
    n_total = len(images)
    to_sample, to_rebuild = plan_images(images, args.rundir, acts_now, args.overwrite)
    if args.acts_only and to_sample:
        first = ", ".join(str(m["image_id"]) for m in to_sample[:10])
        raise SystemExit(f"--acts-only: {len(to_sample)} images have no usable .json to rebuild "
                         f"from (first: {first})")

    proc_kwargs = {"max_pixels": args.max_pixels} if args.max_pixels else {}
    processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
    # Shards on one node load one at a time. Weights stored in another dtype are materialised
    # in host RAM until .to(device) frees them: LLaVA-1.5-13B is fp16 on disk, ~26 GB per
    # shard, and four shards loading at once were OOM-killed in an 80 GB job.
    with open(os.path.join(tempfile.gettempdir(), "rollout_acts.load.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16).to(args.device)
    if args.adapter:
        from peft import PeftModel
        from peft.utils.save_and_load import set_peft_model_state_dict
        from safetensors.torch import load_file

        model = PeftModel.from_pretrained(model, args.adapter)
        adapter_state = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
        set_peft_model_state_dict(model, adapter_state, adapter_name="default")

        saved_lora_b = [tensor for name, tensor in adapter_state.items()
                        if ".lora_B." in name]
        loaded_lora_b = [parameter for name, parameter in model.named_parameters()
                         if ".lora_B.default." in name]
        saved_nonzero = sum(tensor.count_nonzero().item() for tensor in saved_lora_b)
        loaded_nonzero = sum(tensor.count_nonzero().item() for tensor in loaded_lora_b)
        if not saved_lora_b or not loaded_lora_b or loaded_nonzero != saved_nonzero:
            raise RuntimeError(
                "PEFT adapter tensor restore failed: "
                f"saved LoRA-B nonzero={saved_nonzero}, loaded={loaded_nonzero}"
            )
        print(f"loaded PEFT adapter: {args.adapter}; "
              f"verified {loaded_nonzero} nonzero LoRA-B weights", flush=True)
    tok = processor.tokenizer
    # Stop ids come from the model, never a hardcoded list: Qwen ends on <|im_end|> and
    # <|endoftext|>, LLaVA on </s>. generation_config carries whichever applies.
    configured = model.generation_config.eos_token_id
    stop_ids = {tok.eos_token_id,
                *(configured if isinstance(configured, (list, tuple)) else [configured])}
    stop_ids.discard(None)
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    meta_path = os.path.join(args.rundir, "meta.json")
    try:
        have = json.load(open(meta_path))
    except (OSError, ValueError):
        have = None

    # hidden_states[l] is the output of layer l
    L = model.config.text_config.num_hidden_layers
    layers = extract_layers(L, have)
    d = model.config.text_config.hidden_size

    n_done = n_total - len(to_sample) - len(to_rebuild)
    print(f"shard {args.shard}: {n_total} images, {n_done} already done, "
          f"{len(to_rebuild)} to rebuild from saved captions, {len(to_sample)} to sample  "
          f"(acts={'deferred' if args.defer_acts else 'off' if args.no_acts else 'on'}, "
          f"layers={layers}, d={d})", flush=True)

    # Rebuilt activations must be in the representation the saved captions were first
    # teacher-forced in, or the probe would be scored against states it never saw.
    if to_rebuild:
        if have is None:
            raise SystemExit(f"cannot rebuild activations without a readable {meta_path}: it "
                             "records the model and layers the saved captions were run through")
        bad = meta_disagreements(have, {
            "model": args.model, "adapter": args.adapter or None,
            "max_pixels": args.max_pixels or None, "layers": layers,
            "hidden_size": d, "n_hidden_layers": L})
        if bad:
            raise SystemExit("refusing to rebuild activations in a different representation: "
                             + "; ".join(bad))

    # A pure rebuild leaves meta.json as it is: it still describes how the captions were made.
    if args.shard == 0 and (to_sample or not os.path.exists(meta_path)):
        json.dump({"model": args.model, "adapter": args.adapter or None,
                   "manifest": args.manifest,
                   "manifest_sha256": hashlib.sha256(open(args.manifest, "rb").read()).hexdigest(),
                   "prompt": PROMPT,
                   "n_rollouts_per_image": args.num_rollouts,
                   "decoding": "greedy" if args.greedy else "sampling",
                   "temperature": None if args.greedy else args.temperature,
                   "top_p": None if args.greedy else args.top_p,
                   "do_sample": not args.greedy,
                   "max_new_tokens": args.max_new_tokens,
                   "max_pixels": args.max_pixels or None,
                   "layers": layers, "hidden_size": d, "n_hidden_layers": L,
                   "acts": not args.no_acts, "seed_base": 1234},
                  open(meta_path, "w"), indent=1)

    for img_i, m in enumerate(to_rebuild):
        stem = os.path.join(args.rundir, str(m["image_id"]))
        doc = json.load(open(stem + ".json"))
        if doc["path"] != m["path"]:
            raise SystemExit(f"image {m['image_id']}: the .json was made from {doc['path']}, "
                             f"the manifest now says {m['path']}")
        parts, qt = teacher_force(model, prompt_inputs(processor, m["path"], args.device),
                                  saved_caption_ids(doc), layers, pad_id)
        # Written aside and renamed, so an interrupted rebuild never leaves a torn file.
        save_file({"h": torch.cat(parts), "qt": qt}, stem + ".safetensors.tmp")
        os.replace(stem + ".safetensors.tmp", stem + ".safetensors")
        if (img_i + 1) % 25 == 0:
            print(f"{img_i + 1}/{len(to_rebuild)} images rebuilt", flush=True)

    n_rollouts = n_tokens = 0
    for img_i, m in enumerate(to_sample):
        inp = prompt_inputs(processor, m["path"], args.device)
        Tp = inp["input_ids"].shape[1]

        generation = {"max_new_tokens": args.max_new_tokens,
                      "do_sample": not args.greedy,
                      "num_return_sequences": args.num_rollouts}
        if not args.greedy:
            generation.update(temperature=args.temperature, top_p=args.top_p)
        with torch.no_grad():
            gen = model.generate(**inp, **generation)

        caps = []
        for s in range(args.num_rollouts):
            ids, finished = completion_ids(gen[s, Tp:].tolist(), stop_ids)
            text = tok.decode(ids)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            if len(enc["input_ids"]) == 0:
                continue
            caps.append((s, text, enc["input_ids"], enc["offset_mapping"], finished))

        parts, qt = ([], None) if not acts_now else teacher_force(
            model, inp, [c[2] for c in caps], layers, pad_id)

        rollouts, off = [], 0
        for s, text, ids, spans, finished in caps:
            rollouts.append({"rollout": s, "offset": off, "n_tokens": len(ids),
                             "finished": finished, "text": text, "token_ids": ids,
                             "token_char_spans": spans})
            off += len(ids)

        stem = os.path.join(args.rundir, str(m["image_id"]))
        if acts_now:
            save_file({"h": torch.cat(parts), "qt": qt}, stem + ".safetensors")
        elif args.defer_acts and os.path.exists(stem + ".safetensors"):
            # States of the captions being replaced; left in place they would read as done.
            os.remove(stem + ".safetensors")
        json.dump({"image_id": m["image_id"], "path": m["path"],
                   "n_rollouts": len(rollouts), "n_tokens": off, "rollouts": rollouts},
                  open(stem + ".json", "w"))

        n_rollouts += len(rollouts)
        n_tokens += off
        if (img_i + 1) % 25 == 0:
            print(f"{img_i + 1}/{len(to_sample)} images  rollouts={n_rollouts}  "
                  f"tokens={n_tokens}", flush=True)

    print(f"shard {args.shard} done -> {args.rundir}  rebuilt={len(to_rebuild)} "
          f"sampled={len(to_sample)} rollouts={n_rollouts} tokens={n_tokens}")


if __name__ == "__main__":
    main()

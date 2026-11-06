"""Measure a VLM's forward FLOPs once and fit the cost profile that flops.py counts with. The json has the
format of ~/VESRFT/scripts/flop_profile.py, so one profile per backbone serves every baseline.
Same file in ~/SENTINEL/chair_coco and ~/Nullu/chair_coco.

torch's FlopCounterMode counts the matmul / attention FLOPs of real forward passes (eager attention, so the
T x T scores are explicit matmuls). From them:

    lm(T, L) = a*T + b*T^2 + h*L   text-only forwards of T random tokens; h from varying logits_to_keep
    vis(n)   = c*n + e*n^2         forwards on real COCO train2014 images (also at half size, so a resizing
                                   processor gives a second token count) minus their language-model part;
                                   e = 0 when every image becomes the same number of tokens (LLaVA, Gemma 3)

and checks the fit on held-out teacher-forced passes (image + prompt + caption, and text only), storing the
relative error. Images go through vlm_common.load_image, the resize every stage and the eval use.
No GPU needed: a 3B model takes a few minutes on CPU in fp32.

    CUDA_VISIBLE_DEVICES= python chair_coco/flop_profile.py --model Qwen/Qwen2.5-VL-3B-Instruct
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForImageTextToText, AutoProcessor

from flops import image_token_id, lm, vis
from vlm_common import load_image, prompt_text

PROMPT = "Describe this image in detail."
CAPTION = ("The image shows a man riding a bicycle down a city street next to a red bus. A woman with an "
           "umbrella waits at the crosswalk, and two cars are parked by the curb near a fire hydrant. ")


def measure(model, inputs, keep):
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(**inputs, logits_to_keep=keep, use_cache=False)
    return float(counter.get_total_flops())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--image-dir", default="/scratch/balaji/cbms_for_mllms/datasets/coco_2014/mscoco/images/train2014")
    p.add_argument("--n-images", type=int, default=4, help="for the vision fit; 2 more are held out for the check")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", default=None, help="default: /scratch/sathwik/flop_profiles/<model name>.json")
    args = p.parse_args()
    out = args.out or f"/scratch/sathwik/flop_profiles/{os.path.basename(args.model.rstrip('/'))}.json"
    torch.set_num_threads(args.threads)
    start = time.time()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    image_token = image_token_id(model.config)

    # Language model: text-only forwards of random ordinary tokens.
    gen = torch.Generator().manual_seed(0)

    def text(T):
        ids = torch.randint(1000, 20000, (1, T), generator=gen)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    probe = text(128)
    h = (measure(model, probe, 128) - measure(model, probe, 1)) / 127
    lengths = [32, 128, 512, 1024]
    flops = np.array([measure(model, text(T), 1) - h for T in lengths])
    X = np.array([[T, T * T] for T in lengths], dtype=float)
    (a, b), *_ = np.linalg.lstsq(X, flops, rcond=None)
    text_fit_err = float(np.max(np.abs(X @ [a, b] - flops) / flops))

    # Vision: real images, the first two also at half size.
    names = sorted(f for f in os.listdir(args.image_dir) if f.endswith(".jpg"))[: args.n_images + 2]
    images = [load_image(os.path.join(args.image_dir, f), args.max_pixels) for f in names]
    fit_images, held_out = images[: args.n_images], images[args.n_images:]
    fit_images = fit_images + [im.resize((im.width // 2, im.height // 2)) for im in fit_images[:2]]
    chat = prompt_text(processor, PROMPT)
    n_tok, v_flops = [], []
    for image in fit_images:
        enc = processor(text=[chat], images=[image], return_tensors="pt")
        T = enc["input_ids"].shape[1]
        n_tok.append(int((enc["input_ids"] == image_token).sum()))
        v_flops.append(measure(model, enc, 1) - (a * T + b * T * T + h))
    n_tok, v_flops = np.array(n_tok, dtype=float), np.array(v_flops)
    if len(set(n_tok)) > 1:
        (c, e), *_ = np.linalg.lstsq(np.stack([n_tok, n_tok ** 2], 1), v_flops, rcond=None)
        vision_mode = "c*n + e*n^2"
    else:
        c, e, vision_mode = float(v_flops.mean() / n_tok[0]), 0.0, "constant (fixed image token count)"
    profile = {"a": float(a), "b": float(b), "h": float(h), "c": float(c), "e": float(e)}

    # Held-out check: teacher-forced passes with logits everywhere, as extract.py and train_cdpo.py run them.
    checks = []
    for image in held_out:
        enc = processor(text=[chat + CAPTION * 3], images=[image], return_tensors="pt")
        T, n = enc["input_ids"].shape[1], int((enc["input_ids"] == image_token).sum())
        checks.append({"tokens": T, "image_tokens": n, "measured": measure(model, enc, 0),
                       "predicted": lm(profile, T, T) + vis(profile, n)})
    enc = text(700)
    checks.append({"tokens": 700, "image_tokens": 0, "measured": measure(model, enc, 0),
                   "predicted": lm(profile, 700, 700)})
    for ch in checks:
        ch["rel_error"] = (ch["predicted"] - ch["measured"]) / ch["measured"]

    result = {
        "model": args.model, **profile, "vision_fit": vision_mode, "max_pixels": args.max_pixels,
        "n_params": sum(q.numel() for q in model.parameters()),
        "text_fit_max_rel_error": text_fit_err,
        "vision_points": [{"image_tokens": int(n), "flops": float(f)} for n, f in zip(n_tok, v_flops)],
        "held_out_checks": checks,
        "max_abs_rel_error": max(abs(ch["rel_error"]) for ch in checks),
        "method": "torch.utils.flop_counter.FlopCounterMode, eager attention, fp32, cpu",
        "torch": torch.__version__, "seconds": round(time.time() - start, 1),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("a", "b", "h", "c", "e", "vision_fit", "n_params",
                                              "text_fit_max_rel_error", "max_abs_rel_error")}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

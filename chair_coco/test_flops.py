"""Checks SENTINEL's FLOP accounting against torch's FlopCounterMode on the real backbone, on CPU (fp32, eager).

    generation  the generator's prompt count (VLLMGenerator._prompt_tokens) equals the processor's, and an HF
                greedy generate of exactly C tokens after prompt + context measures prefill() + decode();
    train       one C-DPO micro-step (reference forward, LoRA policy forward + backward with gradient
                checkpointing) measures what train_cdpo.sequence_flops() predicts;
    tools       FlopCounterMode runs through the release's YOLO11x, fastcoref, FACTUAL T5 and GroundingDINO calls
                (needs the SENTINEL env); prints the FLOPs and the wall-clock overhead of counting.

    CUDA_VISIBLE_DEVICES= python chair_coco/test_flops.py --config chair_coco/configs/qwen2_5_vl_3b.yaml --part generation train
    CUDA_VISIBLE_DEVICES= $SENTINEL_PY chair_coco/test_flops.py --config chair_coco/configs/qwen2_5_vl_3b.yaml --part tools
"""
import argparse
import logging
import os
import tempfile
import time
from types import SimpleNamespace

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import parse_with_config
from flops import decode, image_token_id, load_profile, lora_flops_per_token, measured_flops, prefill
from vlm_common import LORA_TARGETS, build_batch, load_image, prompt_text, read_jsonl, response_logps

CONTEXT = "The image shows a man riding a bicycle down a city street. A red bus is parked behind him."
TOL = 2e-3


def check(name, measured, predicted):
    err = (predicted - measured) / measured
    print(f"  {name}: measured {measured:.5e} predicted {predicted:.5e} ({err:+.2e})", flush=True)
    assert abs(err) < TOL, f"{name}: relative error {err:+.2e}"


def load_model(model_id):
    return AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager")


def part_generation(args, profile, image):
    from generate import VLLMGenerator

    processor = AutoProcessor.from_pretrained(args.model)
    model = load_model(args.model).eval()
    fake = SimpleNamespace(processor=processor, image_token=image_token_id(model.config), _prompt_cache={})
    fake._template = lambda u: VLLMGenerator._template(fake, u)
    question = "Describe this image in detail."
    for context in ("", CONTEXT):
        n_prompt, n_image = VLLMGenerator._prompt_tokens(fake, question, context, image)
        enc = processor(text=[fake._template(question) + context], images=[image], return_tensors="pt")
        assert n_prompt == enc["input_ids"].shape[1], (n_prompt, enc["input_ids"].shape[1])
        assert n_image == int((enc["input_ids"] == fake.image_token).sum())
        for C in (1, 20):
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                model.generate(**enc, max_new_tokens=C, min_new_tokens=C, do_sample=False)
            check(f"generate P={n_prompt} n_img={n_image} C={C}", fc.get_total_flops(),
                  prefill(profile, n_prompt, n_image) + decode(profile, n_prompt, C))


def part_train(args, profile, image_path):
    from peft import LoraConfig, get_peft_model

    from train_cdpo import FLOP_COMPONENTS, encode_response, sequence_flops, text_input_require_grads

    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    model = load_model(args.model)
    image_token = image_token_id(model.config)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=128, lora_alpha=256, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    text_input_require_grads(model)
    model.train()
    lora_tok = lora_flops_per_token(model)
    enc = processor(text=[prompt_text(processor, "Describe this image in detail.")],
                    images=[load_image(image_path, args.max_pixels)], return_tensors="pt")
    ids, first = encode_response(tok, CONTEXT, "A woman with an umbrella waits at the crosswalk.")
    batch, spans = build_batch([(enc, ids)], tok.pad_token_id or tok.eos_token_id)
    T, n = spans[0][1], int((enc["input_ids"] == image_token).sum())

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad(), model.disable_adapter():
            ref = response_logps(model, batch, spans, [first])
    ref_measured = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        pol = response_logps(model, batch, spans, [first])
    fwd_measured = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        (pol - ref).sum().backward()
    bwd_measured = fc.get_total_flops()
    pred = sequence_flops(profile, [(T, n)], lora_tok)
    print(f"  T={T} n_img={n} lora_flops/token={lora_tok:.4e} components={FLOP_COMPONENTS[:-1]}")
    check("reference forward", ref_measured, pred["reference"])
    check("policy forward (LoRA)", fwd_measured, pred["policy_forward"])
    check("policy backward + checkpoint recompute", bwd_measured, pred["policy_backward"] + pred["checkpoint_recompute"])

    # The LoRA gradients are the same as with transformers' own hooks (gradient_checkpointing_enable ->
    # enable_input_require_grads), which also make the frozen ViT backpropagate.
    grads = {n: q.grad.clone() for n, q in model.named_parameters() if q.requires_grad}
    model.zero_grad(set_to_none=True)
    model.enable_input_require_grads()
    with torch.no_grad(), model.disable_adapter():
        ref = response_logps(model, batch, spans, [first])
    with FlopCounterMode(display=False) as fc:
        (response_logps(model, batch, spans, [first]) - ref).sum().backward()
    worst = max(float((q.grad - grads[n]).abs().max() / grads[n].abs().max().clamp_min(1e-30))
                for n, q in model.named_parameters() if q.requires_grad)
    print(f"  with the ViT hooked: forward + backward {fc.get_total_flops():.5e} FLOPs "
          f"(vs {fwd_measured + bwd_measured:.5e}); LoRA grads max rel. diff {worst:.2e}")
    assert worst < 1e-4, worst


def part_tools(args, image_path):
    """The four networks generate.py measures, called as the release calls them, with and without counting."""
    from generate import install_patches

    logger = logging.getLogger("test_flops")
    ns = argparse.Namespace(**{**vars(args), "name": "test", "run_dir": tempfile.mkdtemp(), "seed": 42})
    G = install_patches(ns, logger)
    from model.auxiliary.dataset import DataPoint
    from model.auxiliary.datastate import DataStateForBuildDataset

    GV = G.GVars
    dino = G.DINO("base", model_dir=GV.model_dir, device=GV.alter_device, logger=logger)
    yolo = G.YoloModel("yolo11x", model_dir=GV.model_dir, logger=logger)
    sg = G.SGParser(G.DEBUG, "base", GV.model_dir, device=GV.alter_device, logger=logger)
    spacy = G.SpacyModel(model_size="md", model_dir=GV.model_dir, device=GV.alter_device, logger=logger)
    sents = [["It has a wicker basket on the front.", "A dog sits next to it."]]
    calls = {
        "yolo11x": lambda: G.yolo_detect(yolo, [DataStateForBuildDataset(data=DataPoint("0", image_path, "q"))]),
        "fastcoref_lingmess": lambda: G.resolve_corefs(spacy, sents, ["A bicycle is parked by a wall."], 1),
        "factual_t5_parser": lambda: sg.pharse(sents[0], force_list=True),
        "grounding_dino_base": lambda: dino._detect([load_image(image_path, 0)], ["bicycle . dog . basket ."]),
    }
    for name, call in calls.items():
        call()  # warm-up (lazy loads)
        t0 = time.time()
        call()
        plain = time.time() - t0
        t0 = time.time()
        _, flops = measured_flops(call)
        counted = time.time() - t0
        print(f"  {name}: {flops:.4e} FLOPs, {plain:.2f} s plain, {counted:.2f} s counted", flush=True)
        assert flops > 0, name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--run-dir")
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--flop-profile")
    p.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE"))
    p.add_argument("--tools-dir")
    for key in ("--batch-size", "--dino-chunk", "--max-model-len", "--max-num-seqs"):  # generate section, unused here
        p.add_argument(key, type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--image", default="/scratch/balaji/cbms_for_mllms/datasets/coco_2014/mscoco/images/train2014/"
                                      "COCO_train2014_000000000009.jpg")
    p.add_argument("--part", nargs="+", default=["generation", "train"], choices=["generation", "train", "tools"])
    p.add_argument("--threads", type=int, default=16)
    args = parse_with_config(p, "generate")
    torch.set_num_threads(args.threads)
    profile = load_profile(args.flop_profile)
    image = load_image(args.image, args.max_pixels)
    if "generation" in args.part:
        print("generation", flush=True)
        part_generation(args, profile, image)
    if "train" in args.part:
        print("train", flush=True)
        part_train(args, profile, args.image)
    if "tools" in args.part:
        print("tools", flush=True)
        part_tools(args, args.image)
    print("all checks passed")


if __name__ == "__main__":
    main()

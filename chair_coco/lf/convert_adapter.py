"""Make a LoRA adapter trained by LLaMA-Factory (transformers 4.51) loadable by the eval (transformers 5.x).

transformers >= 4.52 renamed the language-model layers of VLMs such as Qwen2.5-VL (model.layers.N... ->
model.language_model.layers.N...). LLaMA-Factory stores the LoRA's target modules as full module names, so with the
eval's transformers PEFT would not find them, and the adapter's tensors would not load. This rewrites both to the
names of the model as the installed transformers builds it, and refuses unless every tensor of the adapter maps to
exactly one LoRA tensor of that model with the same shape (built on the meta device: no weights are loaded). Values
are copied unchanged. Run in the pipeline's main venv.

    python chair_coco/lf/convert_adapter.py --base-model Qwen/Qwen2.5-VL-3B-Instruct --adapter LF_OUT --out DIR
"""
import argparse
import json
import os
import re
import shutil

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForImageTextToText

TAIL = re.compile(r"(?:^|\.)(layers\.\d+\..*)$")


def tail(name):
    m = TAIL.search(name)
    return m.group(1) if m else name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True, help="LLaMA-Factory output dir (adapter_config.json + safetensors)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    tensors = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))

    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(a.base_model))
    names = [name for name, _ in model.named_modules()]
    by_tail = {}
    for name in names:
        by_tail.setdefault(tail(name), []).append(name)

    def new_module(old):
        """The eval model's module for a name in the adapter: a full transformers-4.51 name (model.layers.N...) is
        matched by its layers.N... path; PEFT may also store target_modules as the shortest unique name endings
        ("34.mlp.up_proj"), which stay endings of the same modules under the new names. Exactly one must match."""
        hits = by_tail.get(tail(old), []) if TAIL.search(old) else []
        if not hits:
            hits = [n for n in names if n == old or n.endswith("." + old)]
        if len(hits) != 1:
            raise SystemExit(f"target module {old!r}: {len(hits)} modules of the eval model match it")
        return hits[0]

    # The LoRA'd modules, from the tensor names (always full module names). target_modules in the config is not used:
    # PEFT may have stored it as name endings, where one ending ("k_proj") covers many modules.
    prefix = "base_model.model."
    modules = sorted({k[len(prefix):].partition(".lora_")[0] for k in tensors if k.startswith(prefix)})
    if len(modules) * 2 != len(tensors):
        raise SystemExit(f"expected lora_A and lora_B for each module: {len(tensors)} tensors, {len(modules)} modules")
    mapping = {m: new_module(m) for m in modules}
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit("two adapter modules map to the same module of the eval model")
    keep = {k: cfg[k] for k in ("r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora", "fan_in_fan_out")
            if k in cfg}
    peft_model = get_peft_model(model, LoraConfig(**keep, target_modules=sorted(mapping.values())))
    expected = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(peft_model).items()}

    def new_key(key):
        prefix, rest = "base_model.model.", key
        assert rest.startswith(prefix), key
        module, _, param = rest[len(prefix):].partition(".lora_")
        return f"{prefix}{mapping.get(module) or new_module(module)}.lora_{param}"

    converted = {new_key(k): v for k, v in tensors.items()}
    missing = sorted(set(expected) - set(converted))
    extra = sorted(set(converted) - set(expected))
    shape = [k for k in converted if k in expected and tuple(converted[k].shape) != expected[k]]
    if missing or extra or shape or len(converted) != len(tensors):
        raise SystemExit(f"adapter does not match the eval model: {len(missing)} missing (e.g. {missing[:2]}), "
                         f"{len(extra)} unexpected (e.g. {extra[:2]}), {len(shape)} with another shape")

    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in converted.items()}, os.path.join(a.out, "adapter_model.safetensors"),
              metadata={"format": "pt"})
    cfg_out = dict(cfg, target_modules=sorted(mapping.values()), base_model_name_or_path=a.base_model)
    json.dump(cfg_out, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=2)
    for extra_file in ("trainer_state.json", "all_results.json", "train_results.json", "training_loss.png"):
        if os.path.exists(os.path.join(a.adapter, extra_file)):
            shutil.copy(os.path.join(a.adapter, extra_file), a.out)
    renamed = sum(k != v for k, v in mapping.items())
    json.dump({"source": os.path.abspath(a.adapter), "tensors": len(converted), "target_modules": len(mapping),
               "renamed_modules": renamed, "example": next(iter(mapping.items()))},
              open(os.path.join(a.out, "conversion.json"), "w"), indent=1)
    print(f"adapter: {len(converted)} tensors, {len(mapping)} target modules ({renamed} renamed for this transformers), "
          f"all matched with equal shapes -> {a.out}")


if __name__ == "__main__":
    main()

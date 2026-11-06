"""FLOP accounting for the chair_coco baselines, in the convention of ~/VESRFT (vesrft/flops.py), so the
training cost of SENTINEL, Nullu and VES-RFT can be compared. Same file in ~/SENTINEL/chair_coco and ~/Nullu/chair_coco.

Convention ("model FLOPs", as in PaLM's MFU accounting):
- matmul FLOPs of a forward pass on real tokens, a multiply-add counted as 2; padding is not counted;
- attention counted dense: the full T x T score matrix, as an eager forward computes it;
- backward = 2 x forward for weights that get gradients (activation + weight grads), 1 x for frozen weights
  that only pass activation gradients (the base model under LoRA); the attention score and value products
  have no weights and need gradients for both operands, so they stay 2 x; a frozen vision encoder gets no
  backward;
- LoRA adapters are counted: forward 2 r (d_in + d_out) per token per adapted layer, backward 2 x that;
- activation / gradient checkpointing adds one more forward of the checkpointed language model;
- generation = one prefill of the prompt per request, shared by its n samples (vLLM's prefix caching), plus
  a KV-cached decode per sample;
- AdamW ~ 12 FLOPs per trainable parameter per update.

The VLM is counted from its cost profile (flop_profile.py, the same json format as VES-RFT's):
    lm(T, L) = a*T + b*T^2 + h*L     language model over T tokens with logits at L positions
    vis(n)   = c*n + e*n^2           vision encoder + projector for an image that becomes n LM tokens
Every other network (SENTINEL's detectors and parsers) is measured with torch's FlopCounterMode around each
call; those numbers are what ran, padding included, and miss ops torch has no FLOP formula for.
"""
import functools
import json
import os
import time

from torch.utils.flop_counter import FlopCounterMode

ADAMW_FLOPS_PER_PARAM = 12
CONVENTION = __doc__.split("\n\n")[1]


def load_profile(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"FLOP profile {path!r} is missing: run flop_profile.py --model <model> first")
    with open(path) as f:
        return json.load(f)


def lm(p, tokens, logits):
    return p["a"] * tokens + p["b"] * tokens ** 2 + p["h"] * logits


def lm_body(p, tokens):
    return p["a"] * tokens + p["b"] * tokens ** 2


def vis(p, n_image_tokens):
    if n_image_tokens <= 0:
        return 0.0
    return p["c"] * n_image_tokens + p["e"] * n_image_tokens ** 2


def prefill(p, prompt, n_image):
    """The prompt's forward (dense attention), with one logit: the first sampled token."""
    return p["a"] * prompt + p["b"] * prompt ** 2 + p["h"] + vis(p, n_image)


def decode(p, prompt, completion):
    """completion - 1 cached decode steps after the prefill; the step with K cached keys costs a + b*K + h.
    The last sampled token is never fed back."""
    steps = max(completion, 1) - 1
    keys = steps * (2 * prompt + steps + 1) / 2   # sum of K = P+1 .. P+C-1
    return p["a"] * steps + p["b"] * keys + p["h"] * steps


def teacher_forced(p, tokens, logits, n_image):
    return lm(p, tokens, logits) + vis(p, n_image)


def image_token_id(config):
    for key in ("image_token_id", "image_token_index"):
        value = getattr(config, key, None)
        if value is not None:
            return value
    raise ValueError("the model config names no image token id")


def lora_flops_per_token(model):
    """Forward FLOPs per token of every active LoRA adapter (PEFT LoraLayer: lora_A then lora_B)."""
    total = 0
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "active_adapters"):
            for name in module.active_adapters:
                if name in module.lora_A:
                    a, b = module.lora_A[name], module.lora_B[name]
                    total += 2 * (a.in_features * a.out_features + b.in_features * b.out_features)
    return total


class FlopLedger:
    """Cumulative FLOPs by component in a json file; a resumed run loads it and keeps counting.

    by_component sums to `total`. `extra` holds alternative counts that are not part of the total
    (e.g. generation if the n samples of a request did not share their prefill)."""

    def __init__(self, path, meta=None, resume=True):
        self.path, self.meta = path, meta or {}
        self.by_component, self.counters, self.extra, self.sessions = {}, {}, {}, []
        if resume and os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            self.by_component, self.counters = saved["by_component"], saved["counters"]
            self.extra, self.sessions = saved.get("extra", {}), saved.get("sessions", [])
        self.sessions.append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "total_at_start": self.total()})

    def add(self, component, flops):
        self.by_component[component] = self.by_component.get(component, 0.0) + float(flops)

    def add_extra(self, key, flops):
        self.extra[key] = self.extra.get(key, 0.0) + float(flops)

    def count(self, key, n=1):
        self.counters[key] = self.counters.get(key, 0) + n

    def wrap(self, component, fn):
        """fn, with the FLOPs of every call measured by FlopCounterMode and added to `component`."""
        @functools.wraps(fn)
        def counted(*args, **kwargs):
            out, flops = measured_flops(fn, *args, **kwargs)
            self.add(component, flops)
            self.count(f"calls/{component}")
            self.write()
            return out
        return counted

    def total(self):
        return sum(self.by_component.values())

    def summary(self):
        return {"total": self.total(), "by_component": self.by_component, "counters": self.counters,
                "extra": self.extra, "sessions": self.sessions, "convention": CONVENTION, **self.meta}

    def write(self, path=None):
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(self.summary(), f, indent=1)
        os.replace(path + ".tmp", path)


def profile_meta(p):
    return {"profile": {k: p.get(k) for k in ("model", "a", "b", "h", "c", "e", "max_abs_rel_error")}}


def measured_flops(fn, *args, **kwargs):
    """(result, FLOPs) of one call under FlopCounterMode."""
    with FlopCounterMode(display=False) as counter:
        out = fn(*args, **kwargs)
    return out, float(counter.get_total_flops())

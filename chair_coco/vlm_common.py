"""Model-agnostic VLM helpers (Qwen2.5-VL, llava-hf, Gemma 3): image loading, chat prompts, batching, log-probs.

The prompt ("Describe this image in detail." as the only user turn) matches the probe eval, so
training and editing see the inputs the eval feeds the model.
"""
import json

import torch
from PIL import Image

# Every language-model linear layer except lm_head; the vision tower and projector stay frozen
# (LLaMA-Factory's default freeze_vision_tower / freeze_multi_modal_projector), and vLLM can serve the adapter.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_image(path, max_pixels):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if max_pixels and w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return image


def user_messages(question):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]


def prompt_text(processor, question):
    return processor.apply_chat_template(user_messages(question), tokenize=False, add_generation_prompt=True)


def check_lora_targets(model, pattern=LORA_TARGETS):
    import re
    matches = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    assert matches, f"LoRA pattern {pattern!r} matched no modules"
    assert not [n for n in matches if "visual" in n or "vision" in n], "LoRA pattern hits the vision tower"
    return len(matches)


def build_batch(items, pad_id):
    """items: [(prompt_enc, response_ids)], prompt_enc = processor(text=[prompt], images=[img], return_tensors="pt").

    Right-pads prompt + response into one batch. Per-token tensors (input_ids, attention_mask, token_type_ids,
    mm_token_type_ids) are extended over the response; image tensors are concatenated. Returns the batch and,
    per row, (prompt_len, total_len).
    """
    seq_keys = [k for k, v in items[0][0].items()
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == items[0][0]["input_ids"].shape[1]]
    rows, spans = [], []
    for enc, resp in items:
        n = enc["input_ids"].shape[1]
        row = {}
        for k in seq_keys:
            fill = {"input_ids": resp, "attention_mask": [1] * len(resp)}.get(k, [0] * len(resp))
            row[k] = enc[k][0].tolist() + list(fill)
        rows.append(row)
        spans.append((n, n + len(resp)))
    width = max(len(r["input_ids"]) for r in rows)
    batch = {}
    for k in seq_keys:
        pad = pad_id if k == "input_ids" else 0
        batch[k] = torch.tensor([r[k] + [pad] * (width - len(r[k])) for r in rows], dtype=torch.long)
    for k, v in items[0][0].items():
        if k not in seq_keys and torch.is_tensor(v):
            batch[k] = torch.cat([enc[k] for enc, _ in items])
    return batch, spans


def response_logps(model, batch, spans, label_starts):
    """Sum of log p(token) over each row's label tokens (positions label_start..end of the response)."""
    logits = model(**batch, use_cache=False).logits
    out = []
    for i, ((a, b), s) in enumerate(zip(spans, label_starts)):
        lo = a + s
        lg = logits[i, lo - 1 : b - 1].float()
        labels = batch["input_ids"][i, lo:b]
        out.append((lg.gather(-1, labels[:, None]).squeeze(-1) - torch.logsumexp(lg, dim=-1)).sum())
    return torch.stack(out)

"""The truncation null: is the RL arm's gain brevity, or better use of the same budget?

Shorter captions mention fewer objects and hallucinate less for free, so an arm that writes
less will beat base on findings/caption without being any better grounded. The principled
control -- section 3 of docs/rl_checkpoint_evidence.md, which ck250 passed -- is to cut each
BASE caption to its paired RL caption's token count and compare at matched length.

Two truncation variants, as in that section:

  token     cut at exactly the paired RL caption's token count, mid-sentence if that is where
            it lands. Matches length exactly; produces text no model would emit.
  sentence  cut at the last sentence boundary at or before that count. Reads naturally and is
            the fairer comparison for a judge, but lands short of the target, which biases
            against BASE -- fewer tokens, fewer chances to err.

Report both. They bracket the answer: if the RL arm beats BASE-token-truncated it is not merely
shorter, and if it also beats BASE-sentence-truncated -- which is shorter still -- the result is
robust to the truncation artefact.

The object metrics here are judge-free. Mentions come from the caption text via the COCO
synonym table compare_greedy_arms.py already uses, ground truth from the manifest:

    correct_i    GT categories the caption mentions
    halluc_i     COCO categories mentioned that are NOT in the image's annotations
    recall       sum(correct) / sum(GT)          pooled, CHAIR convention
    precision    sum(correct) / sum(mentioned)   pooled

Pooled rather than per-image so a caption that mentions nothing does not force a 0/0. The
bootstrap resamples images and recomputes the pooled ratio inside each draw, so the CI carries
the same convention as the point estimate.

--emit-rollouts writes the truncated captions as a rollouts-style directory that
judge_openai.py can read, for the judged half of the control:

    python src/eval/truncation_null.py ... --emit-rollouts runs/<run>/eval/base_trunc_token
    python src/extract/judge_openai.py --rundir runs/<run>/eval/base_trunc_token --outdir ...
    python src/eval/compare_greedy_arms.py --pre-rollouts runs/<run>/eval/base_trunc_token ...

Re-encoding is deliberate: cutting text at a character offset can retokenise differently, so
each truncated caption is re-tokenised and its token_ids/spans rebuilt from that. Otherwise
compare_greedy_arms.py's `corrupted` guard drops the row and the arms silently stop being paired.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_greedy_arms import COCO_CATEGORIES, load_rollouts, mentions  # noqa: E402

SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def truncate(rollout, n_tokens, boundary):
    """Cut a caption to n_tokens via its stored token_char_spans, optionally back to a stop."""
    spans = rollout["token_char_spans"]
    if n_tokens >= len(spans):
        return rollout["text"]
    text = rollout["text"][:spans[n_tokens - 1][1]]
    if boundary == "sentence":
        stops = [m.end() for m in SENTENCE_END.finditer(text)]
        if stops:
            text = text[:stops[-1]]
    return text


def retokenise(text, tokenizer):
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return {"text": text, "token_ids": encoded["input_ids"],
            "token_char_spans": [list(s) for s in encoded["offset_mapping"]],
            "n_tokens": len(encoded["input_ids"]), "finished": True}


def object_counts(texts, manifest, image_ids):
    correct, mentioned, n_gt = [], [], []
    for text, image_id in zip(texts, image_ids):
        lowered = text.lower()
        truth = manifest[image_id]["objects"]
        correct.append(sum(mentions(lowered, o) for o in truth))
        mentioned.append(sum(mentions(lowered, c) for c in COCO_CATEGORIES))
        n_gt.append(len(truth))
    return (np.array(correct, dtype=float), np.array(mentioned, dtype=float),
            np.array(n_gt, dtype=float))


def summarise(tokens, correct, mentioned, n_gt):
    return {"mean_tokens": float(np.mean(tokens)),
            "recall": float(correct.sum() / n_gt.sum()),
            "precision": float(correct.sum() / max(mentioned.sum(), 1)),
            "halluc_per_caption": float((mentioned - correct).mean()),
            "mentioned_per_caption": float(mentioned.mean())}


def paired(rl, ctrl, rng, samples):
    """RL minus control, resampling images and recomputing pooled ratios inside each draw."""
    n = len(rl["correct"])
    out = {}
    stats = {
        "recall_pp": lambda k: 100 * (rl["correct"][k].sum() / rl["n_gt"][k].sum()
                                      - ctrl["correct"][k].sum() / ctrl["n_gt"][k].sum()),
        "precision_pp": lambda k: 100 * (rl["correct"][k].sum() / max(rl["mentioned"][k].sum(), 1)
                                         - ctrl["correct"][k].sum()
                                         / max(ctrl["mentioned"][k].sum(), 1)),
        "halluc_per_caption": lambda k: ((rl["mentioned"][k] - rl["correct"][k]).mean()
                                         - (ctrl["mentioned"][k] - ctrl["correct"][k]).mean()),
        "tokens": lambda k: rl["tokens"][k].mean() - ctrl["tokens"][k].mean(),
    }
    full = np.arange(n)
    for name, fn in stats.items():
        draws = np.array([fn(rng.integers(0, n, n)) for _ in range(samples)])
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        out[name] = {"mean": float(fn(full)), "ci": [lo, hi],
                     "significant": bool(hi < 0 or lo > 0)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-rollouts", required=True)
    p.add_argument("--rl-rollouts", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--rollout", type=int, default=0)
    p.add_argument("--label", default="arm")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emit-rollouts", help="write the truncated BASE captions here for judging")
    p.add_argument("--emit-variant", choices=["token", "sentence"], default="token")
    p.add_argument("--json")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    manifest = {d["image_id"]: d for d in json.load(open(a.manifest))}
    base = load_rollouts(a.base_rollouts, a.rollout)
    rl = load_rollouts(a.rl_rollouts, a.rollout)
    ids = sorted(set(base) & set(rl) & set(manifest))
    print(f"\n{a.label}   paired images: {len(ids)}")

    arms = {"BASE (full)": [base[i]["text"] for i in ids],
            "BASE trunc (token)": [truncate(base[i], rl[i]["n_tokens"], "token") for i in ids],
            "BASE trunc (sentence)": [truncate(base[i], rl[i]["n_tokens"], "sentence")
                                      for i in ids],
            f"RL ({a.label})": [rl[i]["text"] for i in ids]}
    # token counts for the truncated arms come from re-encoding, not from the cut point
    tokens = {}
    for name, texts in arms.items():
        if name.startswith("BASE trunc"):
            tokens[name] = np.array([len(tokenizer(t, add_special_tokens=False)["input_ids"])
                                     for t in texts], dtype=float)
        elif name.startswith("BASE"):
            tokens[name] = np.array([base[i]["n_tokens"] for i in ids], dtype=float)
        else:
            tokens[name] = np.array([rl[i]["n_tokens"] for i in ids], dtype=float)

    stats, summary = {}, {}
    for name, texts in arms.items():
        correct, mentioned, n_gt = object_counts(texts, manifest, ids)
        stats[name] = {"correct": correct, "mentioned": mentioned, "n_gt": n_gt,
                       "tokens": tokens[name]}
        summary[name] = summarise(tokens[name], correct, mentioned, n_gt)

    width = max(len(n) for n in arms) + 2
    print(f"\n{'arm':<{width}}{'tokens':>9}{'recall':>9}{'precision':>11}{'halluc/cap':>12}")
    for name in arms:
        s = summary[name]
        print(f"{name:<{width}}{s['mean_tokens']:>9.1f}{s['recall']:>8.1%}"
              f"{s['precision']:>11.1%}{s['halluc_per_caption']:>12.3f}")

    rng = np.random.default_rng(a.seed)
    deltas = {}
    rl_key = f"RL ({a.label})"
    for control in ("BASE trunc (token)", "BASE trunc (sentence)", "BASE (full)"):
        print(f"\nRL minus {control}  (paired bootstrap, n={len(ids)})")
        deltas[control] = paired(stats[rl_key], stats[control], rng, a.bootstrap)
        for metric, row in deltas[control].items():
            flag = "SIGNIFICANT" if row["significant"] else "n.s."
            print(f"   {metric:<20} {row['mean']:+8.3f}  "
                  f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]  {flag}")

    if a.emit_rollouts:
        name = f"BASE trunc ({a.emit_variant})"
        os.makedirs(a.emit_rollouts, exist_ok=True)
        for image_id, text in zip(ids, arms[name]):
            row = retokenise(text, tokenizer)
            row["rollout"] = a.rollout
            row["offset"] = 0
            json.dump({"image_id": image_id, "path": manifest[image_id]["path"],
                       "n_rollouts": 1, "n_tokens": row["n_tokens"], "rollouts": [row]},
                      open(os.path.join(a.emit_rollouts, f"{image_id}.json"), "w"))
        json.dump({"source": a.base_rollouts, "truncated_to": a.rl_rollouts,
                   "variant": a.emit_variant, "tokenizer": a.tokenizer,
                   "note": "BASE captions cut to the paired RL caption's token count; "
                           "re-encoded so text and token_ids agree"},
                  open(os.path.join(a.emit_rollouts, "meta.json"), "w"), indent=1)
        print(f"\nwrote {len(ids)} truncated captions -> {a.emit_rollouts}")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"label": a.label, "n_images": len(ids), "seed": a.seed,
                   "bootstrap": a.bootstrap, "arms": summary, "deltas": deltas},
                  open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()

"""Paired base-vs-GRPO report for two greedy rollout runs judged by judge_openai.py.

Reads the raw judge output directly (no judge_labels.py normalisation needed) and writes
one JSON report plus a printed table. Every number in the report is reproducible from the
four input directories.

    python src/eval/compare_greedy_arms.py \\
        --pre-rollouts  /scratch/sathwik/probe_new_runs/comparision/pre_rl_greedy \\
        --pre-judged    /scratch/sathwik/probe_new_runs/judge_data/pre_rl_greedy_judged \\
        --post-rollouts /scratch/sathwik/probe_new_runs/comparision/post_rl_greedy \\
        --post-judged   /scratch/sathwik/probe_new_runs/judge_data/post_rl_greedy_judged \\
        --manifest new_data/coco_v2/rl_test.json \\
        --out /scratch/sathwik/probe_new_runs/comparision/greedy_arms_report.json

Two provenance rules the report makes explicit, because they decide how much the numbers weigh:
  * findings/clean/per-100-token/CHAIR-numerator come from the LLM judge, nothing else;
  * coverage (COCO object recall, categories mentioned) comes from the manifest annotations
    and is judge-independent.
An image whose stored `text` does not re-encode to its stored `token_ids` is DROPPED: the
judge read `text`, so a corrupted row was judged against different input than it records.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# A caption rarely uses the COCO lemma. These are applied identically to both arms, so the
# DELTA is meaningful even where the absolute rate is only a proxy.
SYNONYMS = {
    "person": ["person", "people", "man", "woman", "men", "women", "boy", "girl", "child",
               "children", "guy", "lady", "pedestrian", "player"],
    "tv": ["tv", "television", "monitor"],
    "couch": ["couch", "sofa"],
    "motorcycle": ["motorcycle", "motorbike"],
    "bicycle": ["bicycle", "bike", "cyclist"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "dining table": ["dining table", "table"],
    "cell phone": ["cell phone", "phone", "smartphone"],
    "potted plant": ["potted plant", "houseplant"],
    "sports ball": ["ball"],
    "hot dog": ["hot dog", "hotdog"],
    "traffic light": ["traffic light", "stoplight"],
}

HEDGES = ["appears to", "appear to", "seems to", "seem to", "likely", "possibly", "perhaps",
          "may be", "might be", "suggesting", "suggests", "not clearly visible",
          "difficult to", "cannot be", "unclear", "presumably", "appears"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre-rollouts", required=True)
    p.add_argument("--pre-judged", required=True)
    p.add_argument("--post-rollouts", required=True)
    p.add_argument("--post-judged", required=True)
    p.add_argument("--manifest", default="new_data/coco_v2/rl_test.json")
    p.add_argument("--rollout", type=int, default=0, help="which rollout id to compare")
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-VL-3B-Instruct")
    return p.parse_args()


def image_files(directory):
    skip = {"meta.json", "judge_labels.json", "judge_free.json"}
    return sorted(
        f
        for f in glob.glob(os.path.join(directory, "*.json"))
        if os.path.basename(f) not in skip
    )


def load_rollouts(directory, rollout_id):
    rows = {}
    for path in image_files(directory):
        doc = json.load(open(path))
        match = [r for r in doc["rollouts"] if r["rollout"] == rollout_id]
        if len(match) != 1:
            raise ValueError(f"expected one rollout {rollout_id} in {path}, got {len(match)}")
        rows[int(doc["image_id"])] = match[0]
    return rows


def parse_findings(raw):
    """judge_openai.py stores the model's raw reply; recover the hallucination list."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    decoded = json.loads(raw[start:end + 1]) if _decodable(raw[start:end + 1]) else None
    if not isinstance(decoded, dict) or not isinstance(decoded.get("hallucinations"), list):
        return None
    return decoded["hallucinations"]


def _decodable(text):
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def load_judged(directory, rollout_id):
    rows = {}
    for path in image_files(directory):
        doc = json.load(open(path))
        match = [r for r in doc["rollouts"] if r["rollout"] == rollout_id]
        if len(match) != 1:
            raise ValueError(f"expected one judged rollout in {path}, got {len(match)}")
        findings = parse_findings(match[0]["raw"])
        if findings is not None:
            rows[int(doc["image_id"])] = findings
    return rows


def mentions(text, category):
    for word in SYNONYMS.get(category, [category]):
        if re.search(r"\b" + re.escape(word) + r"e?s?\b", text):
            return True
    return False


def corrupted(rollout, tokenizer):
    """The judge reads `text`; if `text` no longer matches `token_ids` the row is unusable."""
    return tokenizer(rollout["text"], add_special_tokens=False)["input_ids"] != rollout["token_ids"]


def arm_metrics(rollouts, judged, manifest, image_ids):
    findings = np.array([len(judged[i]) for i in image_ids], dtype=float)
    tokens = np.array([rollouts[i]["n_tokens"] for i in image_ids], dtype=float)
    object_findings = np.array(
        [sum(f.get("type") == "object" for f in judged[i]) for i in image_ids], dtype=float)
    mentioned = np.array(
        [sum(mentions(rollouts[i]["text"].lower(), c) for c in COCO_CATEGORIES)
         for i in image_ids], dtype=float)
    recall = np.array(
        [sum(mentions(rollouts[i]["text"].lower(), o) for o in manifest[i]["objects"])
         / len(manifest[i]["objects"]) for i in image_ids], dtype=float)
    words = np.array([len(rollouts[i]["text"].split()) for i in image_ids], dtype=float)
    hedges = np.array([sum(rollouts[i]["text"].lower().count(h) for h in HEDGES)
                       for i in image_ids], dtype=float)
    clipped = np.array([not rollouts[i]["finished"] for i in image_ids], dtype=float)

    summary = {
        "judge_derived": {
            "mean_findings_per_caption": findings.mean(),
            "clean_caption_rate": float((findings == 0).mean()),
            "findings_per_100_tokens": float((100 * findings / tokens).mean()),
            "mean_object_findings_per_caption": object_findings.mean(),
            "finding_types": dict(Counter(
                f.get("type", "?") for i in image_ids for f in judged[i])),
            "finding_confidence": dict(Counter(
                f.get("confidence", "?") for i in image_ids for f in judged[i])),
            "findings_histogram": {str(k): int(v) for k, v in sorted(
                Counter(int(x) for x in findings).items())},
        },
        "judge_independent": {
            "mean_tokens": tokens.mean(),
            "median_tokens": float(np.median(tokens)),
            "clipped_rate": float(clipped.mean()),
            "coco_categories_mentioned_per_caption": mentioned.mean(),
            "gt_object_recall": recall.mean(),
            "hedges_per_100_words": float(100 * hedges.sum() / words.sum()),
            "hedges_per_category_mentioned": float(hedges.sum() / mentioned.sum()),
        },
        "mixed_numerator_judge_denominator_annotation": {
            "chair_style_object_findings_per_category_mentioned":
                float(object_findings.sum() / mentioned.sum()),
            "all_findings_per_category_mentioned": float(findings.sum() / mentioned.sum()),
        },
    }
    vectors = {"findings": findings, "tokens": tokens, "per_100_tokens": 100 * findings / tokens,
               "mentioned": mentioned, "recall": recall}
    return summary, vectors


def confidence_filters(judged, image_ids):
    """Is the judge-derived gap driven by its own marginal flags?"""
    tests = {
        "all_findings": lambda f: True,
        "confidence_high_only": lambda f: f.get("confidence") == "high",
        "type_object_only": lambda f: f.get("type") == "object",
        "type_object_and_confidence_high":
            lambda f: f.get("type") == "object" and f.get("confidence") == "high",
    }
    out = {}
    for name, keep in tests.items():
        counts = np.array([sum(keep(f) for f in judged[i]) for i in image_ids], dtype=float)
        out[name] = {"mean_findings": counts.mean(),
                     "clean_rate": float((counts == 0).mean())}
    return out


def bootstrap_ci(values, rng, samples):
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    manifest = {d["image_id"]: d for d in json.load(open(args.manifest))}
    pre_r = load_rollouts(args.pre_rollouts, args.rollout)
    post_r = load_rollouts(args.post_rollouts, args.rollout)
    pre_j = load_judged(args.pre_judged, args.rollout)
    post_j = load_judged(args.post_judged, args.rollout)

    shared = set(pre_r) & set(post_r) & set(pre_j) & set(post_j) & set(manifest)
    dropped = {}
    for image_id in sorted(shared):
        for arm, rows in (("pre", pre_r), ("post", post_r)):
            if corrupted(rows[image_id], tokenizer):
                dropped[image_id] = f"{arm}: stored text does not re-encode to token_ids"
    image_ids = sorted(shared - set(dropped))
    print(f"paired images: {len(image_ids)}   dropped: {dropped or 'none'}")

    pre_summary, pre_v = arm_metrics(pre_r, pre_j, manifest, image_ids)
    post_summary, post_v = arm_metrics(post_r, post_j, manifest, image_ids)
    rng = np.random.default_rng(args.seed)

    paired = {}
    for name, delta in (
        ("findings_per_caption", post_v["findings"] - pre_v["findings"]),
        ("findings_per_100_tokens", post_v["per_100_tokens"] - pre_v["per_100_tokens"]),
        ("coco_categories_mentioned", post_v["mentioned"] - pre_v["mentioned"]),
        ("gt_object_recall_pp", 100 * (post_v["recall"] - pre_v["recall"])),
    ):
        low, high = bootstrap_ci(delta, rng, args.bootstrap)
        paired[name] = {"mean_delta": float(delta.mean()), "bootstrap_95_ci": [low, high],
                        "significant": (low > 0) == (high > 0)}
    delta = post_v["findings"] - pre_v["findings"]
    paired["per_image_outcomes"] = {"rl_fewer_findings": int((delta < 0).sum()),
                                    "tie": int((delta == 0).sum()),
                                    "rl_more_findings": int((delta > 0).sum())}

    report = {
        "n_paired_images": len(image_ids),
        "dropped_images": dropped,
        "inputs": {"pre_rollouts": args.pre_rollouts, "pre_judged": args.pre_judged,
                   "post_rollouts": args.post_rollouts, "post_judged": args.post_judged,
                   "manifest": args.manifest, "rollout_id": args.rollout},
        "judge": json.load(open(os.path.join(args.post_judged, "meta.json")))["judge_model"],
        "pre": pre_summary,
        "post": post_summary,
        "paired_deltas": paired,
        "judge_robustness": {"pre": confidence_filters(pre_j, image_ids),
                             "post": confidence_filters(post_j, image_ids)},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, default=float)

    def row(label, pre_value, post_value, fmt="{:.3f}"):
        print(f"  {label:<44} {fmt.format(pre_value):>10} {fmt.format(post_value):>12}")

    print(f"\n{'':<46}{'BASE':>10} {'POST-RL':>12}")
    print("  -- from the LLM judge (" + report["judge"] + ") " + "-" * 20)
    for key, fmt in (("mean_findings_per_caption", "{:.3f}"),
                     ("clean_caption_rate", "{:.1%}"),
                     ("findings_per_100_tokens", "{:.3f}"),
                     ("mean_object_findings_per_caption", "{:.3f}")):
        row(key, pre_summary["judge_derived"][key], post_summary["judge_derived"][key], fmt)
    print("  -- judge-independent (COCO annotations / caption text) " + "-" * 6)
    for key, fmt in (("mean_tokens", "{:.1f}"), ("clipped_rate", "{:.1%}"),
                     ("coco_categories_mentioned_per_caption", "{:.2f}"),
                     ("gt_object_recall", "{:.1%}"),
                     ("hedges_per_category_mentioned", "{:.2f}")):
        row(key, pre_summary["judge_independent"][key],
            post_summary["judge_independent"][key], fmt)
    print("  -- per-claim (judge numerator / annotation denominator) " + "-" * 5)
    for key in ("chair_style_object_findings_per_category_mentioned",
                "all_findings_per_category_mentioned"):
        row(key, pre_summary["mixed_numerator_judge_denominator_annotation"][key],
            post_summary["mixed_numerator_judge_denominator_annotation"][key], "{:.4f}")

    print("\n  paired deltas (post - base), bootstrap 95% CI:")
    for name, value in paired.items():
        if name == "per_image_outcomes":
            continue
        low, high = value["bootstrap_95_ci"]
        print(f"    {name:<28} {value['mean_delta']:+8.3f}  [{low:+.3f}, {high:+.3f}]  "
              f"{'significant' if value['significant'] else 'not significant'}")
    print(f"    per-image outcomes: {paired['per_image_outcomes']}")

    print("\n  judge robustness (mean findings / clean rate):")
    for name in report["judge_robustness"]["pre"]:
        pre_f = report["judge_robustness"]["pre"][name]
        post_f = report["judge_robustness"]["post"][name]
        print(f"    {name:<34} {pre_f['mean_findings']:.3f} / {pre_f['clean_rate']:.1%}"
              f"   ->   {post_f['mean_findings']:.3f} / {post_f['clean_rate']:.1%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

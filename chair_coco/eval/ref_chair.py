"""Reference CHAIR (CHAIR_S, CHAIR_I, recall) of one arm against the base model, as the paper's Table 1 computes it.

A generic copy of the probe repo's store/runs/v2_qwen25vl3b/analysis/run_ref_chair.py (the script behind the
paper's CHAIR columns): per_image(), stats() and delta() are unchanged; only its hard-coded paths and arm list are
command-line arguments. CHAIR comes from reference_chair_opera.py (OPERA's copy of Rohrbach et al.'s chair.py), with
the annotations restricted to val2014 (all test images are val2014 files) as in the original. Two ground truths:
  ref = COCO segmentation labels + objects named in the 5 human COCO captions (the standard CHAIR; Table 1 uses this)
  seg = segmentation labels only (what the synonym matcher uses)
CHAIR_S = share of captions with a hallucinated object; CHAIR_I = hallucinated object mentions / all object mentions,
repeated mentions counted; recall = ground-truth objects mentioned / ground-truth objects, pooled over images.

    python ref_chair.py --chair reference_chair_opera.py --ann <dir with instances_val2014.json, captions_val2014.json> \\
        --manifest rl_test.json --base <base captions> --arm <name>=<arm captions>,<matched base captions> --out ref_chair.json
"""
import argparse
import collections
import copy
import glob
import importlib.util
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chair", required=True, help="reference_chair_opera.py")
    p.add_argument("--ann", required=True, help="COCO annotations dir (instances_val2014.json, captions_val2014.json)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--base", required=True, help="base model's greedy caption dir (rollout_acts.py format)")
    p.add_argument("--arm", action="append", default=[],
                   help="NAME=ARM_DIR[,MATCHED_BASE_DIR]; the matched base is truncation_null.py's base_trunc_token")
    p.add_argument("--workdir", required=True, help="scratch dir for the caption files compute_chair reads")
    p.add_argument("--out", required=True)
    return p.parse_args()


args = parse_args()
spec = importlib.util.spec_from_file_location("ref_chair", args.chair)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
# All test images are val2014 files; the reference only differs by also loading train2014.
ref.combine_coco_instances = lambda p: json.load(open(f"{args.ann}/instances_val2014.json"))
ref.combine_coco_captions = lambda p: json.load(open(f"{args.ann}/captions_val2014.json"))

man = {m["image_id"]: m for m in json.load(open(args.manifest))}
ids = sorted(man)
OUT = args.workdir
os.makedirs(OUT, exist_ok=True)

ev_ref = ref.CHAIR(args.ann)
ev_seg = copy.copy(ev_ref)
ev_seg.imid_to_objects = {i: set(man[i]["objects"]) for i in ids}
# 'seg' must equal the segmentation part of 'ref'
for i in ids:
    assert set(man[i]["objects"]) <= ev_ref.imid_to_objects[i], i


def captions(d):
    out = {}
    for f in glob.glob(f"{d}/*.json"):
        if os.path.basename(f) in {"meta.json", "judge_labels.json", "judge_free.json"}:
            continue
        doc = json.load(open(f))
        out[int(doc["image_id"])] = [r for r in doc["rollouts"] if r["rollout"] == 0][0]["text"]
    assert set(out) == set(ids), d
    return out


def per_image(ev, caps, tag):
    path = f"{OUT}/caps_{tag}.jsonl"
    with open(path, "w") as f:
        for i in ids:
            f.write(json.dumps({"image_id": i, "caption": caps[i]}) + "\n")
    res = ev.compute_chair(path, "image_id", "caption")
    rows = {s["image_id"]: s for s in res["sentences"]}
    a = {"hall": [], "ment": [], "hit": [], "ngt": []}
    halluc_words = collections.Counter()
    for i in ids:
        s = rows[i]
        a["hall"].append(len(s["mscoco_hallucinated_words"]))
        a["ment"].append(len(s["mscoco_generated_words"]))
        a["hit"].append(len(set(s["mscoco_generated_words"]) & set(s["mscoco_gt_words"])))
        a["ngt"].append(len(s["mscoco_gt_words"]))
        halluc_words.update(n for _, n in s["mscoco_hallucinated_words"])
    return {k: np.array(v, float) for k, v in a.items()}, halluc_words


def stats(a, idx=None):
    if idx is not None:
        a = {k: v[idx] for k, v in a.items()}
    return {"CHAIR_s": float((a["hall"] > 0).mean()), "CHAIR_i": float(a["hall"].sum() / a["ment"].sum()),
            "recall": float(a["hit"].sum() / a["ngt"].sum())}


def delta(base, rl, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    point = {k: stats(rl)[k] - stats(base)[k] for k in ("CHAIR_s", "CHAIR_i", "recall")}
    draws = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(ids), len(ids))
        b, r = stats(base, idx), stats(rl, idx)
        for k in point:
            draws[k].append(r[k] - b[k])
    return {k: [point[k], *np.percentile(draws[k], [2.5, 97.5]).tolist()] for k in point}


ARMS = {}
for spec_ in args.arm:
    name, dirs = spec_.split("=", 1)
    arm_dir, _, matched = dirs.partition(",")
    ARMS[name] = (arm_dir, matched or None)

result = {}
for gt_name, ev in (("ref", ev_ref), ("seg", ev_seg)):
    base, base_words = per_image(ev, captions(args.base), f"{gt_name}_base")
    r = {"base_full": stats(base), "base_top_hallucinated": base_words.most_common(15), "arms": {}}
    for name, (arm_dir, matched) in ARMS.items():
        rl, _ = per_image(ev, captions(arm_dir), f"{gt_name}_{name}")
        entry = {"rl": stats(rl), "delta_vs_full": delta(base, rl)}
        if matched:
            bm, _ = per_image(ev, captions(matched), f"{gt_name}_{name}_trunc")
            entry.update(base_matched=stats(bm), delta_matched=delta(bm, rl))
        r["arms"][name] = entry
    result[gt_name] = r
json.dump(result, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
pct = lambda x: f"{100 * x:5.1f}%"
print("                      CHAIR_s  CHAIR_i  recall   (ref ground truth)")
print(f"{'base':20s}  " + "  ".join(pct(result['ref']['base_full'][k]) for k in ("CHAIR_s", "CHAIR_i", "recall")))
for name in ARMS:
    print(f"{name:20s}  " + "  ".join(pct(result['ref']['arms'][name]['rl'][k]) for k in ("CHAIR_s", "CHAIR_i", "recall")))

"""Table 1 of the paper (object hallucination on the COCO RL-test split) for one method: the base-model row and the
method's row, mean +- standard deviation (sample, n-1) over seeds, from the files eval_probe.sh writes per seed:

    column              source file                    value
    CHAIR_S, CHAIR_I    ref_chair.json                 ref ground truth (segmentation + human captions), arm "rl"
    Recall              ref_chair.json                 same
    Wrong obj./cap.     truncation_null.json           synonym matcher, segmentation labels: halluc_per_caption
    Precision           truncation_null.json           correct / mentioned, pooled over images
    Tokens              truncation_null.json           mean caption tokens
    Judged err./cap.    judged_compare_full.json       gpt-5.1 (judge_v3) findings per caption
    Clean               judged_compare_full.json       share of captions with no judged finding

Checked against the stored v2 Qwen2.5-VL-3B files: reproduces the paper's base row and its method row exactly.
Also written: each seed's row and its paired differences vs base with the 95% bootstrap intervals the scripts store,
an exact two-sided McNemar test of the clean-caption rate per seed, and the matched-length results where present.

    python paper_table.py --name SENTINEL <eval dir seed 42> <eval dir seed 43> <eval dir seed 44> [--out table.json]
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_greedy_arms import load_judged  # noqa: E402

COLUMNS = [  # key, header, scale, decimals for the value, decimals for the sd (the paper's)
    ("chair_s", "CHAIR_S ↓", 100, 1, 1), ("chair_i", "CHAIR_I ↓", 100, 1, 1),
    ("wrong_obj", "Wrong obj./cap. ↓", 1, 3, 3), ("judged", "Judged err./cap. ↓", 1, 2, 2),
    ("clean", "Clean ↑", 100, 1, 1), ("recall", "Recall ↑", 100, 1, 1),
    ("precision", "Precision ↑", 100, 1, 1), ("tokens", "Tokens", 1, 1, 1),
]


def load(path):
    return json.load(open(path))


def one_arm(d, key):
    """The single arm entry of a ref_chair.json / RL entry of truncation_null.json."""
    entries = [v for k, v in d.items() if key(k)]
    assert len(entries) == 1, f"expected one arm, got {len(entries)}"
    return entries[0]


def seed_values(run):
    ref = load(os.path.join(run, "ref_chair.json"))["ref"]
    arm = one_arm(ref["arms"], lambda k: True)
    tn = load(os.path.join(run, "truncation_null.json"))
    rl = one_arm(tn["arms"], lambda k: k.startswith("RL"))
    base_tn = tn["arms"]["BASE (full)"]
    rep = load(os.path.join(run, "judged_compare_full.json"))
    row = lambda c, t, j: {"chair_s": c["CHAIR_s"], "chair_i": c["CHAIR_i"], "recall": c["recall"],
                           "wrong_obj": t["halluc_per_caption"], "precision": t["precision"], "tokens": t["mean_tokens"],
                           "judged": j["mean_findings_per_caption"], "clean": j["clean_caption_rate"]}
    post = row(arm["rl"], rl, rep["post"]["judge_derived"])
    base = row(ref["base_full"], base_tn, rep["pre"]["judge_derived"])
    deltas = {"CHAIR_S_pp": [100 * x for x in arm["delta_vs_full"]["CHAIR_s"]],
              "CHAIR_I_pp": [100 * x for x in arm["delta_vs_full"]["CHAIR_i"]],
              "recall_pp": [100 * x for x in arm["delta_vs_full"]["recall"]],
              "wrong_obj": tn["deltas"]["BASE (full)"]["halluc_per_caption"],
              "precision_pp": tn["deltas"]["BASE (full)"]["precision_pp"],
              "judged": rep["paired_deltas"]["findings_per_caption"]}
    out = {"run": run, "post": post, "base": base, "deltas_vs_base": deltas,
           "mcnemar_clean": mcnemar(rep["inputs"]["pre_judged"], rep["inputs"]["post_judged"])}
    matched = os.path.join(run, "judged_compare_matched.json")
    if os.path.exists(matched):
        m = load(matched)
        out["matched_length"] = {"judged": m["paired_deltas"]["findings_per_caption"],
                                 "clean": [m["pre"]["judge_derived"]["clean_caption_rate"],
                                           m["post"]["judge_derived"]["clean_caption_rate"]],
                                 "CHAIR_S_pp": [100 * x for x in arm.get("delta_matched", {}).get("CHAIR_s", [])],
                                 "CHAIR_I_pp": [100 * x for x in arm.get("delta_matched", {}).get("CHAIR_i", [])],
                                 "objects_and_recall": tn["deltas"].get("BASE trunc (token)")}
    return out


def mcnemar(pre_judged, post_judged):
    """Exact two-sided McNemar test of the clean-caption rate, paired over images (None if the judgments are absent)."""
    if not (os.path.isdir(pre_judged) and os.path.isdir(post_judged)):
        return None
    pre, post = load_judged(pre_judged, 0), load_judged(post_judged, 0)
    ids = sorted(set(pre) & set(post))
    became_clean = sum(len(pre[i]) > 0 and len(post[i]) == 0 for i in ids)
    became_dirty = sum(len(pre[i]) == 0 and len(post[i]) > 0 for i in ids)
    n, k = became_clean + became_dirty, min(became_clean, became_dirty)
    p = min(1.0, 2 * sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n) if n else 1.0
    return {"images": len(ids), "became_clean": became_clean, "became_dirty": became_dirty, "p_value": p}


def fmt(values, scale, dec, sd_dec):
    vals = [scale * v for v in values]
    if len(vals) == 1:
        return f"{vals[0]:.{dec}f}"
    return f"{statistics.mean(vals):.{dec}f} ± {statistics.stdev(vals):.{sd_dec}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="eval dirs of one method, one per seed")
    p.add_argument("--name", default="Method")
    p.add_argument("--out", help="also write everything as JSON")
    a = p.parse_args()
    seeds = [seed_values(r) for r in a.runs]
    for s in seeds[1:]:
        for k, v in s["base"].items():
            if abs(v - seeds[0]["base"][k]) > 1e-9:
                print(f"WARNING: the base row differs between {seeds[0]['run']} and {s['run']} ({k})")
    header = "| Method | " + " | ".join(h for _, h, *_ in COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 1),
             "| Base model | " + " | ".join(fmt([seeds[0]["base"][k]], sc, d, sd) for k, _, sc, d, sd in COLUMNS) + " |",
             f"| {a.name} | " + " | ".join(fmt([s["post"][k] for s in seeds], sc, d, sd) for k, _, sc, d, sd in COLUMNS) + " |"]
    print(f"Table 1 rows ({len(seeds)} seed{'s' * (len(seeds) > 1)}; ± = sample standard deviation over seeds)\n")
    print("\n".join(lines))
    print("\nper seed, difference vs base [95% bootstrap interval]; clean rate: exact McNemar")
    for s in seeds:
        d = s["deltas_vs_base"]
        ci = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
        tn = lambda x: f"{x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
        j = d["judged"]
        mc = s["mcnemar_clean"]
        print(f"  {s['run']}\n    CHAIR_S {ci(d['CHAIR_S_pp'])}  CHAIR_I {ci(d['CHAIR_I_pp'])}  recall {ci(d['recall_pp'])}"
              f"\n    wrong obj {tn(d['wrong_obj'])}  precision {tn(d['precision_pp'])}"
              f"  judged {j['mean_delta']:+.3f} [{j['bootstrap_95_ci'][0]:+.3f}, {j['bootstrap_95_ci'][1]:+.3f}]"
              + (f"\n    clean: {mc['became_clean']} images became clean, {mc['became_dirty']} became unclean, "
                 f"McNemar p = {mc['p_value']:.2g}" if mc else "\n    clean: McNemar skipped (judgment dirs not found)"))
        if "matched_length" in s:
            m = s["matched_length"]
            print(f"    matched length: judged {m['judged']['mean_delta']:+.3f} "
                  f"[{m['judged']['bootstrap_95_ci'][0]:+.3f}, {m['judged']['bootstrap_95_ci'][1]:+.3f}], "
                  f"clean {100 * m['clean'][0]:.1f}% -> {100 * m['clean'][1]:.1f}%")
    if a.out:
        json.dump({"name": a.name, "table_markdown": lines, "seeds": seeds}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

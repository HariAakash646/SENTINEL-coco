"""Step 0: the COCO train2014 image list SENTINEL samples from (replaces the release's Visual Genome image_data.jsonl).

Built on the probe project's v2 splits (src/prepare/prepare_coco.py, new_data/coco_v2/):
  - first, the v2 RL-training images (--lead-manifest, rl_train.json: 3,000 train2014 images, in manifest order), the
    images the RL policy and the VES-RFT / PAPO baselines train on, so SENTINEL's training images match theirs;
  - then train2014 images with >= --min-objects distinct GT categories, in a fixed shuffled order (--seed), disjoint
    by image id from every v2 split (probe_train, rl_train: train2014; probe_test, rl_test: val2014/val2017) and not
    byte-identical to any of their images (v2 requires its splits to be disjoint by bytes too).
SENTINEL thus never trains on the probe's training images or on a test / validation image. At ~2.45 pairs per image,
7000 pairs need ~2,900 images, i.e. almost only the RL-training images.
The test split is val2014 and the pool train2014, so no test image can enter; both are asserted.
Writes <run_dir>/data/images.jsonl (image_id, image_path, question: the eval prompt), images_meta.json and, with
--tsv, the image id / file / md5 list run_pipeline.sh downloads from (chair_coco/data/train2014_images.tsv).

    python chair_coco/prepare_data.py --config chair_coco/configs/qwen2_5_vl_3b.yaml --run-dir R --n-images 16000 \\
        --tsv chair_coco/data/train2014_images.tsv
"""
import argparse
import glob
import hashlib
import json
import os
import random

from config import parse_with_config

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir")
    p.add_argument("--instances", help="instances_train2014.json")
    p.add_argument("--image-dir", help="COCO train2014 images")
    p.add_argument("--val-image-dir", help="COCO val2014 images (to compare the v2 test images by bytes)")
    p.add_argument("--exclude-manifests", nargs="+", default=sorted(glob.glob(os.path.join(HERE, "data", "coco_v2", "*.json"))))
    p.add_argument("--test-manifest", default=os.path.join(HERE, "data", "coco_v2", "rl_test.json"))
    p.add_argument("--lead-manifest", default=os.path.join(HERE, "data", "coco_v2", "rl_train.json"),
                   help="images that come first, in manifest order ('' = none)")
    p.add_argument("--n-images", type=int, default=16000)
    p.add_argument("--min-objects", type=int, default=2)
    p.add_argument("--question", default="Describe this image in detail.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsv", help="also write image_id<TAB>file<TAB>md5 here")
    p.add_argument("--force", action="store_true")
    return parse_with_config(p, "data")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    args = parse_args()
    out_dir = os.path.join(args.run_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "images.jsonl")
    if os.path.exists(out) and not args.force:
        print(f"exists, skipping: {out} (pass --force to rebuild)")
        return

    ann = json.load(open(args.instances))
    cats = {}
    for a in ann["annotations"]:
        cats.setdefault(a["image_id"], set()).add(a["category_id"])
    excluded, excluded_bytes = set(), set()
    for path in args.exclude_manifests:
        for r in json.load(open(path)):
            excluded.add(int(r["image_id"]))
            image_dir = args.image_dir if r.get("coco_split", "train2014") == "train2014" else args.val_image_dir
            excluded_bytes.add(md5(os.path.join(image_dir, r["file_name"])))
    test_ids = {int(r["image_id"]) for r in json.load(open(args.test_manifest))}

    pool = sorted(im["id"] for im in ann["images"]
                  if len(cats.get(im["id"], ())) >= args.min_objects and im["id"] not in excluded)
    random.Random(args.seed).shuffle(pool)
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    lead = [int(r["image_id"]) for r in json.load(open(args.lead_manifest))] if args.lead_manifest else []
    chosen, sums, dropped_bytes = [], {}, []
    for image_id in lead:
        chosen.append(image_id)
        sums[image_id] = md5(os.path.join(args.image_dir, names[image_id]))
    excluded_bytes -= set(sums.values())   # the lead images' own bytes are in the excluded set; keep them
    for image_id in pool:   # walk the fixed order; an image whose bytes repeat an excluded or chosen one is skipped
        if len(chosen) == args.n_images:
            break
        path = os.path.join(args.image_dir, names[image_id])
        digest = md5(path)
        if digest in excluded_bytes or digest in sums.values():
            dropped_bytes.append(image_id)
            continue
        chosen.append(image_id)
        sums[image_id] = digest
    assert len(chosen) == args.n_images, f"pool too small: {len(chosen)}"
    assert not set(chosen) & test_ids and not set(chosen[len(lead):]) & excluded
    with open(out, "w") as f:
        for image_id in chosen:
            f.write(json.dumps({"image_id": str(image_id), "image_path": os.path.join(args.image_dir, names[image_id]),
                                "question": args.question}) + "\n")
    if args.tsv:
        with open(args.tsv, "w") as f:
            for image_id in chosen:
                f.write(f"{image_id}\t{names[image_id]}\t{sums[image_id]}\n")
    meta = {"train2014_images": len(ann["images"]), "pool": len(pool), "n_images": len(chosen),
            "lead": f"{len(lead)} images of {os.path.basename(args.lead_manifest)}" if lead else None,
            "excluded_v2_ids": len(excluded), "dropped_identical_bytes": dropped_bytes,
            "exclude_manifests": [os.path.basename(p) for p in args.exclude_manifests],
            "min_objects": args.min_objects, "seed": args.seed, "question": args.question}
    json.dump(meta, open(os.path.join(out_dir, "images_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "dropped_identical_bytes"} | {"dropped_identical_bytes": len(dropped_bytes)}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

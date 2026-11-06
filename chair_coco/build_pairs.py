"""Step 2: results/<model>[.shard<i>]_data_pair.jsonl -> data/<out-file> for train_cdpo.py.

Same selection as utils/get_llama_factory_data_pair.py: the first --max-pairs pairs, no shuffling, no padding.
"First" is manifest order (data/images.jsonl), the order one process works through the images, so the result
does not depend on how many shards generated them. Only pairs from the finished prefix of the manifest are
used (every image before the first unfinished one), so a run whose shards are at different points gives the
same pairs as a single process would. --max-images N restricts the prefix to the first N manifest images
(an image-matched arm).

Two differences from the release script: it reads entry["win"] / entry["lose"], which run_gen_dataset never
writes (it writes y_win / y_lose), so it would emit empty responses; this reads y_win / y_lose. And exact
duplicate rows, which a resumed generation can append twice, are dropped.

    python chair_coco/build_pairs.py --config chair_coco/configs/qwen2_5_vl_3b.yaml
    python chair_coco/build_pairs.py --config chair_coco/configs/qwen2_5_vl_3b.yaml --max-images 1000 \
        --max-pairs 0 --out-file train_pairs_1000img.jsonl
"""
import argparse
import json
import os
from collections import Counter

from config import parse_with_config
from generate import finished_images, result_files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--name", help="results/<name>[.shard<i>]_data_pair.jsonl; default: model basename")
    p.add_argument("--run-dir")
    p.add_argument("--max-pairs", type=int, default=7000, help="release DATA_CNT_LIMIT; 0 = no cap")
    p.add_argument("--max-images", type=int, default=0, help="only the first N manifest images; 0 = no cap")
    p.add_argument("--out-file", default="train_pairs.jsonl", help="under <run_dir>/data")
    return parse_with_config(p, "build")


def main():
    args = parse_args()
    name = args.name or os.path.basename(args.model.rstrip("/"))
    manifest = [json.loads(line)["image_id"] for line in open(os.path.join(args.run_dir, "data", "images.jsonl"))]
    order = {image_id: i for i, image_id in enumerate(manifest)}
    done = finished_images(args.run_dir, name)
    prefix = next((i for i, image_id in enumerate(manifest) if image_id not in done), len(manifest))
    limit = min(prefix, args.max_images) if args.max_images else prefix

    sources = result_files(args.run_dir, name, "_data_pair")
    seen, rows = set(), []
    for src in sources:
        for line in open(src):
            if not line.strip():
                continue
            e = json.loads(line)
            key = (e["image_id"], e["context"], e["y_win"], e["y_lose"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"image_id": e["image_id"], "image_path": e["image_path"], "question": e["question"],
                         "context": e["context"], "chosen": e["y_win"], "rejected": e["y_lose"],
                         "objects_of_y_win": e["objects_of_y_win"], "hallu_objects_of_y_lose": e["hallu_objects_of_y_lose"]})
    rows.sort(key=lambda r: order[r["image_id"]])   # stable: keeps each image's pairs in generation order
    in_prefix = [r for r in rows if order[r["image_id"]] < limit]
    kept = in_prefix[: args.max_pairs] if args.max_pairs else in_prefix

    out = os.path.join(args.run_dir, "data", args.out_file)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_image = Counter(r["image_id"] for r in kept)
    truncated = bool(args.max_pairs) and len(in_prefix) > args.max_pairs
    images_used = max(order[r["image_id"]] for r in kept) + 1 if truncated else limit   # incl. images without pairs
    meta = {"sources": sources, "pairs_in_sources": len(seen), "images_finished": len(done),
            "finished_prefix": prefix, "max_images": args.max_images, "max_pairs": args.max_pairs, "kept": len(kept),
            "images_used": images_used, "images_with_pairs": len(per_image),
            "pairs_per_image": round(len(kept) / max(1, images_used), 2),
            "empty_context": sum(not r["context"] for r in kept),
            "chosen_ends_with_newline": sum(r["chosen"].endswith("\n") for r in kept),
            "rejected_ends_with_newline": sum(r["rejected"].endswith("\n") for r in kept)}
    json.dump(meta, open(out.removesuffix(".jsonl") + "_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))
    if args.max_pairs and len(kept) < args.max_pairs:
        print(f"WARNING: only {len(kept)} pairs in the first {limit} manifest images; generate more to reach {args.max_pairs}")


if __name__ == "__main__":
    main()

"""The authors' released preference pairs as this pipeline's train_pairs_<N>.jsonl, to train on their data.

The release (huggingface.co/datasets/psp-dada/SENTINEL) has one LLaMA-Factory file per model, e.g.
Qwen2_5_VL_7B_Instruct_SENTINEL_7k.json: {"instruction": "<image>" + question, "context", "chosen", "rejected",
"images": ["/your/path/to/VisualGenome/VG_100K[_2]/<id>.jpg"]}. This downloads that file and the Visual Genome images
it names (only those, from Stanford's VG mirror, same folder), and writes the rows in the release's order as
build_pairs.py would: image_id, image_path, question (the instruction without "<image>"; the release uses four
prompts), context, chosen, rejected. Every image is opened once to check it. Nothing else is changed.

    python chair_coco/import_release_pairs.py --file Qwen2_5_VL_7B_Instruct_SENTINEL_7k.json \
        --vg-dir WORK/visual_genome --out RUN/data/train_pairs_7000.jsonl
"""
import argparse
import json
import os
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

HF = "https://huggingface.co/datasets/psp-dada/SENTINEL/resolve/main/"
VG = "https://cs.stanford.edu/people/rak248/"


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    tmp = dst + ".part"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dst)
            return
        except Exception as e:  # retry transient network errors
            err = e
    raise RuntimeError(f"download failed: {url}: {err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="release file name (downloaded from the HF dataset) or a local path")
    p.add_argument("--vg-dir", required=True, help="where the Visual Genome images go (VG_100K/, VG_100K_2/)")
    p.add_argument("--out", required=True, help="RUN/data/train_pairs_<N>.jsonl")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    src = a.file
    if not os.path.exists(src):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(a.out)), "release_" + os.path.basename(a.file))
        fetch(HF + os.path.basename(a.file), src)
    rows = json.load(open(src))

    def local(path):
        folder, name = path.split("/")[-2:]
        assert folder in ("VG_100K", "VG_100K_2") and name.endswith(".jpg"), path
        return folder, name

    images = sorted({local(r["images"][0]) for r in rows})
    for folder in {f for f, _ in images}:
        os.makedirs(os.path.join(a.vg_dir, folder), exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda fn: fetch(VG + "/".join(fn), os.path.join(a.vg_dir, *fn)), images))
    for fn in images:
        with Image.open(os.path.join(a.vg_dir, *fn)) as im:
            im.convert("RGB").size

    out = []
    for r in rows:
        assert len(r["images"]) == 1 and r["instruction"].startswith("<image>"), r["instruction"][:40]
        assert r["chosen"] and r["rejected"]
        folder, name = local(r["images"][0])
        out.append({"image_id": f"vg_{name[:-4]}", "image_path": os.path.abspath(os.path.join(a.vg_dir, folder, name)),
                    "question": r["instruction"][len("<image>"):], "context": r.get("context", ""),
                    "chosen": r["chosen"], "rejected": r["rejected"]})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    meta = {"source": HF + os.path.basename(a.file) if src != a.file else os.path.abspath(a.file),
            "kept": len(out), "images": len(images), "questions": Counter(r["question"] for r in out)}
    json.dump(meta, open(a.out[:-len(".jsonl")] + "_meta.json", "w"), indent=1)
    print(f"{len(out)} release pairs, {len(images)} Visual Genome images -> {a.out}")


if __name__ == "__main__":
    main()

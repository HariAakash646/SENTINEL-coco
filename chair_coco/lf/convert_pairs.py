"""The training pairs in LLaMA-Factory's format: what the release's utils/get_llama_factory_data_pair.py writes.

Each pair becomes {"instruction": "<image>" + question, "context", "chosen", "rejected", "images": [path]}, in file
order, no shuffling or padding (the release's settings). One fix: the release script reads the keys "win" / "lose",
which generation does not write (it writes y_win / y_lose; build_pairs.py stores them as chosen / rejected), so it
would emit empty responses. Also writes dataset_info.json with the release's entry for its Qwen2.5-VL set
(llamafactory/data/dataset_info.json), under the dataset name given.

    python chair_coco/lf/convert_pairs.py --pairs RUN/data/train_pairs_2000.jsonl --out-dir DIR --name sentinel
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="build_pairs.py output (jsonl)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="sentinel", help="dataset name for --dataset")
    a = p.parse_args()
    rows = [json.loads(line) for line in open(a.pairs) if line.strip()]
    out = []
    for r in rows:
        assert r["chosen"] and r["rejected"], f"empty response in {r['image_id']}"
        assert os.path.isabs(r["image_path"]) and os.path.exists(r["image_path"]), r["image_path"]
        out.append({"instruction": f"<image>{r['question']}", "context": r.get("context", ""),
                    "chosen": r["chosen"], "rejected": r["rejected"], "images": [r["image_path"]]})
    os.makedirs(a.out_dir, exist_ok=True)
    data_file = f"{a.name}.json"
    with open(os.path.join(a.out_dir, data_file), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    info = {a.name: {"file_name": data_file, "ranking": True,
                     "columns": {"prompt": "instruction", "chosen": "chosen", "context": "context",
                                 "rejected": "rejected", "images": "images"}}}
    json.dump(info, open(os.path.join(a.out_dir, "dataset_info.json"), "w"), indent=1)
    print(f"{len(out)} pairs -> {os.path.join(a.out_dir, data_file)} (dataset '{a.name}')")


if __name__ == "__main__":
    main()

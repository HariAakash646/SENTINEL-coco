"""gpt-5.1 judging under a spend cap. Same file in ~/SENTINEL/chair_coco and ~/Nullu/chair_coco.

Runs judge_openai.py (eval/, vendored unchanged from the probe repo) in chunks of images: that script is resumable (an image already
judged is skipped), takes --limit N (the first N images in sorted order) and ends with a line giving the cost of
its call ("... images=N ... ~$X ..."). Before each chunk the wrapper projects its cost from the most expensive
chunk so far (per image), and stops if the ledger's total plus that projection would exceed --budget. Every chunk's
cost is appended to --ledger (jsonl), so the cap covers all judging recorded there, across re-runs.

The costs are judge_openai.py's own estimate: usage tokens x its PRICE table (gpt-5.1: $1.25 / $10 per 1M input /
output tokens), with no cached-input discount. Replies it could not parse after retries are billed by OpenAI but
not counted, so keep a margin; a hard limit can only be set on the OpenAI account.

Exit codes: 0 = every image judged; 3 = stopped at the budget (judgments so far are kept; raise the budget and
re-run to continue); 4 = a chunk judged no new image (API errors), stopped instead of paying for it again.

The API key comes from OPENAI_API_KEY or from a .env file (OPENAI_API_KEY=...) in --key-dir, by default the
repository root; .env is git-ignored.

    python chair_coco/judge_capped.py --rundir EVAL/NAME --outdir EVAL/judge_NAME --budget 5 --ledger EVAL/judge_spend.jsonl
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR_PER_IMAGE = 0.005   # $ per image before the first chunk is measured (GRPO ck250's 600: $1.42, ~0.0024)
DONE_LINE = re.compile(r"images=(\d+).*~\$([\d.]+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--budget", type=float, required=True, help="dollars, total over the ledger")
    p.add_argument("--ledger", required=True)
    p.add_argument("--key-dir", default=os.path.dirname(HERE), help="where a .env with OPENAI_API_KEY may live")
    p.add_argument("--judge-script", default=os.path.join(HERE, "eval", "judge_openai.py"))
    p.add_argument("--sys-file", default=os.path.join(HERE, "eval", "judge_v3.txt"))
    p.add_argument("--first-chunk", type=int, default=20, help="images in the first chunk, which calibrates the cost")
    p.add_argument("--chunk", type=int, default=100)
    return p.parse_args()


def read_ledger(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    args = parse_args()
    args.judge_script, args.sys_file = os.path.abspath(args.judge_script), os.path.abspath(args.sys_file)
    args.rundir, args.outdir, args.ledger = map(os.path.abspath, (args.rundir, args.outdir, args.ledger))
    os.chdir(args.key_dir)                          # judge_openai.py reads .env from the working directory
    sys.path.insert(0, os.path.dirname(args.judge_script))
    from judge_openai import image_complete         # the judge's own definition of "judged"

    files = sorted(f for f in glob.glob(os.path.join(args.rundir, "*.json")) if os.path.basename(f) != "meta.json")
    per_image, previous = None, None
    while True:
        remaining = [i for i, f in enumerate(files) if not image_complete(f, args.outdir)]
        if previous is not None and len(remaining) >= previous:
            print(f"[judge_capped] the last chunk judged no new image (API errors or unreadable replies); stopping "
                  f"so the same images are not paid for again. See the judge output above.", flush=True)
            return 4
        previous = len(remaining)
        spent = sum(e["cost"] for e in read_ledger(args.ledger))
        if not remaining:
            print(f"[judge_capped] all {len(files)} images judged in {args.outdir}; ledger total ${spent:.2f} "
                  f"of ${args.budget:.2f}", flush=True)
            return 0
        size = min(args.first_chunk if per_image is None else args.chunk, len(remaining))
        projected = (per_image or PRIOR_PER_IMAGE) * size
        if spent + projected > args.budget:
            print(f"[judge_capped] STOP: ledger total ${spent:.2f} + next {size} images (~${projected:.2f}) would exceed "
                  f"the ${args.budget:.2f} budget; {len(remaining)} of {len(files)} images left unjudged in {args.outdir}. "
                  f"Raise JUDGE_BUDGET and re-run to continue.", flush=True)
            return 3
        limit = remaining[size - 1] + 1
        print(f"[judge_capped] judging {size} more images ({len(files) - len(remaining)}/{len(files)} done), "
              f"spent so far ${spent:.2f} of ${args.budget:.2f}", flush=True)
        proc = subprocess.Popen([sys.executable, args.judge_script, "--rundir", args.rundir, "--outdir", args.outdir,
                                 "--sys-file", args.sys_file, "--limit", str(limit)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if m := DONE_LINE.search(line):
                last = (int(m.group(1)), float(m.group(2)))
        rc = proc.wait()
        if last is None:                            # crashed or killed before its summary: book the projection
            entry = {"cost": projected, "images": size, "estimated": True}
        else:
            entry = {"cost": last[1], "images": last[0], "estimated": False}
        entry.update(time=time.strftime("%F %T"), outdir=args.outdir, returncode=rc)
        with open(args.ledger, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if rc != 0:
            print(f"[judge_capped] judge_openai.py exited with {rc}; ledger updated, stopping", flush=True)
            return rc
        if entry["images"]:
            per_image = max(per_image or 0.0, entry["cost"] / entry["images"])


if __name__ == "__main__":
    sys.exit(main())

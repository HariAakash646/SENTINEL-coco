"""Re-judge cached rollouts with an OpenAI vision model (gpt-5.1) + the v3 prompt.

Reads a rollout run from rollout_acts.py and writes one <image_id>.json per image:
{image_id, path, rollouts:[{rollout, raw, n_gen_tokens, attempts}]}, so src/train/judge_labels.py
can normalize the raw findings and align them to caption spans.

Concurrent (ThreadPoolExecutor), resumable (an image with a .json is skipped), and cost-tracked.
JSON is requested in json_object mode + the prompt's schema; a reply that still is not readable
is retried up to --max-retries.

    export OPENAI_API_KEY=...   # or leave it in .env as OPEN_AI_API_KEY
    python src/extract/judge_openai.py --rundir runs/3b_512/rollouts \\
        --outdir runs/3b_512/judge_gpt5v3 --sys-file src/extract/judge_v3.txt
"""
import argparse, base64, glob, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed


def image_complete(source_path, outdir):
    """A resumable judge image must cover every rollout in the current source file."""
    try:
        source = json.load(open(source_path))
        output = json.load(open(os.path.join(outdir, os.path.basename(source_path))))
        expected = {row["rollout"] for row in source["rollouts"]}
        # parseable(), not merely a string: call() returns its error text as the raw reply when
        # retries run out, so an isinstance check counts a rollout lost to a 429 or a timeout as
        # judged and never retries it. judge_labels.py then drops it, shrinking the label set.
        actual = {row["rollout"] for row in output["rollouts"]
                  if isinstance(row.get("raw"), str) and parseable(row["raw"])}
        return output.get("image_id") == source.get("image_id") and actual == expected
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False

MODEL = "gpt-5.1"
# gpt-5.1 (in) / (out) $ per 1M tokens, for the running cost estimate only
PRICE = {"gpt-5.1": (1.25, 10.0), "gpt-5.1-mini": (0.25, 2.0), "gpt-4o": (2.5, 10.0),
         "gpt-5.4-mini": (0.75, 4.5), "gpt-5.4-mini-2026-03-17": (0.75, 4.5),
         "gpt-5.4-nano": (0.20, 1.25), "gpt-5.4-nano-2026-03-17": (0.20, 1.25)}

JUDGE_USER = """Caption to check:
---
{caption}
---
List every statement this image does not support."""


def load_key(var="OPENAI_API_KEY"):
    """Return the API key for `var`, reading .env if it is not already exported."""
    if os.environ.get(var):
        return os.environ[var]
    if os.path.exists(".env"):
        # OPENAI_API_KEY also matches the legacy OPEN_AI_API_KEY spelling.
        pattern = r"\s*OPEN_?AI_API_KEY\s*=\s*(.+)" if var == "OPENAI_API_KEY" \
            else rf"\s*{re.escape(var)}\s*=\s*(.+)"
        for line in open(".env"):
            m = re.match(pattern, line)
            if m:
                os.environ[var] = m.group(1).strip().strip('"').strip("'")
                return os.environ[var]
    raise SystemExit(f"no API key: set {var} in the environment or in .env")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", default="runs/3b_512/rollouts")
    p.add_argument("--outdir", default="runs/3b_512/judge_gpt5v3")
    p.add_argument("--sys-file", default="src/extract/judge_v3.txt")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="env var (or .env entry) holding the key for --base-url")
    p.add_argument("--detail", default="high", choices=["high", "low"])
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=4000)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="first N images (0=all); for a smoke test")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parseable(raw):
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j < i:
        return False
    try:
        d = json.loads(raw[i:j + 1])
    except json.JSONDecodeError:
        return False
    return isinstance(d, dict) and isinstance(d.get("hallucinations"), list)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    key = load_key(args.api_key_env)
    from openai import OpenAI
    # we do our own retry/backoff
    client = OpenAI(api_key=key, timeout=120, max_retries=0,
                    **({"base_url": args.base_url} if args.base_url else {}))
    SYS = open(args.sys_file).read()

    skip = {"meta.json", "judge_free.json", "judge_labels.json"}
    files = sorted(
        f
        for f in glob.glob(os.path.join(args.rundir, "*.json"))
        if os.path.basename(f) not in skip
    )
    if args.limit:
        files = files[:args.limit]
    todo = [f for f in files
            if args.overwrite or not image_complete(f, args.outdir)]
    print(f"{len(files)} images, {len(files)-len(todo)} already judged, {len(todo)} to do "
          f"(model={args.model}, concurrency={args.concurrency})", flush=True)

    json.dump({"judge_model": args.model, "base_url": args.base_url,
               "rundir": args.rundir, "detail": args.detail,
               "sys_file": args.sys_file, "system_prompt": SYS, "user_prompt": JUDGE_USER,
               "response_format": "json_object", "max_new_tokens": args.max_new_tokens},
              open(os.path.join(args.outdir, "meta.json"), "w"), indent=1)

    b64_cache, cache_lock = {}, threading.Lock()

    def img_b64(path):
        with cache_lock:
            if path in b64_cache:
                return b64_cache[path]
        b = base64.b64encode(open(path, "rb").read()).decode()
        with cache_lock:
            b64_cache[path] = b
        return b

    def call(path, caption):
        """One judge call -> (raw_text, completion_tokens, prompt_tokens, attempts)."""
        url = f"data:image/jpeg;base64,{img_b64(path)}"
        last = ""
        # Retry with IDENTICAL params -- the retry is for transient API errors (timeout/429),
        # not for bad JSON (json_object mode is reliable). gpt-5.1 rejects non-default
        # temperature by timing out, so we never touch it.
        for attempt in range(1, args.max_retries + 2):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": [
                                  {"type": "text", "text": JUDGE_USER.format(caption=caption)},
                                  {"type": "image_url",
                                   "image_url": {"url": url, "detail": args.detail}}]}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=args.max_new_tokens)
                raw = r.choices[0].message.content or ""
                if parseable(raw):
                    return raw, r.usage.completion_tokens, r.usage.prompt_tokens, attempt
                last = raw
            except Exception as e:
                last = f"__error__ {type(e).__name__}: {str(e)[:120]}"
                time.sleep(min(2 ** attempt, 30))       # backoff on 429/timeout
        return last, 0, 0, args.max_retries + 1

    # one work item per ROLLOUT; results collected per image, written when the image is complete
    docs = {}       # image path -> loaded doc
    pending = {}    # image_id -> {rollout: result}
    expected = {}   # image_id -> n_rollouts
    for f in todo:
        d = json.load(open(f))
        docs[f] = d
        expected[d["image_id"]] = len(d["rollouts"])
        pending[d["image_id"]] = {}

    tasks = []      # (image_id, path, rollout, caption)
    for f in todo:
        d = docs[f]
        for r in d["rollouts"]:
            tasks.append((d["image_id"], d["path"], r["rollout"], r["text"]))

    t0 = time.time()
    done_imgs = 0
    tot_in = tot_out = n_bad = n_retried = 0
    lock = threading.Lock()
    pi, po = PRICE.get(args.model, (1.25, 10.0))

    def work(t):
        iid, path, roll, cap = t
        raw, ntok, ptok, attempts = call(path, cap)
        return iid, path, roll, raw, ntok, ptok, attempts

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs)):
            iid, path, roll, raw, ntok, ptok, attempts = fut.result()
            with lock:
                tot_in += ptok; tot_out += ntok
                if attempts > 1:
                    n_retried += 1
                if not parseable(raw):
                    n_bad += 1
                pending[iid][roll] = {"rollout": roll, "raw": raw,
                                      "n_gen_tokens": ntok, "attempts": attempts}
                if len(pending[iid]) == expected[iid]:      # image complete -> write it
                    d = next(v for v in docs.values() if v["image_id"] == iid)
                    outp = os.path.join(args.outdir, f"{iid}.json")
                    rolls = [pending[iid][r["rollout"]] for r in d["rollouts"]]
                    json.dump({"image_id": iid, "path": path, "n_rollouts": len(rolls),
                               "rollouts": rolls}, open(outp, "w"))
                    done_imgs += 1
                if (k + 1) % 200 == 0 or k + 1 == len(tasks):
                    el = time.time() - t0
                    cost = tot_in / 1e6 * pi + tot_out / 1e6 * po
                    eta = (len(tasks) - k - 1) * el / (k + 1) / 60
                    print(f"{k+1}/{len(tasks)} rollouts  {done_imgs} imgs  "
                          f"retried={n_retried} unreadable={n_bad}  ${cost:.2f}  "
                          f"{el/60:.1f}m elapsed  eta {eta:.0f}m", flush=True)

    cost = tot_in / 1e6 * pi + tot_out / 1e6 * po
    print(f"\ndone -> {args.outdir}  images={done_imgs}  retried={n_retried}  "
          f"unreadable={n_bad}  tok in={tot_in} out={tot_out}  ~${cost:.2f}  "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

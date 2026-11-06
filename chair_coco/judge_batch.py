"""gpt-5.1 judging through the OpenAI Batch API, under a spend cap: the paper's way of judging, at half the price of
live calls.

A standalone port of the probe repo's src/extract/judge_batch.py (commit b48515c), which the paper's v2 evals used
(their judged meta.json: transport "openai-batch", image_source "url"). It builds exactly the request
eval/judge_openai.py sends live -- the same model, system prompt, JUDGE_USER text, image detail, JSON mode,
max_completion_tokens 4000 and default temperature -- and writes what judge_openai.py writes: one <image_id>.json per
image, rollouts [{rollout, raw, n_gen_tokens, attempts}], plus meta.json, so judge_openai.image_complete(),
compare_greedy_arms.py and paper_table.py read the results unchanged. What differs from the original is only how
inputs are found: targets are given as directories instead of through run.py's run store.

One call judges every target an eval needs (base, model, matched base, hedge-strict) as one campaign, so the wait is
paid once. The call loops until done, and is safe to stop and rerun:
  1. if batches of this campaign are running, wait for them (poll every --interval seconds), download the results
     and fold them into the judged dirs (a caption judged once is never replaced);
  2. otherwise collect the captions without a readable judgment; none left -> done (exit 0);
  3. estimate their cost at batch price; if the ledger's total plus that estimate passes --budget, stop (exit 3);
  4. write the request files (<= --max-requests requests and <= --max-mb MB each), upload, submit, book the
     estimate in the ledger, and go to 1. When a batch is collected its real cost (usage tokens x price x 0.5)
     replaces the estimate in the ledger, by an adjustment entry, so the ledger's sum stays the spend.
A batch cannot be stopped half way: the cap is checked against the estimate before submitting, and the estimate is
deliberately high (150 output tokens a reply, the paper's judgments averaged 83-124; ~1,750 input tokens a request,
about 1.5x what the live judge was billed), ~$1.8 per 1,000 requests against ~$1.2 billed. Ctrl-C stops the waiting, not
the batches; rerun to pick them up, or --cancel them. A retry round (requests that failed or expired) embeds the
images (base64), so a URL the API could not fetch is not a dead end. A round that judges nothing new stops (exit 4);
a rerun tries once more.

Images: --image-source url (the paper's) sends http://images.cocodataset.org/<split>/<file>, which serves the same
bytes as the local copies (setup checks their md5 against the COCO downloads); a file not named like a COCO 2014
image is embedded instead. base64 embeds every image, as judge_openai.py does: ~200 KB a request, ~1 GB to upload
for a first eval.

    python chair_coco/judge_batch.py --workdir EVAL/judge_batch --budget 10 --ledger EVAL/judge_spend.jsonl \\
        --target CAPTIONS_DIR,JUDGED_DIR[,SYS_FILE] [--target ...]
    python chair_coco/judge_batch.py --workdir EVAL/judge_batch --status | --cancel

Exit codes: 0 every target fully judged; 3 stopped at the budget; 4 a round judged nothing new; 5 --no-wait and
batches still running. The key: OPENAI_API_KEY, or OPENAI_API_KEY=... in a .env in --key-dir (the repository root).
"""
import argparse
import base64
import collections
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from judge_openai import JUDGE_USER, MODEL, PRICE, image_complete, load_key, parseable  # noqa: E402

ENDPOINT = "/v1/chat/completions"
MAX_NEW_TOKENS = 4000            # judge_openai.py's --max-new-tokens default
BATCH_PRICE = 0.5                # Batch API price relative to live calls
IMAGE_TOKENS = {"high": 765, "low": 85}   # a 640x480 COCO image in OpenAI's tile accounting (estimate only)
EST_OUTPUT_TOKENS = 150          # per reply, for the estimate (paper's v2 judgments: mean 83-124, max 632)
SKIP = {"meta.json", "judge_free.json", "judge_labels.json"}
TERMINAL = {"completed", "expired", "cancelled", "failed"}
COCO_FILE = re.compile(r"COCO_(train|val)2014_\d{12}\.jpg$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workdir", required=True, help="campaign dir: state.json, request and result files")
    p.add_argument("--target", action="append", default=[], metavar="CAPTIONS_DIR,JUDGED_DIR[,SYS_FILE]")
    p.add_argument("--budget", type=float, help="dollars, total over the ledger")
    p.add_argument("--ledger", help="spend ledger (jsonl), shared with judge_capped.py")
    p.add_argument("--sys-file", default=os.path.join(HERE, "eval", "judge_v3.txt"), help="default system prompt")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--detail", default="high", choices=["high", "low"])
    p.add_argument("--image-source", default="url", choices=["url", "base64"])
    p.add_argument("--max-requests", type=int, default=1000, help="requests per batch file (API cap 50,000)")
    p.add_argument("--max-mb", type=float, default=190, help="MB per batch file (API cap 200)")
    p.add_argument("--interval", type=int, default=120, help="seconds between status checks")
    p.add_argument("--limit-images", type=int, default=0, help="first N images of each target (smoke tests)")
    p.add_argument("--key-dir", default=os.path.dirname(HERE), help="where a .env with OPENAI_API_KEY may live")
    p.add_argument("--dry-run", action="store_true", help="count requests and cost; write and spend nothing")
    p.add_argument("--no-wait", action="store_true", help="submit, then exit 5 instead of waiting")
    p.add_argument("--status", action="store_true", help="print the campaign's batches and exit")
    p.add_argument("--cancel", action="store_true", help="cancel the campaign's running batches and exit")
    a = p.parse_args()
    if not (a.status or a.cancel):
        if not a.target:
            p.error("--target is required")
        if not a.dry_run and (a.budget is None or a.ledger is None):
            p.error("--budget and --ledger are required")
    return a


def log(msg):
    print(f"[judge_batch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path, doc, **kw):
    with open(path + ".tmp", "w") as handle:
        json.dump(doc, handle, **kw)
    os.replace(path + ".tmp", path)


def load_state(workdir):
    path = os.path.join(workdir, "state.json")
    return json.load(open(path)) if os.path.exists(path) else {"rounds": []}


def save_state(workdir, state):
    write_json(os.path.join(workdir, "state.json"), state, indent=1)


def ledger_append(path, entry):
    entry["time"] = time.strftime("%F %T")
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def ledger_spent(path):
    if not path or not os.path.exists(path):
        return 0.0
    return sum(json.loads(line)["cost"] for line in open(path) if line.strip())


def observed_cost_per_request(path):
    """Mean real batch cost per request over the batches collected so far (0 if none)."""
    if not path or not os.path.exists(path):
        return 0.0
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r.get("kind") == "batch-actual" and r.get("requests")]
    n = sum(r["requests"] for r in rows)
    return sum(r["actual"] for r in rows) / n if n else 0.0


def parse_target(spec, default_sys):
    parts = spec.split(",")
    if len(parts) not in (2, 3):
        raise SystemExit(f"--target {spec!r}: expected CAPTIONS_DIR,JUDGED_DIR[,SYS_FILE]")
    sys_file = os.path.abspath(parts[2] if len(parts) == 3 else default_sys)
    return {"source_dir": os.path.abspath(parts[0]), "judged_dir": os.path.abspath(parts[1]), "sys_file": sys_file}


def source_files(source_dir, limit=0):
    files = sorted(f for f in glob.glob(os.path.join(source_dir, "*.json")) if os.path.basename(f) not in SKIP)
    return files[:limit] if limit else files


def judged_rows(judged_dir, image_id):
    """rollout -> row of the image's existing judgment file, if any."""
    try:
        doc = json.load(open(os.path.join(judged_dir, f"{image_id}.json")))
        return {row["rollout"]: row for row in doc["rollouts"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def request_body(model, detail, system, caption, url):
    """judge_openai.call()'s request, as a Batch API body."""
    return {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [
                             {"type": "text", "text": JUDGE_USER.format(caption=caption)},
                             {"type": "image_url", "image_url": {"url": url, "detail": detail}}]}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": MAX_NEW_TOKENS}


def coco_url(path):
    name = os.path.basename(path)              # COCO_val2014_000000001993.jpg
    return f"http://images.cocodataset.org/{name.split('_')[1]}/{name}" if COCO_FILE.match(name) else None


def pending_requests(targets, limit):
    """(target index, source doc, rollout row) for every caption without a readable judgment."""
    out = []
    for k, t in enumerate(targets):
        for f in source_files(t["source_dir"], limit):
            if image_complete(f, t["judged_dir"]):
                continue
            doc = json.load(open(f))
            have = judged_rows(t["judged_dir"], doc["image_id"])
            for row in doc["rollouts"]:
                old = have.get(row["rollout"])
                if not (old and isinstance(old.get("raw"), str) and parseable(old["raw"])):
                    out.append((k, doc, row))
    return out


def client():
    from openai import OpenAI
    return OpenAI(api_key=load_key(), timeout=600, max_retries=3)


def write_round(args, workdir, number, targets, todo, image_source):
    """Write the request files of one round; return its chunks, the estimate and how many images were embedded."""
    rdir = os.path.join(workdir, f"round{number:02d}")
    if not args.dry_run:
        os.makedirs(rdir, exist_ok=True)
    systems = [open(t["sys_file"]).read() for t in targets]
    images, embedded = {}, set()
    chunks, handle, count, size, est_in = [], None, 0, 0, 0.0

    def image_url(path):
        if path not in images:
            url = coco_url(path) if image_source == "url" else None
            if url is None:
                url = "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()
                embedded.add(path)
            images[path] = url
        return images[path]

    for k, doc, row in todo:
        body = request_body(args.model, args.detail, systems[k], row["text"], image_url(doc["path"]))
        line = json.dumps({"custom_id": f"r{number}|t{k}|{doc['image_id']}|{row['rollout']}",
                           "method": "POST", "url": ENDPOINT, "body": body}) + "\n"
        if handle is None or count >= args.max_requests or size + len(line) > args.max_mb * 1e6:
            if handle:
                handle.close()
            chunks.append({"file": os.path.join(rdir, f"requests-{len(chunks):03d}.jsonl"), "n_requests": 0, "bytes": 0})
            handle = open(os.devnull if args.dry_run else chunks[-1]["file"], "w")
            count, size = 0, 0
        handle.write(line)
        count += 1
        size += len(line)
        chunks[-1]["n_requests"] += 1
        chunks[-1]["bytes"] += len(line)
        est_in += (len(systems[k]) + len(row["text"]) + len(JUDGE_USER)) / 4 + IMAGE_TOKENS[args.detail]
    if handle:
        handle.close()
    pi, po = PRICE.get(args.model, (1.25, 10.0))
    estimate = BATCH_PRICE * (est_in / 1e6 * pi + len(todo) * EST_OUTPUT_TOKENS / 1e6 * po)
    return chunks, estimate, len(embedded)


def reply(line):
    """(raw, prompt_tokens, completion_tokens) of one batch output line. A failed request becomes an unparseable
    raw string, as judge_openai.py records a call that ran out of retries, so the next round includes it."""
    response = line.get("response") or {}
    body = response.get("body") or {}
    if line.get("error") or response.get("status_code") != 200:
        error = line.get("error") or body.get("error") or {}
        return (f"__error__ {error.get('code', response.get('status_code'))}: "
                f"{str(error.get('message', ''))[:120]}", 0, 0)
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    return ((choice.get("message") or {}).get("content") or "",
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def write_meta(args, workdir, rnd, t):
    """judge_openai.py's meta.json, plus how the judgments were obtained."""
    path = os.path.join(t["judged_dir"], "meta.json")
    try:
        campaigns = json.load(open(path)).get("campaigns", [])
    except (OSError, ValueError):
        campaigns = []
    tag = f"{workdir}#round{rnd['number']:02d}"
    if tag not in campaigns:
        campaigns.append(tag)
    write_json(path, {"judge_model": rnd["model"], "base_url": None, "rundir": t["source_dir"], "detail": rnd["detail"],
                      "sys_file": t["sys_file"], "system_prompt": open(t["sys_file"]).read(), "user_prompt": JUDGE_USER,
                      "response_format": "json_object", "max_new_tokens": MAX_NEW_TOKENS, "transport": "openai-batch",
                      "endpoint": ENDPOINT, "image_source": rnd["image_source"], "campaigns": campaigns}, indent=1)


def merge(args, workdir, rnd, lines, batch_id):
    """Fold batch output lines into the judged dirs; return (readable, unreadable, prompt tokens, output tokens)."""
    by_image = collections.defaultdict(dict)
    good = bad = tok_in = tok_out = 0
    for line in lines:
        _, tk, image_id, rollout = line["custom_id"].split("|")
        raw, p, c = reply(line)
        tok_in += p
        tok_out += c
        ok = parseable(raw)
        good += ok
        bad += not ok
        by_image[(int(tk[1:]), image_id)][int(rollout)] = (raw, c)
    touched = set()
    for (k, image_id), replies in by_image.items():
        t = rnd["targets"][k]
        source = json.load(open(os.path.join(t["source_dir"], f"{image_id}.json")))
        rows = judged_rows(t["judged_dir"], source["image_id"])
        for rollout, (raw, c) in replies.items():
            old = rows.get(rollout)
            if old and isinstance(old.get("raw"), str) and parseable(old["raw"]):
                continue                # a caption is judged once; a later round never replaces it
            rows[rollout] = {"rollout": rollout, "raw": raw, "n_gen_tokens": c,
                             "attempts": (old or {}).get("attempts", 0) + 1, "batch": batch_id}
        ordered = [rows[r["rollout"]] for r in source["rollouts"] if r["rollout"] in rows]
        os.makedirs(t["judged_dir"], exist_ok=True)
        write_json(os.path.join(t["judged_dir"], f"{source['image_id']}.json"),
                   {"image_id": source["image_id"], "path": source["path"], "n_rollouts": len(ordered),
                    "rollouts": ordered})
        touched.add(k)
    for k in touched:
        write_meta(args, workdir, rnd, rnd["targets"][k])
    return good, bad, tok_in, tok_out


def refresh_and_collect(args, workdir, state, api):
    """Update every open batch; download, merge and book the finished ones. Returns the open batches left."""
    open_chunks = 0
    for rnd in state["rounds"]:
        for chunk in rnd["chunks"]:
            if chunk.get("collected") or not chunk.get("batch_id"):
                continue
            b = api.batches.retrieve(chunk["batch_id"])
            counts = b.request_counts
            chunk.update(status=b.status, output_file_id=b.output_file_id, error_file_id=b.error_file_id,
                         counts={"completed": counts.completed, "failed": counts.failed, "total": counts.total}
                         if counts else None,
                         errors=[e.message for e in (b.errors.data or [])] if b.errors else None)
            save_state(workdir, state)
            if b.status not in TERMINAL:
                open_chunks += 1
                continue
            lines = []
            for kind in ("output_file_id", "error_file_id"):
                if chunk.get(kind):
                    text = api.files.content(chunk[kind]).text
                    with open(chunk["file"].replace("requests-", f"{kind.split('_')[0]}-"), "w") as fh:
                        fh.write(text)
                    lines += [json.loads(x) for x in text.splitlines() if x.strip()]
            good, bad, tok_in, tok_out = merge(args, workdir, rnd, lines, chunk["batch_id"])
            pi, po = PRICE.get(rnd["model"], (1.25, 10.0))
            actual = BATCH_PRICE * (tok_in / 1e6 * pi + tok_out / 1e6 * po)
            if args.ledger:
                ledger_append(args.ledger, {"cost": actual - chunk["estimate"], "kind": "batch-actual",
                                            "actual": actual, "estimate": chunk["estimate"], "requests": good + bad,
                                            "batch": chunk["batch_id"], "workdir": workdir})
            chunk.update(collected=True, judged=good, unreadable=bad, actual_cost=actual)
            save_state(workdir, state)
            log(f"batch {chunk['batch_id']} {b.status}: {good} judged, {bad} unreadable, "
                f"{tok_in} in / {tok_out} out tokens, ${actual:.2f} (estimate was ${chunk['estimate']:.2f})"
                + (f"; errors: {'; '.join(chunk['errors'])}" if chunk.get("errors") else ""))
    return open_chunks


def submit_round(args, workdir, state, rnd, api):
    for chunk in rnd["chunks"]:
        if chunk.get("batch_id"):
            continue
        with open(chunk["file"], "rb") as fh:          # an hour: a large file over a slow uplink
            upload_id = api.files.create(file=fh, purpose="batch", timeout=3600).id
        batch = api.batches.create(input_file_id=upload_id, endpoint=ENDPOINT, completion_window="24h",
                                   metadata={"campaign": os.path.basename(workdir.rstrip("/")),
                                             "round": str(rnd["number"])})
        ledger_append(args.ledger, {"cost": chunk["estimate"], "kind": "batch-estimate", "requests": chunk["n_requests"],
                                    "batch": batch.id, "workdir": workdir})
        chunk.update(input_file_id=upload_id, batch_id=batch.id, status=batch.status,
                     submitted=time.strftime("%F %T"), collected=False)
        save_state(workdir, state)
        log(f"{os.path.relpath(chunk['file'], workdir)}: {chunk['n_requests']} requests "
            f"({chunk['bytes'] / 1e6:.1f} MB) -> batch {batch.id}")


def status(workdir, cancel=False):
    state = load_state(workdir)
    if not state["rounds"]:
        raise SystemExit(f"no campaign in {workdir}")
    api = client()
    for rnd in state["rounds"]:
        for chunk in rnd["chunks"]:
            line = f"round {rnd['number']:02d} {os.path.basename(chunk['file'])} {chunk.get('batch_id', '-'):34}"
            if chunk.get("batch_id") and not chunk.get("collected"):
                b = api.batches.retrieve(chunk["batch_id"])
                if cancel and b.status not in TERMINAL | {"cancelling"}:
                    b = api.batches.cancel(chunk["batch_id"])
                c = b.request_counts
                line += f" {b.status:11} {c.completed if c else 0}/{c.total if c else chunk['n_requests']} done"
            else:
                line += f" {chunk.get('status', 'not submitted'):11} " + (
                    f"collected: {chunk.get('judged')} judged, {chunk.get('unreadable')} unreadable, "
                    f"${chunk.get('actual_cost', 0):.2f}" if chunk.get("collected") else "")
            print(line)


def main():
    args = parse_args()
    workdir = os.path.abspath(args.workdir)
    if args.ledger:
        args.ledger = os.path.abspath(args.ledger)
    os.chdir(args.key_dir)                              # load_key() reads .env from the working directory
    if args.status or args.cancel:
        return status(workdir, args.cancel) or 0
    targets = [parse_target(s, args.sys_file) for s in args.target]
    if not args.dry_run:
        os.makedirs(workdir, exist_ok=True)
    state = load_state(workdir)
    api = None
    first_round_here = len(state["rounds"]) + 1   # the no-progress stop looks only at rounds submitted by this call
    while True:
        if any(c.get("batch_id") and not c.get("collected") for r in state["rounds"] for c in r["chunks"]):
            api = api or client()
            running = refresh_and_collect(args, workdir, state, api)
            if running:
                if args.no_wait:
                    log(f"{running} batch(es) running; rerun to collect them")
                    return 5
                log(f"{running} batch(es) queued or running; next check in {args.interval}s")
                time.sleep(args.interval)
                continue
        todo = pending_requests(targets, args.limit_images)
        if not todo:
            n = sum(len(source_files(t["source_dir"], args.limit_images)) for t in targets)
            log(f"every target fully judged ({len(targets)} targets, {n} images); "
                f"ledger total ${ledger_spent(args.ledger):.2f}")
            return 0
        last = state["rounds"][-1] if state["rounds"] else None
        if last and last["number"] >= first_round_here and last.get("pending_before") is not None \
                and len(todo) >= last["pending_before"]:
            log(f"the last round judged nothing new ({len(todo)} captions still unjudged; see the errors above). "
                f"Stopping rather than paying for the same requests again.")
            return 4
        number = len(state["rounds"]) + 1
        # the first round uses --image-source; a retry round embeds the images, so an unfetchable URL is no dead end
        source = args.image_source if number == 1 else "base64"
        chunks, estimate, embedded = write_round(args, workdir, number, targets, todo, source)
        per_request = max(estimate / len(todo), observed_cost_per_request(args.ledger))
        estimate = per_request * len(todo)
        spent = ledger_spent(args.ledger)
        per_target = collections.Counter(k for k, _, _ in todo)
        log(f"round {number}: {len(todo)} captions to judge in {len(chunks)} batch file(s), "
            f"{sum(c['bytes'] for c in chunks) / 1e6:.0f} MB, images by {source}"
            + (f" ({embedded} embedded)" if embedded and source == "url" else "")
            + f"; estimate ${estimate:.2f} at batch price ({args.model}, detail {args.detail})")
        for k, t in enumerate(targets):
            if per_target[k]:
                log(f"  {per_target[k]:5} -> {t['judged_dir']}  ({os.path.basename(t['sys_file'])})")
        if args.dry_run:
            log("dry run: nothing written or spent")
            return 0
        if spent + estimate > args.budget:
            log(f"STOP: ledger total ${spent:.2f} + this round's estimate ${estimate:.2f} would exceed the "
                f"${args.budget:.2f} budget; nothing submitted. Raise JUDGE_BUDGET and re-run.")
            return 3
        for c in chunks:
            c["estimate"] = per_request * c["n_requests"]
        rnd = {"number": number, "created": time.strftime("%F %T"), "model": args.model, "detail": args.detail,
               "image_source": source, "targets": targets, "pending_before": len(todo), "chunks": chunks}
        state["rounds"].append(rnd)
        save_state(workdir, state)
        api = api or client()
        try:
            submit_round(args, workdir, state, rnd, api)
        except Exception as e:  # keep what was submitted; the files left unsubmitted are dropped, the next round
            rnd["chunks"] = [c for c in rnd["chunks"] if c.get("batch_id")]  # re-prepares their captions
            rnd["pending_before"] = None    # a partial round is no evidence that the requests fail
            if not rnd["chunks"]:
                state["rounds"].pop()
            save_state(workdir, state)
            log(f"submit failed: {type(e).__name__}: {str(e)[:300]}; rerun to continue")
            return 1


if __name__ == "__main__":
    sys.exit(main())

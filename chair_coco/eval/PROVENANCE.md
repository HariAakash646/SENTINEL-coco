# Evaluation code: provenance

`../eval_probe.sh` runs the paper's Table 1 eval (the probe repo's v2 pipeline). The scripts it runs are copies from
the probe repo at commit b48515c ("v2 data pipeline ...", checkout `~/probe-rl-hallucination` on turing), byte-identical
unless noted.

| file here | origin | role |
|---|---|---|
| `rollout_acts.py`, `launch_rollout_acts.sh` | `src/extract/` | greedy captions, HF transformers `generate()` (run.py `eval:rollout`) |
| `judge_openai.py`, `judge_v3.txt` | `src/extract/` | gpt-5.1 judge and its standard prompt (run.py `eval:judge`) |
| `judge_v3_nohedge.txt` | `src/extract/` | the hedge-strict judge prompt |
| `compare_greedy_arms.py` | `src/eval/` | paired judged comparison (run.py `eval:compare`) |
| `truncation_null.py` | `src/eval/` | wrong objects, precision, tokens; matched-length base captions |
| `reference_chair_opera.py` | `store/runs/v2_qwen25vl3b/analysis/` (git-ignored there) | OPERA's copy of Rohrbach et al.'s CHAIR; identical to OPERA's `chair.py` apart from its header |
| `ref_chair.py` | `store/runs/v2_qwen25vl3b/analysis/run_ref_chair.py` (git-ignored there) | adapted: the same scoring functions, paths and arm list made arguments |
| `paper_table.py` | new | collects the eight Table 1 columns, mean ± sd over seeds, McNemar for the clean rate |
| `../data/coco_v2/*.json` | `new_data/coco_v2/` | the v2 split manifests; `rl_test.json` is the test set |

The order of steps and their arguments follow `src/run.py` and `store/post_judge_run.sh` (git-ignored in that repo).
Judging there went through the OpenAI Batch API (`src/extract/judge_batch.py`, image source `url`), with the same
model, prompt, image detail, JSON mode and token limit as `judge_openai.py`. `../judge_batch.py` is a standalone port
of it (same request body, custom ids, reply parsing and judged-file format; inputs given as directories instead of
through `run.py`'s run store) and is the default; `judge_openai.py` is still used for `JUDGE_MODE=live`.

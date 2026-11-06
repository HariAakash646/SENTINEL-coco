"""Step 1: SENTINEL's preference-data generation on COCO train2014, via the release's own code.

run_gen_dataset below is run/generate_dataset.run_gen_dataset line for line, calling the release's own
helpers: n=10 samples per step at T=0.7 that stop at the end of a sentence, coreference resolution
(fastcoref), object extraction (the CHAIR synonym list plus the FACTUAL scene-graph parser), cross-checking
with GroundingDINO-base and YOLO11x, context-coherent pairing (maybe_build_pair), and iterative contextual
bootstrapping. One line differs, selected by --pair-text:

  - raw (default): the sentences stored as y_win / y_lose and appended to the context are the model's own
    text; the coreference-resolved copies are used only for object extraction. This matches the released
    data: in the Qwen2.5-VL-7B set, 470 of 14,000 responses start with a pronoun, 1,226 contexts contain a
    pronoun-initial sentence and 3 responses start lowercase (all quote continuations).
  - resolved: the public code, which stores the resolved text ("It has a basket." -> "a bicycle has a
    basket.", "this image" -> "The image" mid-sentence); in our smoke run 6 of 38 responses started
    lowercase. Pairs and later prompts then contain text the model never wrote.

Three other things are swapped, only because they tie the release to its own environment or model family:

  - the generator: model/generator/{qwen2_5_vl,qwen2_vl,llava}.py are replaced by VLLMGenerator,
    which works with any HF chat template (Qwen2.5-VL, llava-hf, Gemma 3). For Qwen it builds the same
    prompt as the release (chat template + generation prompt + context), stops on the same tokens
    (".", "!", "?", ".\\n", ".\\n\\n", EOS), and pins top_p=1, top_k off, repetition_penalty=1, because
    vLLM >= 0.8 would otherwise apply Qwen's generation_config (top_k=1 = greedy, rep. penalty 1.05);
    the release ran vllm 0.7.3, which ignored it.
  - GroundingDINO post-processing: transformers >= 4.51 renamed box_threshold to threshold and moved
    the phrases to "text_labels". Thresholds stay 0.35 / 0.25. fastcoref's Longformer is loaded with
    eager attention (transformers 5 would pick SDPA, which Longformer lacks; 4.51 used eager).
  - GVars is filled from our config instead of utils/setup_utils.parse_arg, and tool weights live
    on /scratch (YOLO .pt and the spaCy model under <tools_dir>, HF models in the shared hub cache).

Resumable: images already in any results/<model>*.jsonl are skipped (the release's DataSet.filter rule).
Images are processed in manifest order; --num-images caps how many new ones this call takes.
Several GPUs: run one process per GPU with --shard i --num-shards k; shard i takes manifest images i, i+k, ...
and writes results/<model>.shard<i>.jsonl (+ _data_pair); build_pairs.py merges the shards in manifest order.

FLOPs (flops.py convention) are counted as it runs into <run_dir>/flops/generate_<name>.json, which a resumed run
continues, so it holds the compute actually spent, including images redone after an interruption:
  - vlm_prefill / vlm_decode: the policy's sampling, from the profile (--flop-profile) and the token counts;
    one prefill per request (image + prompt + context) shared by its n=10 samples. Each step re-prefills
    the whole context: vLLM may reuse cached KV blocks of the previous step, so this part is an upper
    bound. extra/vlm_prefill_if_unshared is the prefill counted once per sample instead;
  - yolo11x, fastcoref_lingmess, factual_t5_parser, grounding_dino_base: measured with FlopCounterMode
    around the release's calls (yolo_detect, resolve_corefs, SGParser.pharse, b_get_hallu_objects).
    spaCy runs on thinc / cupy, which torch does not see; it is a small CNN and is not counted.

    CUDA_VISIBLE_DEVICES=1 $ENV/bin/python chair_coco/generate.py --config chair_coco/configs/qwen2_5_vl_3b.yaml
"""
import argparse
import json
import logging
import os
import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field

from config import parse_with_config
from flops import FlopLedger, decode, image_token_id, load_profile, prefill, profile_meta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_STRINGS = [".", "!", "?", ".\n", ".\n\n"]  # model/generator/qwen2_5_vl.py _get_stop_words_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--name", help="results/<name>.jsonl; default: model basename")
    p.add_argument("--run-dir")
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=5, help="images in flight; release main.py default 5")
    p.add_argument("--dino-chunk", type=int, default=5, help="max images per GroundingDINO forward (memory only)")
    p.add_argument("--num-images", type=int, default=-1, help="new images to process this call (-1 = all left)")
    p.add_argument("--shard", type=int, default=0, help="this process takes manifest images shard, shard + num_shards, ...")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="release get_generator: 0.7")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE"))
    p.add_argument("--tools-dir", help="holds yolo/<name>.pt and spacy/<model>")
    p.add_argument("--pair-text", choices=("raw", "resolved"), default="raw", help="see module docstring")
    p.add_argument("--flop-profile", help="flop_profile.py output for --model")
    p.add_argument("--count-tool-flops", action=argparse.BooleanOptionalAction, default=True,
                   help="measure the detectors' and parsers' FLOPs (FlopCounterMode around each call)")
    return parse_with_config(p, "generate")


def run_gen_dataset(G, datalist, batch_size, pair_text):
    """run/generate_dataset.run_gen_dataset, with the stored sentences chosen by pair_text (marked below)."""
    from time import time

    from model.auxiliary.datastate import DataStateForBuildDataset

    GVars = G.GVars
    logger, save_path, model_dir, alter_device = GVars.logger, GVars.save_path, GVars.model_dir, GVars.alter_device
    generator = G.get_generator(use_vllm=True, debug=G.DEBUG)
    DINO_detector = G.DINO("base", model_dir=model_dir, device=alter_device, logger=logger)
    yolo = G.YoloModel("yolo11x", model_dir=model_dir, logger=logger)
    SG_parser = G.SGParser(G.DEBUG, "base", model_dir, device=alter_device, logger=logger)
    spacy = G.SpacyModel(model_size="md", model_dir=model_dir, device=alter_device, logger=logger)
    wn = G.WordnetModel(logger=logger)
    ref = G.refModel(args=GVars.args)

    data_states = []
    num_of_data, finished_data_num = len(datalist), 0
    logger.info(f"Start processing {num_of_data} data points.")
    while len(datalist) > 0 or len(data_states) > 0:
        start_time = time()
        while len(data_states) < batch_size and len(datalist) > 0:
            data_states.append(DataStateForBuildDataset(data=datalist.pop(0)))
        G.yolo_detect(yolo, data_states)
        out = generator.gen(images=[s.image for s in data_states], users=[s.question for s in data_states],
                            assistants=[s.assistant for s in data_states], do_sample=True, n=10, temp=0.7,
                            force_list=True, single_sentence=True)
        b_new_sents = out.outputs
        for idx, (new_sents, s) in enumerate(zip(b_new_sents, data_states)):
            b_new_sents[idx], s.is_finished = G.get_finish_flag(new_sents, remove_duplicates=True)
        context = [s.assistant for s in data_states]
        b_resolved_new_sents = G.resolve_corefs(spacy, b_new_sents, context, 1)

        b_object_lists = []
        for s, new_sents in zip(data_states, b_resolved_new_sents):
            object_lists = G.extract_obj_w_gt(new_sents, ref.valid_nouns, ref.double_words, ref.inv_syn_map, wn,
                                              force_list=True, return_repr=False)
            textgraphs = SG_parser.pharse(new_sents, force_list=True)
            new_object_lists = G.extract_obj_from_textgraphs(textgraphs, spacy, wn, force_list=True)
            b_object_lists.append([o + n for o, n in zip(object_lists, new_object_lists)])

        b_haluci_objects_list, b_nonhallu_objects_list = G.b_get_hallu_objects(
            b_object_lists, [s.nonhallu_objects for s in data_states], [s.hallu_objects for s in data_states],
            spacy=spacy, wn=wn, images=[s.image for s in data_states], dino=DINO_detector,
            b_yolo_results=[s.yolo_result.labels for s in data_states] if yolo else None,
            yolo_labels=yolo.labels if yolo else None, b_uncertain_objects=[s.uncertain_objects for s in data_states],
            b_detector_rejects=[s.detector_reject for s in data_states], inv_syn_map=ref.inv_syn_map)

        for s, raw_sents, new_sents, object_lists, haluci_objects_list, nonhallu_objects_list in zip(
            data_states, b_new_sents, b_resolved_new_sents, b_object_lists, b_haluci_objects_list,
            b_nonhallu_objects_list
        ):
            if not new_sents:
                continue
            stored = raw_sents if pair_text == "raw" else new_sents  # <-- the one changed line (release: new_sents)
            s.generated_sentences.append(stored)
            s.generated_objects.append(object_lists)
            s.generated_hallu_objects.append(haluci_objects_list)
            s.generated_nonhallu_objects.append(nonhallu_objects_list)
            best_idx = G.maybe_build_pair(save_path, s, spacy, wn, ref.inv_syn_map)
            s.app_assistant(stored, best_idx)

        [G.save_data_state(save_path, s, spacy, wn, ref.inv_syn_map) for s in data_states if s.is_finished]
        finished_data_num += len([s for s in data_states if s.is_finished])
        G.log_progress(logger, finished_data_num, num_of_data, batch_size, time() - start_time)
        data_states = [s for s in data_states if not s.is_finished]


@dataclass
class GenOutput:  # the one field of model/utils/gen_utils.GenOutput that run_gen_dataset reads
    outputs: list = field(default_factory=list)


def stop_token_ids(tok, text):
    """Ids that decode to exactly `text`: the encoded piece and, for SentencePiece vocabularies, the bare piece."""
    ids = set()
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc) == 1 and tok.decode(enc).strip(" ") == text:
        ids.add(enc[0])
    tid = tok.convert_tokens_to_ids(text)
    if isinstance(tid, int) and tid != tok.unk_token_id and tok.decode([tid]).strip(" ") == text:
        ids.add(tid)
    return ids


class VLLMGenerator:
    """Model-agnostic replacement for the release's generator classes; same gen() signature."""

    def __init__(self, model, max_pixels, gpu_util, max_model_len, max_num_seqs, seed, enforce_eager, logger,
                 flops=None, profile=None):
        from transformers import AutoConfig, AutoProcessor, GenerationConfig
        from vllm import LLM

        self.processor = AutoProcessor.from_pretrained(model)
        tok = self.processor.tokenizer
        self.max_pixels = max_pixels
        self.flops, self.profile = flops, profile
        self.image_token = image_token_id(AutoConfig.from_pretrained(model))
        self._prompt_cache = {}
        eos = {tok.eos_token_id}
        try:
            g = GenerationConfig.from_pretrained(model).eos_token_id
            eos |= set(g if isinstance(g, list) else [g])
        except OSError:
            pass
        self.eos_ids = sorted(i for i in eos if i is not None)
        self.stop_ids = sorted(set().union(*(stop_token_ids(tok, w) for w in STOP_STRINGS))) + self.eos_ids
        logger.info(f"generator {model}: eos {self.eos_ids}, sentence stops "
                    f"{[(i, tok.decode([i])) for i in self.stop_ids]}")
        self.llm = LLM(model=model, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=gpu_util,
                       max_model_len=max_model_len, max_num_seqs=max_num_seqs, dtype="bfloat16", seed=seed,
                       enforce_eager=enforce_eager, generation_config="vllm")

    def _image(self, image):
        w, h = image.size
        if self.max_pixels and w * h > self.max_pixels:
            s = (self.max_pixels / (w * h)) ** 0.5
            image = image.resize((max(1, int(w * s)), max(1, int(h * s))))
        return image

    def gen(self, images, users="Describe this image.", assistants="", do_sample=False, n=5, temp=0.3,
            max_tokens=512, force_list=False, single_sentence=False):
        from vllm import SamplingParams

        from model.utils.utils import ensure_lists

        images, users, assistants = ensure_lists(images, users, assistants)
        images = [self._image(im) for im in images]
        prompts = []
        for u, a in zip(users, assistants):
            # = the release's chat template with the assistant turn minus its closing "<|im_end|>\n"
            prompts.append(self._template(u) + a)
        params = SamplingParams(n=n if do_sample else 1, temperature=temp if do_sample else 0.0, top_p=1.0,
                                top_k=-1, repetition_penalty=1.0, max_tokens=max_tokens,
                                stop_token_ids=self.stop_ids if single_sentence else self.eos_ids,
                                include_stop_str_in_output=True)
        reqs = [{"prompt": p, "multi_modal_data": {"image": im}} for p, im in zip(prompts, images)]
        outs = self.llm.generate(reqs, params, use_tqdm=False)
        if self.flops is not None:
            self._count_flops(users, assistants, images, outs)
        texts = [[o.text.strip(" ") for o in c.outputs] if len(c.outputs) > 1 else c.outputs[0].text.strip(" ")
                 for c in outs]
        return GenOutput(texts if force_list or len(texts) > 1 else texts[0])

    def _template(self, user):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _prompt_tokens(self, user, assistant, image):
        """(prompt tokens with the image expanded, image tokens) of template(user) + assistant, as the model sees it.
        The image part depends only on the (resized) image size, so it is cached per (user, size)."""
        tok = self.processor.tokenizer
        key = (user, image.size)
        if key not in self._prompt_cache:
            template = self._template(user)
            ids = self.processor(text=[template], images=[image], return_tensors="pt")["input_ids"][0]
            self._prompt_cache[key] = (len(ids), int((ids == self.image_token).sum()),
                                       len(tok.encode(template, add_special_tokens=False)))
        n_prompt, n_image, n_template = self._prompt_cache[key]
        if assistant:
            n_prompt += len(tok.encode(self._template(user) + assistant, add_special_tokens=False)) - n_template
        return n_prompt, n_image

    def _count_flops(self, users, assistants, images, outs):
        for u, a, im, c in zip(users, assistants, images, outs):
            n_prompt, n_image = self._prompt_tokens(u, a, im)
            pre = prefill(self.profile, n_prompt, n_image)
            self.flops.add("vlm_prefill", pre)
            self.flops.add_extra("vlm_prefill_if_unshared", pre * len(c.outputs))
            for o in c.outputs:
                self.flops.add("vlm_decode", decode(self.profile, n_prompt, len(o.token_ids)))
            self.flops.count("vlm_requests")
            self.flops.count("vlm_samples", len(c.outputs))
            self.flops.count("vlm_prompt_tokens", n_prompt)
            self.flops.count("vlm_generated_tokens", sum(len(o.token_ids) for o in c.outputs))
        self.flops.write()


def install_patches(args, logger, flops=None, profile=None):
    """Swap the generator module, the DINO post-processing and the tool paths, and count FLOPs if `flops` is
    given; return run.generate_dataset."""
    sys.path.insert(0, REPO)
    import torch

    gen_utils = types.ModuleType("model.utils.gen_utils")
    gen_utils.GenOutput = GenOutput
    gen_utils.get_generator = lambda use_vllm=True, debug=False: VLLMGenerator(
        args.model, args.max_pixels, args.gpu_memory_utilization, args.max_model_len, args.max_num_seqs,
        args.seed, debug, logger, flops, profile)
    import model.utils  # noqa: F401  (package must exist before the submodule is injected)
    sys.modules["model.utils.gen_utils"] = gen_utils

    from model.auxiliary.global_vars import GVars
    from model.detector.grounding_dino import BOX_THRESHOLD, DINO, TEXT_THRESHOLD
    from model.detector.yolo_model import YoloModel
    from model.others.spacy_model import SpacyModel

    class DINOCompat(DINO):
        """Release _detect with the renamed threshold argument, run in chunks of at most --dino-chunk images
        (each image is detected independently, so chunking only bounds memory when --batch-size is raised)."""

        def _detect(self, images, captions, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
            results = []
            for i in range(0, len(images), args.dino_chunk):
                ims, caps = images[i : i + args.dino_chunk], captions[i : i + args.dino_chunk]
                with torch.inference_mode():
                    enc = self.processor(images=ims, text=caps, max_length=200, return_tensors="pt",
                                         padding=True, truncation=True).to(self.device)
                    outputs = self.model(**enc)
                    results += self.processor.post_process_grounded_object_detection(
                        outputs, enc["input_ids"], threshold=box_threshold, text_threshold=text_threshold,
                        target_sizes=[im.size[::-1] for im in ims])
            for r in results:
                if "text_labels" in r:
                    r["labels"] = r["text_labels"]
            return results

    # fastcoref under transformers 5: LingMess (Longformer) has no SDPA kernel, so load it with eager attention
    # (the only kind 4.51 had), and finish its __init__ with post_init(), which v5 needs for its tied-weight
    # bookkeeping (fastcoref calls the pre-v5 init_weights() only). The loaded weights are the checkpoint's.
    from fastcoref.coref_models.modeling_lingmess import LingMessModel
    _lingmess_from_pretrained = LingMessModel.from_pretrained.__func__
    LingMessModel.from_pretrained = classmethod(
        lambda cls, *a, **kw: _lingmess_from_pretrained(cls, *a, **{"attn_implementation": "eager", **kw}))
    _lingmess_init = LingMessModel.__init__

    def _lingmess_init_v5(self, config):
        _lingmess_init(self, config)
        self.post_init()
    LingMessModel.__init__ = _lingmess_init_v5

    tools_hub = os.path.join(args.tools_dir, "hub")  # YoloModel / SpacyModel look in dirname(model_dir)/{yolo,spacy}
    n_gpu = torch.cuda.device_count()
    GVars.args = Namespace(model=args.name, batch_size=args.batch_size, seed=args.seed)
    GVars.save_path = os.path.join(args.run_dir, "results", f"{args.name}.jsonl")
    GVars.model_dir = args.hf_cache
    GVars.main_device = "cuda:0" if n_gpu else "cpu"
    GVars.alter_device = "cuda:1" if n_gpu > 1 else GVars.main_device
    GVars.device, GVars.gpu_count = GVars.main_device, n_gpu
    GVars.logger = logger

    import run.generate_dataset as G
    G.DINO = DINOCompat
    G.YoloModel = lambda name, model_dir=None, **kw: YoloModel(name, model_dir=tools_hub, **kw)
    G.SpacyModel = lambda model_size="md", model_dir=None, **kw: SpacyModel(model_size, model_dir=tools_hub, **kw)
    if flops is not None and args.count_tool_flops:
        G.yolo_detect = flops.wrap("yolo11x", G.yolo_detect)
        G.resolve_corefs = flops.wrap("fastcoref_lingmess", G.resolve_corefs)
        G.b_get_hallu_objects = flops.wrap("grounding_dino_base", G.b_get_hallu_objects)  # DINO is its only torch model
        G.SGParser = type("SGParser", (G.SGParser,), {"pharse": flops.wrap("factual_t5_parser", G.SGParser.pharse)})
    return G


def result_files(run_dir, base, suffix=""):
    """results/<base><suffix>.jsonl and every results/<base>.shard<i><suffix>.jsonl, sorted."""
    import re

    pattern = re.compile(rf"{re.escape(base)}(\.shard\d+)?{re.escape(suffix)}\.jsonl")
    folder = os.path.join(run_dir, "results")
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if pattern.fullmatch(f))


def finished_images(run_dir, base):
    """image_ids saved by any process, unsharded or any shard."""
    return {json.loads(line)["image_id"] for path in result_files(run_dir, base)
            for line in open(path) if line.strip()}


def main():
    args = parse_args()
    base = os.path.basename(args.model.rstrip("/"))
    assert 0 <= args.shard < args.num_shards
    args.name = args.name or (base if args.num_shards == 1 else f"{base}.shard{args.shard}")
    os.makedirs(os.path.join(args.run_dir, "results"), exist_ok=True)
    logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(args.run_dir, "results", f"{args.name}.log"))])
    logger = logging.getLogger("sentinel")
    profile = load_profile(args.flop_profile)
    flops = FlopLedger(os.path.join(args.run_dir, "flops", f"generate_{args.name}.json"), meta=profile_meta(profile))
    G = install_patches(args, logger, flops, profile)
    from model.auxiliary.dataset import DataPoint

    save_path = os.path.join(args.run_dir, "results", f"{args.name}.jsonl")
    done = finished_images(args.run_dir, base)
    rows = [json.loads(line) for line in open(os.path.join(args.run_dir, "data", "images.jsonl"))]
    todo = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard and r["image_id"] not in done]
    if args.num_images >= 0:
        todo = todo[: args.num_images]
    json.dump(vars(args), open(os.path.join(args.run_dir, "results", f"{args.name}_args.json"), "w"), indent=1)
    logger.info(f"{len(rows)} images in manifest, {len(done)} done, {len(todo)} this call -> {save_path}")
    if todo:
        flops.sessions[-1]["images_this_call"] = len(todo)
        run_gen_dataset(G, [DataPoint(r["image_id"], r["image_path"], r["question"]) for r in todo], args.batch_size,
                        args.pair_text)
        flops.count("images_finished", len(todo))
        flops.write()
        logger.info(f"generation FLOPs so far: {flops.total():.4e} -> {flops.path}")


if __name__ == "__main__":
    main()

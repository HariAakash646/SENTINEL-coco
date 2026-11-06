"""One YAML per model as argparse defaults: top-level keys are shared, a section per stage.

Each script calls parse_with_config(parser, "<section>"). Shared keys the script has no flag for are
ignored; section keys must all be known. Explicit CLI flags win over the YAML.
Adapted from OPA-DPO/qwen_coco/config.py.
"""
import yaml


def parse_with_config(parser, section):
    parser.add_argument("--config", help="per-model YAML (see configs/)")
    args, _ = parser.parse_known_args()
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        known = {a.dest for a in parser._actions}
        shared = {k: v for k, v in cfg.items() if not isinstance(v, dict) and k in known}
        own = cfg.get(section) or {}
        unknown = set(own) - known
        assert not unknown, f"{args.config} [{section}]: unknown keys {sorted(unknown)}"
        parser.set_defaults(**{**shared, **own})
    return parser.parse_args()

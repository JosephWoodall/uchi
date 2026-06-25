"""
skill_registry.py
=================
Markdown-based extensible skill system for Uchi.

Skills are .md files with YAML frontmatter that define how a /command
routes through the engine. Drop a file in:

  uchi/skills/       — built-in skills (ship with Uchi)
  ~/.uchi/skills/    — user-installed personal skills

Analytical modes (new)
----------------------
  classify     — TabularPredictor classification
  regress      — TabularRegressor regression
  anomaly      — AnomalyDetector outlier detection
  forecast     — MultivariateTSPredictor step-ahead forecasting
  tsclassify   — TimeSeriesClassifier window classification

Skill file format
-----------------
  ---
  name: search
  description: Search memory and web for information
  args: <query>
  mode: web_search
  prefix: search for
  ---

  Any markdown body here becomes the /help description.

Modes
-----
  chat        — routes through router.chat() with optional prefix prepended
  code        — bypasses chat; forces CodeEngine + REPL oracle path
  web_search  — invokes uchi.web_search; falls back to router.chat
  ingest      — calls cli.ingest_file(router, path)
  memory      — queries AssociativeMemory directly (no trie generation)
  teach       — streams a Q|A pair directly into the trie
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    mode: str = "chat"
    args_hint: str = "<text>"
    prefix: str = ""
    body: str = ""
    source_path: str = ""


def _parse_md(path: str) -> Optional[Skill]:
    """Parse a skill .md file into a Skill dataclass. Returns None on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return None

    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return None

    front, body = m.group(1), m.group(2).strip()
    meta: Dict[str, str] = {}
    for line in front.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()

    name = meta.get("name", "")
    if not name:
        return None

    return Skill(
        name=name.lower(),
        description=meta.get("description", ""),
        mode=meta.get("mode", "chat"),
        args_hint=meta.get("args", "<text>"),
        prefix=meta.get("prefix", ""),
        body=body,
        source_path=path,
    )


def _builtin_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "skills")


def _user_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".uchi", "skills")


class SkillRegistry:
    """
    Discovers and dispatches markdown-defined skills.

    Load order (later overrides earlier):
      1. uchi/skills/      — built-in skills
      2. ~/.uchi/skills/   — user-installed skills
    """

    def __init__(self, router):
        self.router = router
        self._skills: Dict[str, Skill] = {}
        self._intent_encoder = None
        self._reload()
        self._init_intent_encoder()

    def reload(self):
        """Re-scan skill directories — picks up new files without restart."""
        self._reload()
        self._init_intent_encoder()

    def match_intent(self, query_tokens: List[str], trie_dist: Optional[dict] = None):
        """
        Use the LatentIntentEncoder to find the best skill for a natural-language query.
        Returns (skill_name, confidence) or (None, 0.0).
        """
        if self._intent_encoder is None or not self._intent_encoder.is_ready():
            return None, 0.0
        return self._intent_encoder.match(query_tokens, trie_dist)

    def dispatch(self, name: str, args: str, callback=None) -> str:
        """Execute skill by name. Returns the response string."""
        skill = self._skills.get(name.lower())
        if skill is None:
            available = "  ".join(f"/{s}" for s in sorted(self._skills))
            return f"Unknown skill '/{name}'.\nAvailable: {available}"
        return self._execute(skill, args.strip(), callback)

    def list_skills(self) -> List[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def has(self, name: str) -> bool:
        return name.lower() in self._skills

    def get_all_vectors(self) -> dict:
        """Return all registered skill vectors for unified vector routing."""
        if self._intent_encoder is None:
            return {}
        return dict(self._intent_encoder._skill_vectors)

    def help_text(self) -> str:
        """Build the /help skills section."""
        lines = []
        for skill in self.list_skills():
            lines.append(f"  /{skill.name:<12} {skill.args_hint:<18} {skill.description}")
        return "\n".join(lines)

    # ── internal ─────────────────────────────────────────────────────────────

    def _init_intent_encoder(self):
        """Build (or rebuild) skill vectors in the LatentIntentEncoder."""
        try:
            from uchi.neuro_symbolic import get_ssm
            from uchi.intent_encoder import LatentIntentEncoder
            ssm = get_ssm()
            enc = LatentIntentEncoder(ssm)
            for skill in self._skills.values():
                tokens = (skill.name + " " + skill.description).lower().split()
                enc.register_skill(skill.name, tokens)
            self._intent_encoder = enc
        except Exception:
            self._intent_encoder = None

    def _reload(self):
        self._skills.clear()
        self._load_dir(_builtin_dir())
        self._load_dir(_user_dir())

    def _load_dir(self, path: str):
        if not path or not os.path.isdir(path):
            return
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".md"):
                skill = _parse_md(os.path.join(path, fname))
                if skill:
                    self._skills[skill.name] = skill

    def _execute(self, skill: Skill, args: str, callback) -> str:
        message = f"{skill.prefix} {args}".strip() if skill.prefix else args

        if skill.mode == "chat":
            return self.router.chat(message, callback=callback)

        elif skill.mode == "code":
            tokens = message.split()
            concepts = self.router.tokenizer.tokenize(tokens, is_inference=True)
            return self.router._handle_code_intent(message, tokens, concepts, callback)

        elif skill.mode == "web_search":
            try:
                from uchi.web_search import search as _search
                result = _search(args)
                if result:
                    self.router.stream(result.split())
                    return result
            except Exception:
                pass
            return self.router.chat(f"search for information about {args}", callback=callback)

        elif skill.mode == "ingest":
            from uchi.cli import ingest_file, preload_context
            path = args
            if not os.path.exists(path):
                return f"Path not found: {path}"
            if os.path.isdir(path):
                preload_context(self.router, path)
            else:
                ingest_file(self.router, path)
            return f"Ingested: {path}"

        elif skill.mode == "memory":
            tokens = args.split()
            result = self.router.query(tokens)
            if result == "[Unknown Context]":
                return "Nothing found in memory for that query."
            return result

        elif skill.mode == "teach":
            # Format: /teach <question> | <answer>
            if "|" not in args:
                return (
                    "Format: /teach <question> | <answer>\n"
                    "Example: /teach what is gravity | the force of attraction between masses"
                )
            question, _, answer = args.partition("|")
            question, answer = question.strip(), answer.strip()
            if not question or not answer:
                return "Both question and answer are required."
            seq = (
                ["<|user|>"] + question.split()
                + ["<|assistant|>"] + answer.split()
                + ["<|end|>"]
            )
            self.router.stream(seq)
            return f"Learned: '{question}' → '{answer}'"

        elif skill.mode == "overview":
            return _knowledge_overview(self.router)

        elif skill.mode in ("classify", "regress", "anomaly", "forecast", "tsclassify"):
            return _run_analytical(skill.mode, args, self.router)

        # Fallback
        return self.router.chat(message, callback=callback)


def _run_analytical(mode: str, args: str, router) -> str:
    """
    Execute an analytical ML skill.

    Loads data, instantiates the correct ML class, fits + predicts, and
    streams a text summary back into the trie so future queries can recall it.
    """
    from uchi.data_loader import parse_args, load_data, split_features, to_numeric_rows, train_test_split

    parsed = parse_args(args)
    path   = parsed["path"]
    label  = parsed.get("label") or parsed.get("target")
    steps  = parsed.get("steps", 10)

    if not path:
        return (
            f"Usage: /{mode} <path.csv> "
            + ("" if mode == "anomaly" else "[--label <col>]")
            + ("\n       e.g. /{} data.csv".format(mode))
        )

    if not __import__("os").path.exists(path):
        return f"File not found: {path}"

    try:
        header, rows = load_data(path)
    except Exception as exc:
        return f"Failed to load '{path}': {exc}"

    if not rows:
        return f"No data rows found in '{path}'."

    try:
        if mode == "classify":
            result = _skill_classify(header, rows, label)
        elif mode == "regress":
            result = _skill_regress(header, rows, label)
        elif mode == "anomaly":
            result = _skill_anomaly(header, rows, path)
        elif mode == "forecast":
            result = _skill_forecast(header, rows, steps, path)
        elif mode == "tsclassify":
            result = _skill_tsclassify(header, rows, label)
        else:
            result = "Unknown analytical mode."
    except Exception as exc:
        import traceback
        return f"Analysis error: {exc}\n{traceback.format_exc(limit=3)}"

    # Stream the summary into the trie so it can be recalled later
    try:
        summary_tokens = result.replace("\n", " ").split()[:40]
        router.stream(summary_tokens)
    except Exception:
        pass

    return result


def _skill_classify(header, rows, label_col) -> str:
    from uchi.data_loader import split_features, train_test_split
    from uchi.tabular import TabularPredictor

    X, y = split_features(header, rows, label_col)
    if len(X) < 4:
        return f"Not enough rows for classification (need ≥ 4, got {len(X)})."

    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    clf = TabularPredictor()
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    classes = list(clf.classes_)
    used_label = header[-1] if not label_col else label_col

    lines = [
        f"Classification complete.",
        f"  Label column : {used_label}",
        f"  Classes      : {classes[:8]}{'…' if len(classes) > 8 else ''}",
        f"  Train rows   : {len(X_tr)}",
        f"  Test rows    : {len(X_te)}",
        f"  Accuracy     : {acc:.1%}",
    ]
    return "\n".join(lines)


def _skill_regress(header, rows, label_col) -> str:
    from uchi.data_loader import split_features, train_test_split
    from uchi.tabular import TabularRegressor

    X, y_raw = split_features(header, rows, label_col)
    if len(X) < 4:
        return f"Not enough rows for regression (need ≥ 4, got {len(X)})."

    try:
        y = [float(v) for v in y_raw]
    except ValueError:
        return "Target column contains non-numeric values. Use /classify for categorical targets."

    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    reg = TabularRegressor()
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    mae = sum(abs(p - t) for p, t in zip(preds, y_te)) / max(len(y_te), 1)
    used_label = header[-1] if not label_col else label_col

    lines = [
        f"Regression complete.",
        f"  Target column : {used_label}",
        f"  Train rows    : {len(X_tr)}",
        f"  Test rows     : {len(X_te)}",
        f"  MAE           : {mae:.4f}",
        f"  Target range  : [{min(y_te):.3f}, {max(y_te):.3f}]",
    ]
    return "\n".join(lines)


def _skill_anomaly(header, rows, path) -> str:
    from uchi.data_loader import to_numeric_rows
    from uchi.timeseries import AnomalyDetector

    X = to_numeric_rows(header, rows)
    if len(X) < 4:
        return f"Not enough numeric rows for anomaly detection (need ≥ 4, got {len(X)})."

    det = AnomalyDetector()
    det.fit(X)
    labels = det.predict(X)
    scores = det.score_samples(X)
    anomalous = [i for i, l in enumerate(labels) if l == 1]
    n = len(anomalous)
    top = sorted(anomalous, key=lambda i: scores[i], reverse=True)[:5]

    lines = [
        f"Anomaly detection complete.",
        f"  File       : {path}",
        f"  Total rows : {len(X)}",
        f"  Anomalies  : {n}  ({100*n/len(X):.1f}%)",
    ]
    if top:
        lines.append(f"  Top anomalous row indices: {top}")
    else:
        lines.append("  No anomalies detected (all rows within 2σ).")
    return "\n".join(lines)


def _skill_forecast(header, rows, steps, path) -> str:
    from uchi.data_loader import to_numeric_rows
    from uchi.timeseries import MultivariateTSPredictor

    X = to_numeric_rows(header, rows)
    if len(X) < 4:
        return f"Not enough rows for forecasting (need ≥ 4, got {len(X)})."

    pred = MultivariateTSPredictor()
    pred.fit(X)
    forecast = pred.forecast(steps)

    dims = len(header)
    last = X[-1] if X else []
    lines = [
        f"Forecast complete.",
        f"  File       : {path}",
        f"  Dimensions : {dims}",
        f"  History    : {len(X)} steps",
        f"  Forecasted : {steps} steps ahead",
        f"  Last obs   : [{', '.join(f'{v:.3f}' for v in last[:6])}{'…' if len(last)>6 else ''}]",
    ]
    for i, step in enumerate(forecast[:5]):
        vals = [f"{v:.3f}" for v in step[:6]]
        suffix = "…" if len(step) > 6 else ""
        lines.append(f"  t+{i+1:<3}      : [{', '.join(vals)}{suffix}]")
    if steps > 5:
        lines.append(f"  … ({steps - 5} more steps not shown)")
    return "\n".join(lines)


def _skill_tsclassify(header, rows, label_col) -> str:
    from uchi.data_loader import split_features, train_test_split
    from uchi.timeseries import TimeSeriesClassifier

    X_raw, y = split_features(header, rows, label_col)
    if len(X_raw) < 4:
        return f"Not enough windows for classification (need ≥ 4, got {len(X_raw)})."

    # Each X row is a flat window — wrap each scalar in a list for the TSC API
    X_windows = [[[v] for v in window] for window in X_raw]

    X_tr, X_te, y_tr, y_te = train_test_split(X_windows, y)
    clf = TimeSeriesClassifier()
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    used_label = header[-1] if not label_col else label_col

    lines = [
        f"Time series classification complete.",
        f"  Label column : {used_label}",
        f"  Window length: {len(X_raw[0]) if X_raw else 0} features",
        f"  Train windows: {len(X_tr)}",
        f"  Test windows : {len(X_te)}",
        f"  Accuracy     : {acc:.1%}",
    ]
    return "\n".join(lines)



def _knowledge_overview(router) -> str:
    import re

    _STOP = {
        "you", "i", "is", "a", "the", "to", "can", "that", "my", "am", "it",
        "are", "we", "me", "in", "of", "and", "or", "not", "be", "do", "if",
        "for", "at", "but", "this", "so", "with", "from", "there", "how",
        "who", "what", "no", "yes", "have", "your", "their", "our", "by",
        "an", "as", "on", "up", "will", "<|user|>", "<|assistant|>", "<|end|>",
        "see.n.01", "later.s.01",
    }

    _CLUSTERS = [
        ("python / code",   {"def", "return", "class", "function", "trie", "sequence",
                              "predict", "python", "code", "import", "```python",
                              "learn", "train", "parameter", "operation", "optimum"}),
        ("conversational",  {"hello", "thank", "appreciate", "good", "goodbye", "today",
                              "aid", "correct", "sorry", "bye", "farewell", "adieu",
                              "morning", "problem", "joke", "cool"}),
        ("self-knowledge",  {"deterministic", "concept", "uchi", "forecaster", "asset",
                              "capability", "sequence", "predict"}),
        ("math / numbers",  {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                              "square", "root", "sum", "mean"}),
    ]

    def _clean(tok: str) -> str:
        return re.sub(r"\.[a-z]+\.\d+$", "", tok)

    inner   = router.predictor._pred
    n_nodes = len(inner._nodes)
    n_vocab = len(inner._vocab)

    co = router.memory.co_occurrence
    totals = {
        k: sum(v.values())
        for k, v in co.items()
        if k not in _STOP and not k.startswith("<|")
    }
    ranked = sorted(totals.items(), key=lambda x: -x[1])

    placed: dict[str, list] = {name: [] for name, _ in _CLUSTERS}
    placed["other"] = []
    for tok, _ in ranked:
        clean = _clean(tok)
        matched = False
        for name, keywords in _CLUSTERS:
            if clean in keywords or tok in keywords:
                placed[name].append(clean)
                matched = True
                break
        if not matched:
            placed["other"].append(clean)

    n_episodic = len(router.memory.cpu_mem.records)
    proc_keys  = list(getattr(router.procedural, "_store", {}).keys())

    lines = [
        "Knowledge Overview",
        "==================",
        f"  Trie          : {n_nodes:,} nodes · {n_vocab} concepts",
        f"  Episodic mem  : {n_episodic:,} stored associations",
        f"  Procedural    : {', '.join(proc_keys) if proc_keys else 'none'}",
        "",
        "Top concept clusters:",
    ]
    for name, _ in _CLUSTERS:
        tokens = placed[name][:8]
        if tokens:
            lines.append(f"  {name:<20} {', '.join(tokens)}")
    if placed["other"]:
        lines.append(f"  {'other':<20} {', '.join(placed['other'][:8])}")

    return "\n".join(lines)

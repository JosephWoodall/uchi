"""
skill_registry.py
=================
Markdown-based extensible skill system for Uchi.

Skills are .md files with YAML frontmatter that define how a /command
routes through the engine. Drop a file in:

  uchi/skills/       — built-in skills (ship with Uchi)
  ~/.uchi/skills/    — user-installed personal skills

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
  dream       — spawns one offline dreaming cycle in background
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
        self._reload()

    def reload(self):
        """Re-scan skill directories — picks up new files without restart."""
        self._reload()

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

    def help_text(self) -> str:
        """Build the /help skills section."""
        lines = []
        for skill in self.list_skills():
            lines.append(f"  /{skill.name:<12} {skill.args_hint:<18} {skill.description}")
        return "\n".join(lines)

    # ── internal ─────────────────────────────────────────────────────────────

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

        elif skill.mode == "dream":
            return _fire_dream(self.router)

        # Fallback
        return self.router.chat(message, callback=callback)


def _fire_dream(router) -> str:
    import sys, subprocess, os
    script = os.path.normpath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "offline_dreaming.py")
    )
    if not os.path.exists(script):
        return "Offline dreaming script not found."
    try:
        subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "Offline dreaming cycle started in background."
    except Exception as e:
        return f"Dream failed to start: {e}"

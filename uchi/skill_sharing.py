"""skill_sharing.py — Weaponize Procedural Memory: ``.uchi_skill`` export/import
(0.4.0 Item 16.3).

Skills are already markdown files with YAML frontmatter
(``uchi/skill_registry.py``) — a ``.uchi_skill`` file is exactly that same
format under a friendlier, shareable extension. No new file format, no
new parser: export copies a skill's existing ``.md`` content to a
``.uchi_skill`` file; import validates a ``.uchi_skill`` file with the
*same* frontmatter parser already used to load built-in skills, then
drops it into the target skills directory so ``SkillRegistry.reload()``
picks it up. This is how a community shares verified Procedural-Memory
tools without inventing any new packaging.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

from .skill_registry import Skill, _builtin_dir, _parse_md, _user_dir


def _find_skill_file(name: str) -> Optional[str]:
    name = name.lower()
    for directory in (_user_dir(), _builtin_dir()):
        if not directory or not os.path.isdir(directory):
            continue
        candidate = os.path.join(directory, f"{name}.md")
        if os.path.exists(candidate):
            return candidate
    return None


def export_skill(name: str, out_dir: str = ".") -> str:
    """Export skill *name* as a shareable ``<name>.uchi_skill`` file in
    *out_dir*. Raises ``FileNotFoundError`` if no such skill exists."""
    src = _find_skill_file(name)
    if src is None:
        raise FileNotFoundError(f"no skill named {name!r} found")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{name.lower()}.uchi_skill")
    shutil.copyfile(src, dest)
    return dest


def import_skill(path: str, skills_dir: Optional[str] = None) -> Skill:
    """Import a ``.uchi_skill`` file, validating it with the same
    frontmatter parser used for built-in skills, and install it into
    *skills_dir* (defaults to the user skills dir, ``~/.uchi/skills``) so
    a subsequent ``SkillRegistry.reload()`` picks it up.
    """
    skill = _parse_md(path)
    if skill is None:
        raise ValueError(f"{path!r} is not a valid Uchi skill file (missing/invalid frontmatter)")

    target_dir = skills_dir or _user_dir()
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, f"{skill.name}.md")
    shutil.copyfile(path, dest)
    return skill

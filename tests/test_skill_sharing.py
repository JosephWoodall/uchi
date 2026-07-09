import os

import pytest

from uchi.skill_sharing import export_skill, import_skill

SAMPLE_SKILL_MD = """---
name: mytool
description: Does a thing
mode: chat
args: <text>
---

Body text here.
"""


def test_export_skill_creates_uchi_skill_file(tmp_path, monkeypatch):
    skills_dir = tmp_path / "builtin_skills"
    skills_dir.mkdir()
    (skills_dir / "mytool.md").write_text(SAMPLE_SKILL_MD)

    monkeypatch.setattr("uchi.skill_sharing._builtin_dir", lambda: str(skills_dir))
    monkeypatch.setattr("uchi.skill_sharing._user_dir", lambda: str(tmp_path / "no_user_dir"))

    out_dir = str(tmp_path / "export_out")
    path = export_skill("mytool", out_dir=out_dir)
    assert path.endswith("mytool.uchi_skill")
    assert os.path.exists(path)
    assert "name: mytool" in open(path).read()


def test_export_skill_raises_for_unknown_skill(tmp_path, monkeypatch):
    monkeypatch.setattr("uchi.skill_sharing._builtin_dir", lambda: str(tmp_path / "empty"))
    monkeypatch.setattr("uchi.skill_sharing._user_dir", lambda: str(tmp_path / "also_empty"))
    with pytest.raises(FileNotFoundError):
        export_skill("nonexistent")


def test_import_skill_installs_into_target_dir(tmp_path):
    shareable = tmp_path / "mytool.uchi_skill"
    shareable.write_text(SAMPLE_SKILL_MD)

    target_dir = str(tmp_path / "installed_skills")
    skill = import_skill(str(shareable), skills_dir=target_dir)

    assert skill.name == "mytool"
    assert skill.description == "Does a thing"
    installed_path = os.path.join(target_dir, "mytool.md")
    assert os.path.exists(installed_path)
    assert open(installed_path).read() == SAMPLE_SKILL_MD


def test_import_skill_rejects_invalid_file(tmp_path):
    bad = tmp_path / "bad.uchi_skill"
    bad.write_text("not a valid skill file, no frontmatter")
    with pytest.raises(ValueError):
        import_skill(str(bad))


def test_import_skill_is_actually_loadable_by_skill_registry(tmp_path):
    """The whole point: an imported .uchi_skill must be indistinguishable
    from a built-in skill once installed -- SkillRegistry's own parser
    must accept it without modification."""
    from uchi.skill_registry import _parse_md

    shareable = tmp_path / "mytool.uchi_skill"
    shareable.write_text(SAMPLE_SKILL_MD)
    target_dir = str(tmp_path / "installed_skills")
    import_skill(str(shareable), skills_dir=target_dir)

    reparsed = _parse_md(os.path.join(target_dir, "mytool.md"))
    assert reparsed is not None
    assert reparsed.name == "mytool"

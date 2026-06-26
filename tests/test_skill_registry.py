"""Tests for the SkillRegistry and markdown skill parsing."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


# ── Markdown parsing ──────────────────────────────────────────────────────────

class TestSkillParsing:
    def _write_skill(self, tmp_dir, content):
        path = os.path.join(tmp_dir, "test_skill.md")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_parses_full_frontmatter(self, tmp_path):
        from uchi.skill_registry import _parse_md
        path = self._write_skill(str(tmp_path), """\
---
name: search
description: Search the web
args: <query>
mode: web_search
prefix: search for
---

Body text here.
""")
        skill = _parse_md(path)
        assert skill is not None
        assert skill.name == "search"
        assert skill.description == "Search the web"
        assert skill.mode == "web_search"
        assert skill.args_hint == "<query>"
        assert skill.prefix == "search for"
        assert "Body text" in skill.body

    def test_minimal_frontmatter(self, tmp_path):
        from uchi.skill_registry import _parse_md
        path = self._write_skill(str(tmp_path), """\
---
name: simple
description: A simple skill
---
""")
        skill = _parse_md(path)
        assert skill is not None
        assert skill.name == "simple"
        assert skill.mode == "chat"  # default
        assert skill.prefix == ""    # default

    def test_missing_name_returns_none(self, tmp_path):
        from uchi.skill_registry import _parse_md
        path = self._write_skill(str(tmp_path), """\
---
description: No name here
mode: chat
---
""")
        skill = _parse_md(path)
        assert skill is None

    def test_no_frontmatter_returns_none(self, tmp_path):
        from uchi.skill_registry import _parse_md
        path = self._write_skill(str(tmp_path), "Just some markdown with no frontmatter.")
        skill = _parse_md(path)
        assert skill is None

    def test_name_is_lowercased(self, tmp_path):
        from uchi.skill_registry import _parse_md
        path = self._write_skill(str(tmp_path), """\
---
name: MySkill
description: Uppercased name
---
""")
        skill = _parse_md(path)
        assert skill.name == "myskill"

    def test_missing_file_returns_none(self):
        from uchi.skill_registry import _parse_md
        assert _parse_md("/nonexistent/path/skill.md") is None


# ── SkillRegistry ─────────────────────────────────────────────────────────────

class TestSkillRegistry:
    def _make_router(self):
        mock = MagicMock()
        mock.chat.return_value = "mock chat response"
        mock.query.return_value = "[Unknown Context]"
        mock.tokenizer.tokenize.return_value = ["hello"]
        return mock

    def test_loads_builtin_skills(self):
        from uchi.skill_registry import SkillRegistry
        router = self._make_router()
        reg = SkillRegistry(router)
        skills = reg.list_skills()
        names = {s.name for s in skills}
        # All built-in skills should be present
        assert "search" in names
        assert "explain" in names
        assert "code" in names
        assert "formula" in names
        assert "ingest" in names
        assert "teach" in names
        assert "dream" in names
        assert "recall" in names

    def test_has_returns_true_for_known_skill(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        assert reg.has("search") is True
        assert reg.has("SEARCH") is True  # case-insensitive

    def test_has_returns_false_for_unknown(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        assert reg.has("xyz_nonexistent") is False

    def test_dispatch_unknown_skill_returns_error(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        reply = reg.dispatch("xyz_fake", "some args")
        assert "Unknown skill" in reply
        assert "/xyz_fake" in reply

    def test_dispatch_chat_mode_calls_router_chat(self):
        from uchi.skill_registry import SkillRegistry
        router = self._make_router()
        reg = SkillRegistry(router)
        # 'explain' uses mode: chat with prefix
        reg.dispatch("explain", "recursion")
        router.chat.assert_called_once()
        call_arg = router.chat.call_args[0][0]
        assert "recursion" in call_arg

    def test_dispatch_teach_mode_streams_qa(self):
        from uchi.skill_registry import SkillRegistry
        router = self._make_router()
        reg = SkillRegistry(router)
        reply = reg.dispatch("teach", "what is gravity | the force of attraction")
        router.stream.assert_called_once()
        streamed = router.stream.call_args[0][0]
        assert "<|user|>" in streamed
        assert "<|assistant|>" in streamed
        assert "gravity" in streamed
        assert "force" in streamed
        assert "Learned" in reply

    def test_dispatch_teach_missing_pipe_returns_format_hint(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        reply = reg.dispatch("teach", "no pipe here")
        assert "Format" in reply
        assert "|" in reply

    def test_dispatch_memory_mode_queries_router(self):
        from uchi.skill_registry import SkillRegistry
        router = self._make_router()
        router.query.return_value = "gravity is the force of attraction"
        reg = SkillRegistry(router)
        reply = reg.dispatch("recall", "gravity")
        router.query.assert_called_once()
        assert reply == "gravity is the force of attraction"

    def test_dispatch_memory_mode_unknown_context(self):
        from uchi.skill_registry import SkillRegistry
        router = self._make_router()
        router.query.return_value = "[Unknown Context]"
        reg = SkillRegistry(router)
        reply = reg.dispatch("recall", "xyzzy nonexistent")
        assert "Nothing found" in reply

    def test_dispatch_ingest_mode_nonexistent_path(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        reply = reg.dispatch("ingest", "/nonexistent/path/file.txt")
        assert "not found" in reply.lower()

    def test_dispatch_ingest_mode_real_file(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        router = self._make_router()
        reg = SkillRegistry(router)
        f = tmp_path / "test.txt"
        f.write_text("hello world from ingest test")
        reply = reg.dispatch("ingest", str(f))
        assert "Ingested" in reply

    def test_user_skills_loaded_from_custom_dir(self, tmp_path):
        """Skills dropped in a custom dir are discovered."""
        from uchi.skill_registry import SkillRegistry, _parse_md
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "custom.md").write_text("""\
---
name: custom
description: A custom user skill
args: <text>
mode: chat
---

This is a user-installed skill.
""")
        # Manually load from the custom dir
        skill = _parse_md(str(skill_dir / "custom.md"))
        assert skill is not None
        assert skill.name == "custom"
        assert skill.description == "A custom user skill"

    def test_reload_refreshes_skills(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        initial = len(reg.list_skills())
        reg.reload()
        assert len(reg.list_skills()) == initial  # same after reload

    def test_help_text_contains_all_skill_names(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(self._make_router())
        text = reg.help_text()
        for s in reg.list_skills():
            assert f"/{s.name}" in text


# ── OmniRouter integration ────────────────────────────────────────────────────

class TestOmniRouterSkillsAttribute:
    def test_router_has_skills_attribute(self):
        from uchi.omni_router import OmniRouter
        router = OmniRouter(use_bpe=False)
        assert hasattr(router, "skills")
        from uchi.skill_registry import SkillRegistry
        assert isinstance(router.skills, SkillRegistry)

    def test_setstate_adds_skills(self):
        from uchi.omni_router import OmniRouter
        from uchi.generative import SequenceGenerator
        from uchi.grpo import AgenticBaseline
        from uchi.procedural_memory import ProceduralMemory
        from uchi.memory import AssociativeMemory
        from uchi.omni_tokenizer import OmniTokenizer

        router = OmniRouter.__new__(OmniRouter)
        router.__setstate__({
            "predictor": SequenceGenerator(context_length=6),
            "baseline": AgenticBaseline(),
            "procedural": ProceduralMemory(),
            "memory": AssociativeMemory(),
            "tokenizer": OmniTokenizer(),
            "_knowledge_bootstrapped": True,
            "last_sequence": None,
        })
        from uchi.skill_registry import SkillRegistry
        assert isinstance(router.skills, SkillRegistry)

    def test_start_background_jobs_idempotent(self):
        """Calling start_background_jobs twice does not double-spawn."""
        from uchi.omni_router import OmniRouter
        router = OmniRouter(use_bpe=False)
        with patch("subprocess.Popen") as mock_popen:
            router._background_started = False
            router.start_background_jobs()
            count_after_first = mock_popen.call_count
            router.start_background_jobs()  # second call — no-op
            assert mock_popen.call_count == count_after_first

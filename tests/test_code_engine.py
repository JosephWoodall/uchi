"""Tests for CodeEngine, REPLOracle, ASTCodeTokenizer, and SpecialistPool."""

import pytest
from unittest.mock import MagicMock, patch


# ── ASTCodeTokenizer ─────────────────────────────────────────────────────────

class TestASTCodeTokenizer:
    def setup_method(self):
        from uchi.code_tokenizer import ASTCodeTokenizer
        self.tok = ASTCodeTokenizer()

    def test_simple_function(self):
        src = "def add(a, b):\n    return a + b"
        tokens = self.tok.tokenize_source(src)
        assert "DEF" in tokens
        assert "FNAME:add" in tokens
        assert "ARG:a" in tokens
        assert "RETURN" in tokens

    def test_class_definition(self):
        src = "class Foo:\n    def bar(self):\n        pass"
        tokens = self.tok.tokenize_source(src)
        assert "CLASS:Foo" in tokens
        assert "DEF" in tokens
        assert "PASS" in tokens

    def test_if_else(self):
        src = "if x > 0:\n    return x\nelse:\n    return -x"
        tokens = self.tok.tokenize_source(src)
        assert "IF" in tokens
        assert "RETURN" in tokens

    def test_for_loop(self):
        src = "for i in range(10):\n    print(i)"
        tokens = self.tok.tokenize_source(src)
        assert "FOR" in tokens
        assert "CALL:print" in tokens

    def test_import(self):
        src = "import os\nimport sys"
        tokens = self.tok.tokenize_source(src)
        assert "IMPORT:os" in tokens
        assert "IMPORT:sys" in tokens

    def test_invalid_syntax_fallback(self):
        src = "def broken("
        tokens = self.tok.tokenize_source(src)
        assert isinstance(tokens, list)
        # Falls back to word split — should not raise
        assert len(tokens) > 0

    def test_tokenize_query(self):
        words = ["write", "a", "function", "that", "returns", "list"]
        tokens = self.tok.tokenize_query(words)
        assert "DEF" in tokens
        assert "RETURN" in tokens
        assert "LIST" in tokens

    def test_list_comprehension(self):
        src = "result = [x * 2 for x in items]"
        tokens = self.tok.tokenize_source(src)
        assert "ASSIGN" in tokens
        assert "LISTCOMP" in tokens

    def test_try_except(self):
        src = "try:\n    x = int(s)\nexcept ValueError:\n    x = 0"
        tokens = self.tok.tokenize_source(src)
        assert "TRY" in tokens
        assert "EXCEPT" in tokens


# ── REPLOracle ────────────────────────────────────────────────────────────────

class TestREPLOracle:
    def setup_method(self):
        from uchi.code_engine import REPLOracle
        self.oracle = REPLOracle()

    def test_valid_code_passes(self):
        code = "def add(a, b):\n    return a + b\n"
        passed, reward = self.oracle.verify(code)
        assert passed is True
        assert reward > 0

    def test_syntax_error_fails(self):
        code = "def broken("
        passed, reward = self.oracle.verify(code)
        assert passed is False
        assert reward < 0

    def test_empty_function_passes(self):
        code = "def noop():\n    pass\n"
        passed, reward = self.oracle.verify(code)
        assert passed is True

    def test_import_code_passes(self):
        code = "import os\nresult = os.path.join('a', 'b')\n"
        passed, reward = self.oracle.verify(code)
        assert passed is True


# ── CodeEngine ────────────────────────────────────────────────────────────────

class TestCodeEngine:
    def _make_mock_predictor(self, output_tokens=None):
        mock = MagicMock()
        if output_tokens is None:
            output_tokens = ["def", "noop", "(", ")", ":", "pass"]
        mock.generate.return_value = output_tokens
        return mock

    def test_generate_code_returns_tuple(self):
        from uchi.code_engine import CodeEngine
        predictor = self._make_mock_predictor()
        engine = CodeEngine(predictor, n_workers=1)
        seed = ["<|user|>", "write", "a", "function", "<|assistant|>"]
        result = engine.generate_code(seed, max_tokens=20)
        assert len(result) == 3
        code_str, reward, passed = result
        assert isinstance(code_str, str)
        assert isinstance(reward, float)
        assert isinstance(passed, bool)

    def test_hole_synthesis_when_empty(self):
        from uchi.code_engine import CodeEngine
        predictor = self._make_mock_predictor(output_tokens=[])
        engine = CodeEngine(predictor, n_workers=1)
        seed = ["<|user|>", "fibonacci", "<|assistant|>"]
        code_str, reward, passed = engine.generate_code(seed)
        assert passed is False

    def test_has_holes_detection(self):
        from uchi.code_engine import CodeEngine
        code = "def f():\n    # ??HOLE:fibonacci logic??\n    pass"
        assert CodeEngine.has_holes(code) is True
        assert CodeEngine.has_holes("def f():\n    return 1\n") is False

    def test_extract_holes(self):
        from uchi.code_engine import CodeEngine
        code = "def f(x):\n    # ??HOLE:base case??\n    # ??HOLE:recursive case??\n    pass"
        holes = CodeEngine.extract_holes(code)
        assert len(holes) == 2
        assert "base case" in holes
        assert "recursive case" in holes

    def test_fill_hole(self):
        from uchi.code_engine import CodeEngine
        code = "def f(x):\n    # ??HOLE:the logic??\n    pass"
        filled = CodeEngine.fill_hole(code, "the logic", "return x * 2")
        assert "??HOLE:" not in filled
        assert "return x * 2" in filled

    def test_repl_oracle_validates_good_code(self):
        from uchi.code_engine import CodeEngine, REPLOracle
        good_code = "def add(a, b):\n    return a + b\n"
        mock_pred = self._make_mock_predictor(["def", "add", "(", "a", ",", "b", ")", ":", "return", "a"])
        engine = CodeEngine(mock_pred, n_workers=1)
        passed, reward = engine.oracle.verify(good_code)
        assert passed


# ── SpecialistPool ────────────────────────────────────────────────────────────

class TestSpecialistPool:
    def _make_mock_router(self):
        mock = MagicMock()
        mock.predictor = MagicMock()
        mock.chat.return_value = "mock response"
        return mock

    def test_falls_back_to_default(self):
        from uchi.specialist_pool import SpecialistPool
        default = self._make_mock_router()
        with patch("uchi.specialist_pool.os.path.exists", return_value=False):
            pool = SpecialistPool(default)
            result = pool.route("code")
        assert result is default

    def test_has_specialist_false_when_no_brain(self):
        from uchi.specialist_pool import SpecialistPool
        with patch("uchi.specialist_pool.os.path.exists", return_value=False):
            pool = SpecialistPool(self._make_mock_router())
            assert pool.has_specialist("code") is False
            assert pool.has_specialist("math") is False

    def test_get_predictor_returns_default_predictor(self):
        from uchi.specialist_pool import SpecialistPool
        default = self._make_mock_router()
        with patch("uchi.specialist_pool.os.path.exists", return_value=False):
            pool = SpecialistPool(default)
            pred = pool.get_predictor("code")
        assert pred is default.predictor

    def test_intent_mapping_unknown_falls_to_convo(self):
        from uchi.specialist_pool import _INTENT_TO_SPECIALIST
        assert _INTENT_TO_SPECIALIST.get("code") == "code"
        assert _INTENT_TO_SPECIALIST.get("physics") == "math"
        assert _INTENT_TO_SPECIALIST.get("search") == "convo"

    def test_route_unknown_intent_returns_default(self):
        from uchi.specialist_pool import SpecialistPool
        default = self._make_mock_router()
        with patch("uchi.specialist_pool.os.path.exists", return_value=False):
            pool = SpecialistPool(default)
            result = pool.route("unknown_intent_xyz")
        assert result is default


# ── OmniRouter integration ────────────────────────────────────────────────────

class TestOmniRouterIntegration:
    """Verify new attributes and helper methods are present after unpickling."""

    def test_omni_router_has_code_engine(self, tmp_path):
        from uchi.omni_router import OmniRouter
        router = OmniRouter(use_bpe=False)
        assert hasattr(router, 'code_engine')
        assert hasattr(router, 'specialist_pool')

    def test_setstate_adds_code_engine(self):
        from uchi.omni_router import OmniRouter
        from uchi.generative import SequenceGenerator
        from uchi.grpo import AgenticBaseline
        from uchi.procedural_memory import ProceduralMemory
        from uchi.memory import AssociativeMemory
        from uchi.omni_tokenizer import OmniTokenizer

        router = OmniRouter.__new__(OmniRouter)
        # Simulate a minimal pickled state that lacks the new attributes
        mock_state = {
            'predictor': SequenceGenerator(context_length=6),
            'baseline': AgenticBaseline(),
            'procedural': ProceduralMemory(),
            'memory': AssociativeMemory(),
            'tokenizer': OmniTokenizer(),
            '_knowledge_bootstrapped': True,
            'last_sequence': None,
        }
        router.__setstate__(mock_state)
        assert hasattr(router, 'code_engine')
        assert hasattr(router, 'specialist_pool')
        assert hasattr(router, 'ssm_optimizer')

    def test_get_intent_key(self):
        from uchi.procedural_memory import ProceduralMemory
        pm = ProceduralMemory()
        assert pm.get_intent_key("write a python function") == "code"
        assert pm.get_intent_key("calculate the force") in ("math", "physics")
        assert pm.get_intent_key("xyzzy unknown query") is None

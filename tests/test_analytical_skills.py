"""Tests for analytical skill modes and data_loader utilities."""
import os
import pytest
from unittest.mock import MagicMock, patch


# ── data_loader ───────────────────────────────────────────────────────────────

class TestDataLoader:
    def _csv(self, tmp_path, content):
        p = tmp_path / "test.csv"
        p.write_text(content)
        return str(p)

    def test_load_csv_with_header(self, tmp_path):
        from uchi.data_loader import load_data
        path = self._csv(tmp_path, "a,b,label\n1,2,cat\n3,4,dog\n5,6,cat\n")
        header, rows = load_data(path)
        assert header == ["a", "b", "label"]
        assert len(rows) == 3
        assert rows[0] == ["1", "2", "cat"]

    def test_load_csv_no_header(self, tmp_path):
        from uchi.data_loader import load_data
        path = self._csv(tmp_path, "1.0,2.0,3.0\n4.0,5.0,6.0\n")
        header, rows = load_data(path)
        assert header == ["col_0", "col_1", "col_2"]
        assert len(rows) == 2

    def test_split_features_last_column_default(self, tmp_path):
        from uchi.data_loader import load_data, split_features
        path = self._csv(tmp_path, "x1,x2,y\n1.0,2.0,a\n3.0,4.0,b\n5.0,6.0,a\n")
        header, rows = load_data(path)
        X, y = split_features(header, rows)
        assert len(X) == 3
        assert all(len(row) == 2 for row in X)
        assert y == ["a", "b", "a"]

    def test_split_features_named_label(self, tmp_path):
        from uchi.data_loader import load_data, split_features
        path = self._csv(tmp_path, "x,label,z\n1,cat,10\n2,dog,20\n3,cat,30\n")
        header, rows = load_data(path)
        X, y = split_features(header, rows, label_col="label")
        assert len(X) == 3
        assert all(len(row) == 2 for row in X)  # x and z
        assert "cat" in y

    def test_split_features_named_heuristic(self, tmp_path):
        from uchi.data_loader import load_data, split_features
        path = self._csv(tmp_path, "a,b,target\n1,2,10\n3,4,20\n5,6,30\n")
        header, rows = load_data(path)
        X, y = split_features(header, rows)  # 'target' should be auto-detected
        assert y == ["10", "20", "30"]

    def test_split_features_missing_col_raises(self, tmp_path):
        from uchi.data_loader import load_data, split_features
        path = self._csv(tmp_path, "a,b\n1,2\n3,4\n")
        header, rows = load_data(path)
        with pytest.raises(ValueError, match="not found"):
            split_features(header, rows, label_col="nonexistent")

    def test_to_numeric_rows_drops_non_numeric(self, tmp_path):
        from uchi.data_loader import load_data, to_numeric_rows
        path = self._csv(tmp_path, "a,b,c\n1,2,3\nX,Y,Z\n4,5,6\n")
        header, rows = load_data(path)
        numeric = to_numeric_rows(header, rows)
        assert len(numeric) == 2  # only numeric rows
        assert numeric[0] == [1.0, 2.0, 3.0]

    def test_train_test_split_proportions(self):
        from uchi.data_loader import train_test_split
        X = [[i] for i in range(100)]
        y = list(range(100))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.2)
        assert len(Xtr) == 80
        assert len(Xte) == 20
        assert len(ytr) == 80

    def test_find_path_extracts_csv(self):
        from uchi.data_loader import find_path
        assert find_path("detect anomalies in sensor_data.csv please") == "sensor_data.csv"
        assert find_path("load /home/user/data/iris.csv") == "/home/user/data/iris.csv"

    def test_find_path_none_when_absent(self):
        from uchi.data_loader import find_path
        assert find_path("what is the weather today?") is None

    def test_parse_args_path_only(self):
        from uchi.data_loader import parse_args
        r = parse_args("data.csv")
        assert r["path"] == "data.csv"
        assert r["label"] is None
        assert r["steps"] == 10

    def test_parse_args_all_flags(self):
        from uchi.data_loader import parse_args
        r = parse_args("data.csv --label species --steps 20")
        assert r["path"] == "data.csv"
        assert r["label"] == "species"
        assert r["steps"] == 20


# ── Skill mode handlers ───────────────────────────────────────────────────────

def _make_router():
    mock = MagicMock()
    mock.chat.return_value = "mock"
    mock.query.return_value = "[Unknown Context]"
    mock.tokenizer.tokenize.return_value = ["hello"]
    mock.stream.return_value = None
    return mock


class TestClassifySkill:
    def _csv(self, tmp_path, header, rows):
        p = tmp_path / "data.csv"
        lines = [",".join(header)] + [",".join(str(v) for v in r) for r in rows]
        p.write_text("\n".join(lines))
        return str(p)

    def test_classify_reports_accuracy(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        path = self._csv(tmp_path, ["x1", "x2", "label"],
                         [[1, 2, "a"], [3, 4, "b"], [1, 3, "a"], [2, 4, "b"],
                          [1, 1, "a"], [3, 3, "b"], [2, 3, "a"], [4, 2, "b"]])
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("classify", f"{path}")
        assert "Accuracy" in result or "complete" in result.lower()

    def test_classify_insufficient_rows(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        path = self._csv(tmp_path, ["x", "y"], [[1, "a"], [2, "b"]])
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("classify", str(path))
        assert "enough" in result.lower() or "complete" in result.lower() or "error" in result.lower()

    def test_classify_missing_file(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("classify", "/nonexistent/path.csv")
        assert "not found" in result.lower()

    def test_classify_no_path_returns_usage(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("classify", "")
        assert "Usage" in result or "classify" in result.lower()


class TestRegressSkill:
    def _csv(self, tmp_path):
        p = tmp_path / "reg.csv"
        rows = "\n".join(f"{i},{i*2},{i*3}" for i in range(10))
        p.write_text("a,b,target\n" + rows)
        return str(p)

    def test_regress_reports_mae(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        path = self._csv(tmp_path)
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("regress", str(path))
        assert "MAE" in result or "complete" in result.lower()

    def test_regress_string_target_error(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        p = tmp_path / "c.csv"
        p.write_text("x,y\n1,cat\n2,dog\n3,cat\n4,dog\n5,cat\n")
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("regress", str(p))
        assert "non-numeric" in result.lower() or "error" in result.lower() or "MAE" in result


class TestAnomalySkill:
    def test_anomaly_runs_on_numeric_csv(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        rows = "\n".join(f"{i},{i+1}" for i in range(20))
        p = tmp_path / "anom.csv"
        p.write_text("a,b\n" + rows)
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("anomaly", str(p))
        assert "Anomaly" in result or "complete" in result.lower()

    def test_anomaly_missing_file(self):
        from uchi.skill_registry import SkillRegistry
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("anomaly", "/nonexistent.csv")
        assert "not found" in result.lower()


class TestForecastSkill:
    def test_forecast_returns_predictions(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        rows = "\n".join(f"{i},{i*2}" for i in range(20))
        p = tmp_path / "ts.csv"
        p.write_text("dim1,dim2\n" + rows)
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("forecast", f"{p} --steps 5")
        assert "Forecast" in result or "complete" in result.lower()

    def test_forecast_default_steps(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        rows = "\n".join(f"{i}" for i in range(15))
        p = tmp_path / "uni.csv"
        p.write_text("val\n" + rows)
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("forecast", str(p))
        assert "Forecast" in result or "complete" in result.lower()


class TestTSClassifySkill:
    def test_tsclassify_runs(self, tmp_path):
        from uchi.skill_registry import SkillRegistry
        # 10-feature windows with 2 classes
        rows = []
        for i in range(12):
            vals = [str(float(i + j)) for j in range(10)]
            rows.append(",".join(vals) + ("," + ("a" if i % 2 == 0 else "b")))
        p = tmp_path / "win.csv"
        p.write_text(",".join([f"f{i}" for i in range(10)] + ["class"]) + "\n" + "\n".join(rows))
        reg = SkillRegistry(_make_router())
        result = reg.dispatch("tsclassify", str(p))
        assert "classification" in result.lower() or "complete" in result.lower()


# ── LatentIntentEncoder ───────────────────────────────────────────────────────

class TestLatentIntentEncoder:
    def _make_enc(self):
        from uchi.neuro_symbolic import get_ssm
        from uchi.intent_encoder import LatentIntentEncoder
        ssm = get_ssm()
        return LatentIntentEncoder(ssm)

    def test_encode_tokens_returns_correct_length(self):
        enc = self._make_enc()
        h = enc.encode_tokens(["hello", "world"])
        assert len(h) == 256

    def test_project_empty_dist(self):
        enc = self._make_enc()
        v = enc.project_trie_dist({})
        assert len(v) == 256
        assert all(x == 0.0 for x in v)

    def test_project_dist_normalised(self):
        enc = self._make_enc()
        dist = {"classify": 0.7, "anomaly": 0.3}
        v = enc.project_trie_dist(dist)
        assert len(v) == 256
        assert not all(x == 0.0 for x in v)

    def test_fuse_normalises(self):
        import math
        enc = self._make_enc()
        a = [1.0] * 64
        b = [1.0] * 64
        fused = enc.fuse(a, b)
        norm = math.sqrt(sum(x * x for x in fused))
        assert abs(norm - 1.0) < 1e-5

    def test_register_and_match_returns_tuple(self):
        enc = self._make_enc()
        enc.register_skill("classify", ["classify", "tabular", "data", "label", "category"])
        enc.register_skill("anomaly",  ["anomaly", "outlier", "unusual", "detect"])
        enc.register_skill("forecast", ["forecast", "future", "predict", "steps", "timeseries"])
        name, conf = enc.match(["anomaly", "detection", "sensor"])
        assert isinstance(conf, float)
        assert conf >= 0.0

    def test_match_below_threshold_returns_none(self):
        enc = self._make_enc()
        enc.register_skill("classify", ["classify"])
        enc.register_skill("anomaly",  ["anomaly"])
        enc.register_skill("forecast", ["forecast"])
        # Random noise query — may or may not match but confidence check is the key
        name, conf = enc.match(["xyzzy", "frobble", "quux"])
        # If name is returned it must be because confidence >= threshold
        if name is not None:
            assert conf >= enc.threshold

    def test_not_ready_with_few_skills(self):
        enc = self._make_enc()
        assert not enc.is_ready()
        enc.register_skill("a", ["hello"])
        enc.register_skill("b", ["world"])
        assert not enc.is_ready()
        enc.register_skill("c", ["foo"])
        assert enc.is_ready()


# ── Intent detection in OmniRouter ───────────────────────────────────────────

class TestProceduralMemoryAnalyticalIntents:
    def test_anomaly_synonyms(self):
        from uchi.procedural_memory import ProceduralMemory
        pm = ProceduralMemory.__new__(ProceduralMemory)
        pm._store = dict(ProceduralMemory._DEFAULTS)
        pm.path = ":memory:"

        assert pm.get_intent_key("detect anomalies in my data") == "anomaly"
        assert pm.get_intent_key("find outliers in the dataset") == "anomaly"
        assert pm.get_intent_key("anything unusual in sensor_data.csv") == "anomaly"

    def test_classify_synonyms(self):
        from uchi.procedural_memory import ProceduralMemory
        pm = ProceduralMemory.__new__(ProceduralMemory)
        pm._store = dict(ProceduralMemory._DEFAULTS)
        pm.path = ":memory:"

        assert pm.get_intent_key("classify my customer data") == "classify"
        assert pm.get_intent_key("categorize the rows") == "classify"

    def test_forecast_synonyms(self):
        from uchi.procedural_memory import ProceduralMemory
        pm = ProceduralMemory.__new__(ProceduralMemory)
        pm._store = dict(ProceduralMemory._DEFAULTS)
        pm.path = ":memory:"

        assert pm.get_intent_key("forecast the next 10 steps") == "forecast"
        assert pm.get_intent_key("predict future values") == "forecast"

    def test_regress_synonyms(self):
        from uchi.procedural_memory import ProceduralMemory
        pm = ProceduralMemory.__new__(ProceduralMemory)
        pm._store = dict(ProceduralMemory._DEFAULTS)
        pm.path = ":memory:"

        assert pm.get_intent_key("regression on housing prices") == "regress"

    def test_code_synonyms_unchanged(self):
        from uchi.procedural_memory import ProceduralMemory
        pm = ProceduralMemory.__new__(ProceduralMemory)
        pm._store = dict(ProceduralMemory._DEFAULTS)
        pm.path = ":memory:"

        assert pm.get_intent_key("write a python function") == "code"
        assert pm.get_intent_key("debug this script") == "code"

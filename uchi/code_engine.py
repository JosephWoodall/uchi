"""
code_engine.py
==============
Phase 1 + 3: Parallel MCTS code generation with REPL oracle and hole synthesis.

Workers run in threads (GIL-safe for read-only trie inference) with varied
temperatures for diversity. REPL oracle picks first candidate to compile.
When all candidates fail: emit ??HOLE:desc?? markers the user can fill in.
"""

import ast
import concurrent.futures
import subprocess
import sys
import tempfile
import os
import re
from typing import List, Tuple

HOLE_PATTERN = re.compile(r'\?\?HOLE:([^?]+)\?\?')


class REPLOracle:
    """Stateless compile-time verification. Returns (passed, reward)."""

    def verify(self, code: str, timeout: float = 3.0) -> Tuple[bool, float]:
        """Syntax parse + py_compile check. Returns (passed, reward)."""
        try:
            ast.parse(code)
        except SyntaxError:
            return False, -1.0

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return False, -0.5
            return True, 1.0
        except subprocess.TimeoutExpired:
            return False, -0.3
        except Exception:
            return False, -0.3
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class CodeEngine:
    """
    Phase 1 + 3: Parallel MCTS code generation with REPL oracle and hole synthesis.

    n_workers MCTS threads run concurrently with varied temperatures.
    First candidate to pass the REPL oracle is returned.
    If none pass, best candidate is returned with ??HOLE?? markers.
    """

    def __init__(self, predictor, n_workers: int = 4):
        self.predictor = predictor
        self.n_workers = n_workers
        self.oracle = REPLOracle()

    def generate_code(
        self,
        seed_tokens: list,
        max_tokens: int = 80,
    ) -> Tuple[str, float, bool]:
        """
        Run n_workers MCTS threads. Return (code_str, reward, passed_oracle).
        First candidate to pass compile check wins; falls back to best score + holes.
        """
        results: List[Tuple[str, float]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(self._mcts_worker, seed_tokens, max_tokens, i): i
                for i in range(self.n_workers)
            }
            for future in concurrent.futures.as_completed(futures, timeout=20.0):
                try:
                    code_str, score = future.result(timeout=15.0)
                    if code_str:
                        results.append((code_str, score))
                except Exception:
                    continue

        if not results:
            return self._synthesize_hole(seed_tokens), 0.0, False

        # Best score first → REPL oracle vets each candidate
        for code_str, score in sorted(results, key=lambda x: -x[1]):
            passed, reward = self.oracle.verify(code_str)
            if passed:
                return code_str, reward, True

        # None passed — return best candidate, possibly with holes
        best_code, best_score = max(results, key=lambda x: x[1])
        if best_score < 0.2 or len(best_code.strip()) < 10:
            best_code = self._synthesize_hole(seed_tokens)
        return best_code, best_score * 0.5, False

    def _mcts_worker(
        self, seed_tokens: list, max_tokens: int, worker_id: int
    ) -> Tuple[str, float]:
        """Single MCTS worker with varied temperature for diversity."""
        temperature = 0.05 + worker_id * 0.12
        try:
            pred = self.predictor.generate(
                n_tokens=max_tokens,
                seed=seed_tokens,
                temperature=temperature,
                use_mcts=True,
                stop_tokens=["<|user|>", "<|end|>"],
            )
            code_str = " ".join(str(t) for t in pred)
            # Quality score: favour longer, diverse outputs
            n = len(pred)
            unique_ratio = len(set(pred)) / max(1, n)
            score = min(1.0, n / 40.0) * (0.4 + unique_ratio * 0.6)
            return code_str, score
        except Exception:
            return "", 0.0

    def _synthesize_hole(self, seed_tokens: list) -> str:
        """Emit a skeleton with ??HOLE?? markers when trie can't fill the gap."""
        topic = " ".join(str(t) for t in seed_tokens[-6:])
        return f"def solution():\n    # ??HOLE:{topic}??\n    pass"

    # ── Hole utilities ────────────────────────────────────────────────────────

    @staticmethod
    def has_holes(code: str) -> bool:
        return bool(HOLE_PATTERN.search(code))

    @staticmethod
    def extract_holes(code: str) -> List[str]:
        return HOLE_PATTERN.findall(code)

    @staticmethod
    def fill_hole(code: str, hole_desc: str, replacement: str) -> str:
        """Replace first matching ??HOLE:desc?? with user-provided code."""
        return code.replace(f"??HOLE:{hole_desc}??", replacement, 1)

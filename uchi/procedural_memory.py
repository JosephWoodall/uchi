import os
import ast
import tempfile
import subprocess
import sys
from typing import Tuple

class REPLOracle:
    """Stateless compile-time verification for generated Python code."""
    def verify(self, code: str, timeout: float = 3.0) -> Tuple[bool, float]:
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
                capture_output=True, timeout=timeout
            )
            return (result.returncode == 0), (1.0 if result.returncode == 0 else -0.5)
        except Exception:
            return False, -0.3
        finally:
            try: os.remove(path)
            except OSError: pass

    def execute(self, code: str, timeout: float = 3.0) -> Tuple[bool, str]:
        """Executes the code and returns (success, stdout/stderr)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            # If the code defines `def run():`, call it and print the result
            f.write("\n\nif __name__ == '__main__':\n    try:\n        print(run())\n    except Exception as e:\n        print('Error:', e)\n")
            path = f.name

        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, timeout=timeout, text=True
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip() or result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "TimeoutExpired"
        except Exception as e:
            return False, str(e)
        finally:
            try: os.remove(path)
            except OSError: pass

class ProceduralMemory:
    """
    Autonomous Tool Creation.
    Uses FLUX to write a Python script for a missing skill, verifies it via REPL,
    and saves it permanently to the skills registry.
    """
    def __init__(self, proposer, skills_dir="uchi/skills"):
        self.proposer = proposer
        self.oracle = REPLOracle()
        self.skills_dir = skills_dir
        os.makedirs(self.skills_dir, exist_ok=True)

    def create_skill(self, name: str, description: str) -> bool:
        """Propose, verify, and persist a new skill."""
        if not self.proposer: return False

        prompt = (f"Write a complete, standalone Python function named 'run' "
                  f"that takes (args: str, **kwargs) and solves this: {description}. "
                  f"Only output the Python code.")
        
        try:
            code = self.proposer.propose(prompt, evidence=[])
            
            # Clean markdown code block syntax if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
                
            passed, _ = self.oracle.verify(code)
            if passed:
                self._save_skill_file(name, description, code)
                return True
        except Exception:
            pass
        return False

    def _save_skill_file(self, name: str, desc: str, code: str):
        skill_path = os.path.join(self.skills_dir, f"{name}.md")
        content = f"---\nname: {name}\ndescription: {desc}\nmode: code\nargs: <input>\n---\n\n```python\n{code}\n```"
        with open(skill_path, "w") as f:
            f.write(content)

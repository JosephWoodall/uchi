import os

from .code_engine import REPLOracle

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

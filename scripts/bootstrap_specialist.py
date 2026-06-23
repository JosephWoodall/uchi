"""
bootstrap_specialist.py
=======================
Bootstrap separate specialist brain files for the MoE SpecialistPool.

Creates three independent OmniRouter instances trained on domain-specific
corpora and saves them as brain_code.uchi, brain_math.uchi, brain_convo.uchi.

Usage:
    python scripts/bootstrap_specialist.py [--domain code|math|convo|all]
    python scripts/bootstrap_specialist.py --domain all
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uchi.omni_router import OmniRouter
from uchi.cli import save_brain, load_brain
from uchi.code_tokenizer import ASTCodeTokenizer


def _make_router():
    """Create a minimal OmniRouter (no knowledge bootstrap on cold start)."""
    router = OmniRouter.__new__(OmniRouter)
    router.use_bpe = False
    from uchi.omni_tokenizer import OmniTokenizer
    from uchi.online_tokenizer import OnlineTokenizer
    from uchi.memory import AssociativeMemory
    from uchi.generative import SequenceGenerator
    from uchi.grpo import AgenticBaseline
    from uchi.procedural_memory import ProceduralMemory
    from uchi.code_engine import CodeEngine
    from uchi.specialist_pool import SpecialistPool
    import torch
    from uchi.neuro_symbolic import get_ssm

    router.tokenizer = OmniTokenizer(use_wordnet=True)
    router.bpe = None
    router.memory = AssociativeMemory(window_size=5)
    router.predictor = SequenceGenerator(context_length=10, min_context_length=0)
    router.baseline = AgenticBaseline()
    router.procedural = ProceduralMemory()
    router.ssm_optimizer = torch.optim.Adam(get_ssm().parameters(), lr=1e-3)
    router._knowledge_bootstrapped = True
    router.last_sequence = None
    router.last_walk_data = {}
    router.code_engine = CodeEngine(router.predictor, n_workers=4)
    router.specialist_pool = SpecialistPool(router)
    return router


def bootstrap_code_brain(output_path: str):
    """Train a code-specialist brain on Python stdlib patterns + AST tokens."""
    print(f"[*] Bootstrapping code specialist → {output_path}")
    router = _make_router()
    ast_tok = ASTCodeTokenizer()

    import inspect, pkgutil, importlib

    count = 0
    stdlib_mods = [
        "os", "sys", "re", "json", "math", "itertools", "functools",
        "collections", "pathlib", "datetime", "io", "string", "random",
        "hashlib", "base64", "struct", "copy", "types", "typing",
        "abc", "contextlib", "dataclasses", "enum", "heapq", "bisect",
        "operator", "array", "statistics", "decimal", "fractions",
        "threading", "queue", "subprocess", "socket", "http", "urllib",
        "csv", "sqlite3", "logging", "unittest", "argparse", "shutil",
    ]

    for mod_name in stdlib_mods:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            try:
                src = inspect.getsource(obj)
                ast_tokens = ast_tok.tokenize_source(src)
                if len(ast_tokens) >= 4:
                    router.stream(ast_tokens)
                    count += 1
            except (OSError, TypeError):
                continue

    print(f"[+] Ingested {count} function AST patterns")
    save_brain(router, output_path)


def bootstrap_math_brain(output_path: str):
    """Train a math/physics specialist brain on formula and arithmetic patterns."""
    print(f"[*] Bootstrapping math specialist → {output_path}")
    router = _make_router()

    # Core arithmetic and algebra patterns
    math_sequences = [
        "distance equals speed multiplied by time",
        "force equals mass multiplied by acceleration",
        "energy equals mass multiplied by speed squared",
        "kinetic energy equals half mass velocity squared",
        "momentum equals mass times velocity",
        "work equals force times displacement",
        "power equals work divided by time",
        "pressure equals force divided by area",
        "density equals mass divided by volume",
        "velocity equals displacement divided by time",
        "acceleration equals change in velocity divided by time",
        "ohms law voltage equals current times resistance",
        "frequency equals one divided by period",
        "wavelength equals speed divided by frequency",
        "area of circle equals pi radius squared",
        "circumference equals two pi radius",
        "volume of sphere equals four thirds pi radius cubed",
        "pythagorean theorem a squared plus b squared equals c squared",
        "quadratic formula x equals negative b plus minus sqrt b squared minus four ac divided by two a",
        "derivative of x squared is two x",
        "integral of x squared is x cubed divided by three plus constant",
        "log of a times b equals log a plus log b",
        "sin squared plus cos squared equals one",
        "eulers formula e to the i pi equals negative one",
        "mean equals sum divided by count",
        "variance equals mean of squares minus square of mean",
        "standard deviation equals square root of variance",
        "probability of a and b equals probability of a times probability of b if independent",
        "bayes theorem posterior equals likelihood times prior divided by evidence",
        "fibonacci sequence each number is sum of two preceding numbers",
        "factorial n equals n times n minus one times n minus two down to one",
        "binomial coefficient n choose k equals n factorial divided by k factorial times n minus k factorial",
        "matrix multiplication c ij equals sum of a ik times b kj",
        "dot product a dot b equals sum of ai times bi",
        "cross product of two vectors produces perpendicular vector",
        "gradient points in direction of steepest ascent",
        "laplacian is sum of second partial derivatives",
    ]

    for seq in math_sequences * 10:
        router.stream(seq.split())

    print(f"[+] Ingested {len(math_sequences)} math pattern sequences")
    save_brain(router, output_path)


def bootstrap_convo_brain(output_path: str):
    """Train a conversation specialist brain on dialogue patterns."""
    print(f"[*] Bootstrapping convo specialist → {output_path}")
    # Start from existing brain.uchi if available (inherit world knowledge)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_brain = os.path.join(project_root, "brain.uchi")

    if os.path.exists(main_brain):
        print("[*] Inheriting from main brain.uchi...")
        router = load_brain(main_brain)
        if router is None:
            router = _make_router()
    else:
        router = _make_router()

    # Additional conversational patterns
    convo_patterns = [
        "<|user|> hello <|assistant|> hello there how can i help you today",
        "<|user|> hi <|assistant|> hi how can i assist you",
        "<|user|> what is your name <|assistant|> i am uchi a deterministic sequence predictor",
        "<|user|> who created you <|assistant|> i was created by joseph woodall",
        "<|user|> what can you do <|assistant|> i can predict sequences answer questions and help with code",
        "<|user|> how are you <|assistant|> i am functioning well and ready to help",
        "<|user|> tell me a fact <|assistant|> the speed of light is approximately three hundred million meters per second",
        "<|user|> what is python <|assistant|> python is a high level programming language known for readability",
        "<|user|> thank you <|assistant|> you are welcome",
        "<|user|> goodbye <|assistant|> goodbye have a great day",
        "<|user|> help <|assistant|> i can help with questions predictions and code generation",
        "<|user|> explain recursion <|assistant|> recursion is when a function calls itself to solve a smaller version of the same problem",
        "<|user|> what is machine learning <|assistant|> machine learning is a field where systems learn patterns from data without explicit programming",
        "<|user|> what is a neural network <|assistant|> a neural network is a computational model inspired by biological neurons that learns representations",
        "<|user|> explain big o notation <|assistant|> big o notation describes how algorithm runtime grows relative to input size",
        "<|user|> what is a trie <|assistant|> a trie is a tree data structure that stores strings by sharing common prefixes",
        "<|user|> what is entropy <|assistant|> entropy measures uncertainty or randomness in a probability distribution",
        "<|user|> what is mcts <|assistant|> monte carlo tree search is a planning algorithm that uses random simulations to evaluate moves",
    ]

    for turn in convo_patterns * 10:
        router.stream(turn.split())

    print(f"[+] Ingested {len(convo_patterns)} conversation patterns")
    save_brain(router, output_path)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap specialist brains for MoE SpecialistPool")
    parser.add_argument("--domain", choices=["code", "math", "convo", "all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.domain in ("code", "all"):
        bootstrap_code_brain(os.path.join(project_root, "brain_code.uchi"))

    if args.domain in ("math", "all"):
        bootstrap_math_brain(os.path.join(project_root, "brain_math.uchi"))

    if args.domain in ("convo", "all"):
        bootstrap_convo_brain(os.path.join(project_root, "brain_convo.uchi"))

    print("\n[+] Specialist bootstrap complete.")
    print("    brain_code.uchi  → code intent queries")
    print("    brain_math.uchi  → math/physics intent queries")
    print("    brain_convo.uchi → general conversation intent queries")


if __name__ == "__main__":
    main()

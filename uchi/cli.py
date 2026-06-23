import argparse
import sys
import os
from .omni_router import OmniRouter

from tqdm import tqdm
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Log
from textual import work

def ingest_file(router, filepath, quiet=False):
    """Injects massive context from a file into the OmniRouter with structural bounds."""
    if not os.path.exists(filepath):
        if not quiet:
            print(f"Error: File '{filepath}' not found.")
        return
        
    if not quiet:
        print(f"[*] Ingesting massive context from {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().split()
            
        filename = os.path.basename(filepath)
        bounded_text = [f"<|file:{filename}|>"] + text + ["<|/file|>"]
        router.stream(bounded_text)
        if not quiet:
            print(f"[+] Successfully injected {len(bounded_text)} bounded tokens into the Deterministic LLM.")
    except Exception as e:
        if not quiet:
            print(f"[-] Failed to ingest {filepath}: {e}")

def debate_loop(topic: str, rounds: int = 10):
    """
    Spawns two OmniRouter agents and forces them to predict/debate 
    continuously over the same topic.
    """
    print("===============================================================")
    print(" Uchi v0.2.0 - Multi-Agent Deterministic Debate")
    print(f" Topic: {topic}")
    print(" Engine: OmniRouter (BPE Stream Compression ENABLED)")
    print("===============================================================")
    
    # Initialize two separate deterministic brains
    # They both use BPE compression to prevent infinite RAM explosion
    agent_alpha = OmniRouter(use_bpe=True, memory_window=5)
    agent_beta  = OmniRouter(use_bpe=True, memory_window=5)
    
    # Seed their initial context
    seed_tokens = topic.split()
    agent_alpha.stream(seed_tokens)
    agent_beta.stream(seed_tokens)
    
    current_context = seed_tokens
    
    for i in range(rounds):
        import time
        time.sleep(1) # Slow down for readability
        
        print(f"\n--- Round {i+1} ---")
        
        # Alpha speaks
        alpha_reply = agent_alpha.predict_future(current_context, steps=5)
        print(f"Agent Alpha: {' '.join(alpha_reply)}")
        
        # Stream Alpha's reply into Beta's brain so Beta hears it
        agent_beta.stream(alpha_reply)
        
        # Beta speaks
        beta_reply = agent_beta.predict_future(alpha_reply, steps=5)
        print(f"Agent Beta:  {' '.join(beta_reply)}")
        
        # Stream Beta's reply back into Alpha's brain
        agent_alpha.stream(beta_reply)
        
        current_context = beta_reply
        
    print("\n[+] Debate concluded. Both agents successfully compressed the entire context history via BPE.")

import pickle
import concurrent.futures

ASCII_LOGO = r"""
       _  _
      (o)(o)
     /  __  \
    |  \__/  |
     \______/
ODUSP Daemon v0.2.0
"""

def save_brain(router, path: str = "brain.uchi"):
    """Serializes the entire Deterministic Cognitive Engine to disk."""
    print(f"\n[+] Saving neural state to {path}...")
    try:
        with open(path, "wb") as f:
            pickle.dump(router, f)
        print(f"[+] Brain successfully persisted. ({os.path.getsize(path)} bytes)")
    except Exception as e:
        print(f"[-] Failed to save brain: {e}")

def load_brain(path: str = "brain.uchi") -> OmniRouter:
    """Deserializes the Cognitive Engine from disk."""
    print(f"[*] Loading persistent brain state from {path}...")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[-] Failed to load brain: {e}")
        return None

def preload_context(router, path: str):
    """
    Recursively preloads a file or directory into the router using parallel processing.
    """
    if not os.path.exists(path):
        print(f"Error: Preload path '{path}' not found.")
        return
        
    if os.path.isfile(path):
        ingest_file(router, path)
    else:
        filepaths = []
        for root, dirs, files in os.walk(path):
            # Skip virtual environments, hidden folders, and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env', 'uchi.egg-info']]
            for file in files:
                filepath = os.path.join(root, file)
                if file.endswith((".txt", ".md", ".py", ".cpp", ".js", ".json")):
                    filepaths.append(filepath)
        
        print(f"[*] Pre-training ODUSP from {path} in parallel...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda f: ingest_file(router, f, quiet=True), filepaths), total=len(filepaths), desc="Ingesting files"))

# ANSI Color Codes
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_ai_msg(prefix, msg):
    print(f"\n{CYAN}{BOLD}ODUSP ({prefix}):{RESET} {msg}\n")

def print_help():
    print(f"\n{YELLOW}{BOLD}Available Commands:{RESET}")
    print(f"  {GREEN}/help{RESET}             Show this help menu")
    print(f"  {GREEN}/load <file>{RESET}      Dynamically stream a new file into the Geometric Trie")
    print(f"  {GREEN}/query <text>{RESET}     Execute Zero-Shot Q&A against the Associative Memory")
    print(f"  {GREEN}/predict <steps>{RESET}  Force the engine to hallucinate forward <steps> tokens")
    print(f"  {GREEN}/save{RESET}             Force serialize the current brain state to disk")
    print(f"  {GREEN}/quit{RESET}             Exit the session and save\n")

def main():
    parser = argparse.ArgumentParser(description="Uchi Omni-modal Deterministic Universal Sequence Predictor (ODUSP) CLI")
    parser.add_argument("--preload", type=str, default=None, help="Directory or file to preload context from")
    parser.add_argument("--brain", type=str, default="brain.uchi", help="Path to the persistent brain file")
    
    args = parser.parse_args()
    
    from uchi.tui.app import UchiApp
    app = UchiApp(args.brain, args.preload)
    app.run()

if __name__ == "__main__":
    main()

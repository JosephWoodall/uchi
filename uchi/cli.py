import argparse
import os
import pickle
import concurrent.futures

from tqdm import tqdm

def ingest_file(u, filepath, quiet=False):
    """Ingest a single file into a Uchi instance (FLUX + Uchi architecture).

    `u` is a `uchi.Uchi` (the TUI/skills pass their live instance). We delegate
    to `Uchi.ingest`, which reads the file and feeds it into the compounding
    semantic index via `learn()`.
    """
    if not os.path.exists(filepath):
        if not quiet:
            print(f"Error: File '{filepath}' not found.")
        return
    if not quiet:
        print(f"[*] Ingesting {filepath} into the brain...")
    try:
        u.ingest(filepath)
        if not quiet:
            print(f"[+] Ingested {os.path.basename(filepath)}.")
    except Exception as e:
        if not quiet:
            print(f"[-] Failed to ingest {filepath}: {e}")

ASCII_LOGO = r"""
       _  _
      (o)(o)
     /  __  \
    |  \__/  |
     \______/
ODUSP Daemon v0.2.0
"""

def save_brain(u, path: str = "brain.uchi"):
    """Persist a Uchi instance's compounding semantic index to disk.

    Delegates to `Uchi.save` (gzip-compressed pickle of the index). `u` is a
    `uchi.Uchi` instance passed by the TUI.
    """
    try:
        u.save(path)
        print(f"[+] Brain persisted to {path}.")
    except Exception as e:
        print(f"[-] Failed to save brain: {e}")

def get_bundled_brain_path() -> str:
    """Return the path to the brain bundled with the uchi package, or '' if absent."""
    try:
        import importlib.resources as _ir
        # Python 3.9+ path
        pkg_data = _ir.files("uchi") / "data" / "brain.uchi"
        candidate = str(pkg_data)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    # Fallback: resolve relative to this file
    fallback = os.path.join(os.path.dirname(__file__), "data", "brain.uchi")
    return fallback if os.path.exists(fallback) else ""


def load_brain(path: str = "brain.uchi"):
    """Deserialize the brain from disk (gzip or plain pickle).

    Returns the OmniRouter, or ``None`` if no loadable brain exists. Callers
    create a fresh cold router on ``None`` — load failures no longer trigger a
    heavy auto-rebuild (that was the retired Family C pipeline).
    """
    import gzip
    print(f"[*] Loading persistent brain state from {path}...")

    if not os.path.exists(path):
        bundled = get_bundled_brain_path()
        if bundled and bundled != path:
            path = bundled
        else:
            print(f"[-] Brain file not found: {path}")
            return None

    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except gzip.BadGzipFile:
        pass  # not gzip — try plain pickle
    except Exception as e:
        print(f"[-] Failed to load brain (gzip/pickle): {e}")
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[-] Failed to load brain: {e}")
        return None

def preload_context(u, path: str):
    """Recursively preload a file or directory into a Uchi instance.

    `Uchi.ingest` already walks directories (txt/md/py/json/csv/pdf) and feeds
    each file into the compounding index, so we delegate to it.
    """
    if not os.path.exists(path):
        print(f"Error: Preload path '{path}' not found.")
        return
    print(f"[*] Preloading context from {path}...")
    try:
        u.ingest(path)
        print(f"[+] Preloaded {path}.")
    except Exception as e:
        print(f"[-] Failed to preload {path}: {e}")

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
    parser = argparse.ArgumentParser(
        prog="uchi",
        description="Uchi — FLUX (Proposer) + Uchi (Verifier). "
                    "Run the terminal UI or the REST API server.",
    )
    parser.add_argument("command", nargs="?", default="tui", choices=["tui", "serve"],
                        help="tui: interactive terminal UI (default). serve: REST API server.")
    parser.add_argument("--preload", type=str, default=None,
                        help="[tui] File or directory to preload context from")
    parser.add_argument("--brain", type=str, default="brain.uchi",
                        help="[tui] Path to the persistent brain file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="[serve] Bind host")
    parser.add_argument("--port", type=int, default=8000, help="[serve] Bind port")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        print(f"[*] Serving Uchi REST API on http://{args.host}:{args.port} ...")
        uvicorn.run("uchi.api_server:app", host=args.host, port=args.port)
        return

    from uchi.tui.app import UchiApp
    app = UchiApp(args.brain, args.preload)
    app.run()

if __name__ == "__main__":
    main()

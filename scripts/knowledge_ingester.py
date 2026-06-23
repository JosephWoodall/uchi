import os
import sys
import glob
import ast
from pathlib import Path
from uchi.omni_router import OmniRouter
from uchi.cli import load_brain, save_brain
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def strip_docstrings_and_comments(source_code: str) -> str:
    """
    Parses Python source code and strips out all docstrings.
    Also removes full-line comments to prevent English words from polluting the trie.
    """
    try:
        # Parse AST to cleanly remove docstrings
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                if ast.get_docstring(node):
                    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                        node.body.pop(0)
                        
        cleaned_source = ast.unparse(tree)
        return cleaned_source
    except SyntaxError:
        # Fallback if invalid syntax: return raw without stripping
        return source_code

def ingest_directory(router: OmniRouter, directory_path: str, max_files: int = -1):
    """
    Crawls a directory recursively for .py files and streams them into the graph.
    Relies heavily on FractalBPETokenizer and NodeCompressor to prevent RAM explosion.
    Uses strict boundaries to isolate code from natural language graphs.
    """
    py_files = glob.glob(os.path.join(directory_path, "**", "*.py"), recursive=True)
    
    if max_files > 0:
        py_files = py_files[:max_files]
        
    logging.info(f"Found {len(py_files)} Python files in {directory_path} to ingest.")
    
    total_tokens = 0
    for idx, filepath in enumerate(py_files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Strip English words (docstrings) to prevent context collision
            cleaned_content = strip_docstrings_and_comments(content)
            
            # Wrap in strict topological boundary tags
            turn_str = f"[SYS_CODE_START] {cleaned_content} [SYS_CODE_END]"
            
            tokens = turn_str.split()
            total_tokens += len(tokens)
            
            # Stream directly into the OmniRouter
            router.stream(tokens)
            
            if idx > 0 and idx % 10 == 0:
                logging.info(f"Ingested {idx + 1}/{len(py_files)} files... (Total tokens: {total_tokens})")
                
                if hasattr(router.predictor, "compressor") and router.predictor.compressor:
                    stats = router.predictor.compressor.compress_pass(router.predictor._root)
                    
        except Exception as e:
            logging.error(f"Failed to ingest {filepath}: {e}")

    logging.info(f"Ingestion complete. Total raw tokens streamed: {total_tokens}")

if __name__ == "__main__":
    router = load_brain()
    
    uchi_dir = str(Path(__file__).parent.parent / "uchi")
    ingest_directory(router, uchi_dir)
    
    import collections
    stdlib_dir = os.path.dirname(collections.__file__)
    ingest_directory(router, stdlib_dir, max_files=50)
    
    save_brain(router)

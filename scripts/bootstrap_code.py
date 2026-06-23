import os
import sys
import ast
import logging
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def bootstrap_code(brain_path="brain.uchi"):
    logging.info("Starting AST-based Code Bootstrapping")

    from uchi.cli import load_brain, save_brain
    from uchi.omni_router import OmniRouter

    router = load_brain(brain_path)
    if router is None:
        logging.info("Creating new brain for bootstrapping")
        router = OmniRouter(use_bpe=False)
        
    from uchi.neuro_symbolic import get_ssm
    ssm = get_ssm()
    ssm.train()
    
    import torch
    optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)
    
    lib_path = os.path.join(sys.base_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}")
    logging.info(f"Scanning Python standard library at {lib_path}")
    
    files_processed = 0
    functions_extracted = 0
    
    for root, dirs, files in os.walk(lib_path):
        if "test" in root or "site-packages" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                        
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name.startswith("_"):
                                continue
                                
                            docstring = ast.get_docstring(node)
                            if not docstring:
                                continue
                                
                            body_source = ast.get_source_segment(source, node)
                            
                            user_prompt = f"write a function that: {docstring.split('.')[0]}"
                            assistant_response = body_source
                            
                            tokens = ["<|user|>"] + router.tokenizer.tokenize(user_prompt.split(), is_inference=False) + \
                                     ["<|assistant|>"] + router.tokenizer.tokenize(assistant_response.split(), is_inference=False)
                            
                            router.stream(tokens)
                            
                            optimizer.zero_grad()
                            v_loss = ssm.update_value(tokens, reward=1.0)
                            d_loss = ssm.train_dynamics(tokens)
                            loss = v_loss + d_loss
                            loss.backward()
                            optimizer.step()
                            
                            functions_extracted += 1
                            if functions_extracted % 100 == 0:
                                logging.info(f"Extracted {functions_extracted} functions...")
                                
                    files_processed += 1
                    if functions_extracted > 1000:
                        break
                except Exception as e:
                    pass
        if functions_extracted > 1000:
            break
            
    logging.info(f"Finished code bootstrap. Processed {files_processed} files and {functions_extracted} functions.")
    save_brain(router, brain_path)
    torch.save(ssm.state_dict(), "ssm_dynamics.pt")
    logging.info("Code knowledge ingested.")

def run(router, progress_callback=None):
    """
    Ingests Python stdlib function patterns into the trie.
    Accepts an existing router instance — does not load or save brain.uchi.
    """
    from uchi.neuro_symbolic import get_ssm
    import torch

    ssm = get_ssm()
    ssm.train()
    optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)

    lib_path = os.path.join(sys.base_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}")

    cap = 500
    functions_extracted = 0

    for root, dirs, files in os.walk(lib_path):
        if "test" in root or "site-packages" in root:
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()

                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name.startswith("_"):
                                continue

                            docstring = ast.get_docstring(node)
                            if not docstring:
                                continue

                            body_source = ast.get_source_segment(source, node)

                            user_prompt = f"write a function that: {docstring.split('.')[0]}"
                            assistant_response = body_source

                            tokens = ["<|user|>"] + router.tokenizer.tokenize(user_prompt.split(), is_inference=False) + \
                                     ["<|assistant|>"] + router.tokenizer.tokenize(assistant_response.split(), is_inference=False)

                            router.stream(tokens)

                            optimizer.zero_grad()
                            v_loss = ssm.update_value(tokens, reward=1.0)
                            d_loss = ssm.train_dynamics(tokens)
                            loss = v_loss + d_loss
                            loss.backward()
                            optimizer.step()

                            functions_extracted += 1
                            if progress_callback:
                                progress_callback(functions_extracted, cap)

                            if functions_extracted >= cap:
                                return

                except Exception:
                    pass

        if functions_extracted >= cap:
            return


if __name__ == "__main__":
    bootstrap_code()

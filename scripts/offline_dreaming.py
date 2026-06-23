import os
import subprocess
import tempfile
from pathlib import Path
from uchi.omni_router import OmniRouter
from uchi.cli import load_brain, save_brain
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHALLENGES = [
    {
        "prompt": "write a python function called is_even that takes an integer n and returns true if n is even else false",
        "test": "assert is_even(4) is True\nassert is_even(5) is False"
    },
    {
        "prompt": "write a python function called reverse_string that takes a string s and returns it reversed",
        "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('abc') == 'cba'"
    },
    {
        "prompt": "write a python function to add two numbers",
        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    }
]

def rl_event(event_type, msg):
    logging.info(f"[RL ENGINE: {event_type.upper()}] {msg}")

def offline_dreaming(router: OmniRouter, iterations: int = 10):
    """
    Simulates offline evolutionary RL. Uchi attempts challenges, 
    injects failures back as context, and uses creativity > 0 to stochastically mutate.
    """
    logging.info("Starting Offline Dreaming RL Loop...")
    
    for i in range(iterations):
        for challenge in CHALLENGES:
            prompt = challenge["prompt"]
            test_code = challenge["test"]
            
            logging.info(f"Attempting: {prompt}")
            
            # Predict with stochastic mutation (creativity)
            tokens = ["<|user|>"] + router.tokenizer.tokenize(prompt.split(), is_inference=True) + ["<|assistant|>"]
            
            # Use creativity > 0 to explore the graph randomly if it gets stuck
            pred = router.predict_future(tokens, steps=60, temperature=0.5, creativity=0.3)
            
            reply = []
            recording = True
            for p in pred:
                if recording and p in ("<|user|>", "<|assistant|>", "<|end|>"):
                    break
                if recording:
                    reply.append(p)
                    
            if not reply:
                continue
                
            predicted_tokens = router.tokenizer.detokenize(reply)
            reply_str = ' '.join([str(p) for p in predicted_tokens])
            
            if "```python" in reply_str and "```" in reply_str.split("```python")[1]:
                code_block = reply_str.split("```python")[1].split("```")[0].strip()
                
                # Try running the sandbox
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code_block + "\n\n" + test_code)
                    temp_path = f.name
                    
                try:
                    result = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        logging.info("Success! Double-reinforcing topological path.")
                        # Double Reinforce the sequence!
                        success_sequence = tokens + reply
                        router.stream(success_sequence)
                        router.stream(success_sequence)
                    else:
                        logging.warning("Failed. Pruning and injecting traceback.")
                        failure_sequence = tokens + reply
                        router.predictor.unlearn(failure_sequence)
                        
                        # Inject error context
                        error_msg = result.stderr.strip().split('\n')[-1]
                        error_context = f"the code you wrote raised an error: {error_msg} fix the code for: {prompt}"
                        
                        err_tokens = ["<|user|>"] + router.tokenizer.tokenize(error_context.split(), is_inference=True) + ["<|assistant|>"]
                        router.stream(err_tokens)
                        
                except subprocess.TimeoutExpired:
                    logging.error("Sandbox timeout.")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

if __name__ == "__main__":
    router = load_brain()
    offline_dreaming(router, iterations=5)
    save_brain(router)

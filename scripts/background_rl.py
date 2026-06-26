import os
import sys
import time
import subprocess
import tempfile
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

def run_rl_loop(router):
    logging.info("Running RL minute challenge...")
    import random
    challenge = random.choice(CHALLENGES)
    prompt = challenge["prompt"]
    test_code = challenge["test"]
    
    tokens = ["<|user|>"] + router.tokenizer.tokenize(prompt.split(), is_inference=True) + ["<|assistant|>"]
    pred = router.predict_future(tokens, steps=60, temperature=0.5, creativity=0.3)
    
    reply = []
    recording = True
    for p in pred:
        if recording and p in ("<|user|>", "<|assistant|>", "<|end|>"):
            break
        if recording:
            reply.append(p)
            
    if not reply:
        return
        
    predicted_tokens = router.tokenizer.detokenize(reply)
    reply_str = ' '.join([str(p) for p in predicted_tokens])
    
    if "```python" in reply_str and "```" in reply_str.split("```python")[1]:
        code_block = reply_str.split("```python")[1].split("```")[0].strip()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_block + "\n\n" + test_code)
            temp_path = f.name
            
        try:
            result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=3)
            sequence = tokens + reply
            
            # Neural SSM Tuning
            from uchi.neuro_symbolic import get_ssm
            ssm = get_ssm()
            import torch
            optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-4)
            optimizer.zero_grad()
            
            if result.returncode == 0:
                logging.info(f"[RL PASS] {prompt}")
                router.stream(sequence)
                v_loss = ssm.update_value(sequence, reward=1.0)
            else:
                logging.warning(f"[RL FAIL] {prompt}. Pruning...")
                router.predictor.unlearn(sequence, strength=0.9)
                v_loss = ssm.update_value(sequence, reward=-1.0)
                
            d_loss = ssm.train_dynamics(sequence)
            loss = v_loss + d_loss
            if hasattr(loss, 'backward'):
                loss.backward()
                optimizer.step()
                torch.save(ssm.state_dict(), "ssm_dynamics.pt")
                
        except subprocess.TimeoutExpired:
            logging.error("Sandbox timeout.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    brain_path = "brain.uchi"
    if not os.path.exists(brain_path):
        logging.error("Brain not found. Start Uchi first.")
        exit(1)
        
    logging.info("Starting Background RL Daemon...")
    while True:
        try:
            router = load_brain(brain_path)
            if router:
                run_rl_loop(router)
                save_brain(router, brain_path)
        except Exception as e:
            logging.error(f"RL loop error: {e}")
        time.sleep(10)

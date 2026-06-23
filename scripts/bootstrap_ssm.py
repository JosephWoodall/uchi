import os
import sys
import logging
from datasets import load_dataset
import torch
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uchi.neuro_symbolic import get_ssm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def bootstrap_agent(brain_path="brain.uchi", sample_size=500):
    logging.info("Starting Massive Heavy-Lifting Pre-training (Bootstrapping Phase)")
    
    # Check if brain exists, else create new
    if os.path.exists(brain_path):
        logging.info(f"Loading existing brain from {brain_path}")
        with open(brain_path, "rb") as f:
            router = pickle.load(f)
    else:
        from uchi.omni_router import OmniRouter
        logging.info("Creating new brain for bootstrapping")
        router = OmniRouter(use_bpe=False)
        
    ssm = get_ssm()
    ssm.train()
    
    optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)
    
    logging.info("Downloading/Loading databricks-dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{sample_size}]")
    
    total_turns = 0
    total_loss = 0.0
    
    for i, item in enumerate(dataset):
        user_turn = item["instruction"].strip()
        assistant_turn = item["response"].strip()
        
        # Format sequence
        tokens = ["<|user|>"] + router.tokenizer.tokenize(user_turn.split(), is_inference=False) + \
                 ["<|assistant|>"] + router.tokenizer.tokenize(assistant_turn.split(), is_inference=False)
        
        # Train deterministic trie
        router.stream(tokens)
        
        # Train SSM Parametric Network
        optimizer.zero_grad()
        # We train dynamics to predict the sequence, and value to be 1.0 (good conversation)
        v_loss = ssm.update_value(tokens, reward=1.0)
        d_loss = ssm.train_dynamics(tokens)
        loss = v_loss + d_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_turns += 1
        
        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i+1} dialogs. Avg Loss: {total_loss/total_turns:.4f}")
            
    logging.info("Bootstrapping complete. Saving brain and SSM weights...")
    with open(brain_path, "wb") as f:
        pickle.dump(router, f)
    # SSM weights are auto-saved by its internal logic when we call update_value/train_dynamics,
    # but let's make sure by saving explicitly if needed, although it saves on __del__ or inside.
    # We can explicitly save by instantiating ssm and letting it save.
    torch.save(ssm.state_dict(), "ssm_dynamics.pt")
    logging.info("Saved ssm_dynamics.pt. Uchi is ready for interaction.")

if __name__ == "__main__":
    bootstrap_agent(sample_size=200)

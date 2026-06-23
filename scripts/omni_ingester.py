import os
import sys
import yaml
import requests
import tempfile
from tqdm import tqdm

from uchi.omni_router import OmniRouter
from uchi.cli import save_brain, load_brain

CHATTERBOT_CATEGORIES = [
    "ai", "botprofile", "computers", "conversations", "emotion", "food",
    "gossip", "greetings", "health", "history", "humor", "literature",
    "money", "movies", "politics", "psychology", "science", "sports", "trivia"
]

def download_chatterbot_corpus():
    """Downloads the chatterbot conversational dataset from GitHub."""
    print("[*] Downloading Conversational Dataset (Chatterbot Corpus)...")
    dataset = []
    base_url = "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/{}.yml"
    
    for category in tqdm(CHATTERBOT_CATEGORIES, desc="Fetching Categories"):
        try:
            url = base_url.format(category)
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = yaml.safe_load(resp.text)
                if data and "conversations" in data:
                    dataset.extend(data["conversations"])
        except Exception as e:
            print(f"[-] Failed to download {category}: {e}")
            
    print(f"[+] Downloaded {len(dataset)} conversation threads.")
    return dataset

def ingest_dataset(router, dataset):
    """
    Streams the dataset into the OmniRouter.
    Replaces the tiny persona.txt bottleneck with massive automated ingestion.
    """
    print("[*] Streaming massive conversational dataset into Deterministic Trie...")
    
    # We flatten the conversations into "<|user|> X <|assistant|> Y" format
    for thread in tqdm(dataset, desc="Ingesting Conversations"):
        if not thread or len(thread) < 2:
            continue
            
        # We pair them up: user -> assistant
        for i in range(len(thread) - 1):
            user_msg = str(thread[i]).strip()
            assistant_msg = str(thread[i+1]).strip()
            
            if not user_msg or not assistant_msg:
                continue
                
            sequence = ["<|user|>"] + user_msg.split() + ["<|assistant|>"] + assistant_msg.split() + ["<|end|>"]
            router.stream(sequence)
            
    print("[+] Ingestion complete. Knowledge graph expanded.")

if __name__ == "__main__":
    brain_path = "brain.uchi"
    
    if os.path.exists(brain_path):
        router = load_brain(brain_path)
    else:
        print("[*] Initializing fresh OmniRouter for ingestion...")
        router = OmniRouter(use_bpe=True, memory_window=5)
        
    dataset = download_chatterbot_corpus()
    ingest_dataset(router, dataset)
    
    save_brain(router, brain_path)

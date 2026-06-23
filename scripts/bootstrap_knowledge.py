#!/usr/bin/env python3
"""
Root Problem 4: Volume Ingestion via HuggingFace Datasets
Streams raw knowledge into Uchi's neuro-symbolic engine as a second pass.
"""

import sys
import os
import argparse

def stream_huggingface(dataset_name: str, split: str = "train", limit: int = 1000,
                       brain_path: str = "brain.uchi"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install 'datasets': pip install datasets")
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from uchi.cli import load_brain, save_brain
    from uchi.omni_router import OmniRouter

    # Load existing brain so we don't overwrite prior knowledge
    print(f"Loading brain from {brain_path}...")
    router = load_brain(brain_path)
    if router is None:
        print("No existing brain found — creating fresh OmniRouter.")
        router = OmniRouter(use_bpe=False)

    print(f"Streaming dataset: {dataset_name} ({split}), limit={limit}")
    # wikipedia dataset requires a config name in newer datasets versions
    if dataset_name == "wikipedia":
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=split, streaming=True)
    elif dataset_name == "code_search_net":
        dataset = load_dataset("code-search-net/code_search_net", "python", split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    count = 0
    for item in dataset:
        if count >= limit:
            break

        if dataset_name == "wikipedia":
            text = item.get("text", "")
        elif dataset_name == "code_search_net":
            text = item.get("func_documentation_string", "") + " " + item.get("func_code_string", "")
        else:
            text = str(item)

        if not text.strip():
            continue

        tokens = text.split()
        if len(tokens) > 500:
            tokens = tokens[:500]

        router.stream(tokens)
        count += 1

        if count % 100 == 0:
            print(f"Ingested {count}/{limit} records from {dataset_name}...")

    print(f"Completed! Ingested {count} records. Saving brain to {brain_path}...")
    save_brain(router, brain_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap volume knowledge.")
    parser.add_argument("--dataset", type=str, default="wikipedia", help="Dataset name: wikipedia, code_search_net, etc.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=1000, help="Number of records to ingest")
    parser.add_argument("--brain", type=str, default="brain.uchi", help="Path to brain.uchi file")

    args = parser.parse_args()
    stream_huggingface(args.dataset, args.split, args.limit, args.brain)

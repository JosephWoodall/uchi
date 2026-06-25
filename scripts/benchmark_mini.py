import sys
import os
import glob
import time
import json
from uchi.cli import load_brain

MMLU_QUESTIONS = [
    {
        "q": "The first law of thermodynamics states that:",
        # Accept any common phrasing — DDG and textbooks phrase this variously
        "a": ["energy cannot be created or destroyed",
              "energy can neither be created nor destroyed",
              "conservation of energy",
              "created nor destroyed",
              "created or destroyed"]
    },
    {
        "q": "In Python, which keyword is used to handle exceptions?",
        "a": "except"
    }
]

SWE_QUESTIONS = [
    {
        "q": "Fix this python function so it returns the sum of a and b: def add(a, b): return a - b",
        "a": "a + b"
    },
    {
        "q": "Write a python loop that prints 0 through 4.",
        "a": "range(5)"
    }
]

def run_benchmark(questions, name):
    print(f"\n[*] Booting Uchi for {name} mini-benchmark...")
    
    # This single call will auto-trigger the Universal Builder if brain.uchi is missing!
    router = load_brain("brain.uchi")
    if router is None:
        print("[-] Brain missing, creating a fresh one.")
        from uchi.omni_router import OmniRouter
        router = OmniRouter(use_bpe=False, memory_window=5)
        
    score = 0
    total = len(questions)
    
    start_time = time.time()
    
    for i, q in enumerate(questions):
        print(f"\n--- {name} Question {i+1}/{total} ---")
        print(f"Prompt: {q['q']}")
        
        try:
            # We measure latency
            t0 = time.time()
            reply = router.chat(q['q'])
            latency = time.time() - t0
            
            print(f"Uchi: {reply}")
            print(f"Latency: {latency:.2f} seconds")
            
            # Support both single string and list of acceptable phrasings
            accepted = q['a'] if isinstance(q['a'], list) else [q['a']]
            if any(a.lower() in reply.lower() for a in accepted):
                print("[+] Result: PASS")
                score += 1
            else:
                print("[-] Result: FAIL")
                print(f"    Expected one of: {[a[:40] for a in accepted]}")
                
        except Exception as e:
            print(f"[-] CRITICAL ENGINE CRASH: {e}")
            
    total_time = time.time() - start_time
    print(f"\n======================================")
    print(f"{name} Benchmark Complete")
    print(f"Score: {score}/{total} ({(score/total)*100:.1f}%)")
    print(f"Total Time: {total_time:.2f}s")
    print(f"======================================")


def main():
    args = sys.argv[1:]
    if not args or not any(x in args for x in ["--mmlu", "--swe"]):
        print("Usage: python benchmark_mini.py [--mmlu | --swe] [--wipe]")
        sys.exit(1)

    if "--wipe" in args:
        print("[*] --wipe flag detected. Deleting brain.uchi to trigger Universal Rebuild...")
        if os.path.exists("brain.uchi"):
            os.remove("brain.uchi")

    if "--mmlu" in args:
        run_benchmark(MMLU_QUESTIONS, "MMLU")
    elif "--swe" in args:
        run_benchmark(SWE_QUESTIONS, "SWE-Bench")

if __name__ == "__main__":
    main()

import os
import sys

# Change working directory to the workspace so imports work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from uchi.omni_router import OmniRouter

def main():
    print("Initializing OmniRouter...")
    router = OmniRouter(use_bpe=False)
    
    print("\n--- Test 1: Intent Routing & Trie Generation ---")
    response1 = router.chat("write a python script")
    print(f"Uchi: {response1}")
    
    print("\n--- Test 2: Sentiment Training (GRPO) ---")
    response2 = router.chat("good job")
    print(f"Uchi: {response2}")
    
    print("\n--- Test 3: Novel Query ---")
    response3 = router.chat("what is the capital of France")
    print(f"Uchi: {response3}")

if __name__ == "__main__":
    main()

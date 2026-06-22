import argparse
import sys
import os
from .omni_router import OmniRouter

def ingest_file(router, filepath):
    """Injects massive context from a file into the OmniRouter."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return
        
    print(f"[*] Ingesting massive context from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        # Simple word tokenization for the demo
        text = f.read().split()
        
    router.stream(text)
    print(f"[+] Successfully injected {len(text)} tokens into the Deterministic LLM.")

def interactive_chat():
    """Starts an interactive looping terminal UI."""
    print("===============================================================")
    print(" Uchi v0.2.0 - The Multi-Modal Deterministic LLM")
    print("===============================================================")
    print(" Commands:")
    print("   /load <filepath>   - Ingest a file to build context")
    print("   /predict <steps>   - Predict the future based on context")
    print("   /query <question>  - Zero-shot ad-hoc retrieval")
    print("   /quit              - Exit")
    print(" Anything else will be streamed into the engine as context.")
    print("===============================================================")
    
    router = OmniRouter(use_bpe=True, memory_window=5)
    
    while True:
        try:
            cmd = input("\nuchi> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not cmd:
            continue
            
        if cmd.lower() in ["/quit", "/exit"]:
            break
            
        if cmd.startswith("/load "):
            filepath = cmd.split(" ", 1)[1]
            ingest_file(router, filepath)
            
        elif cmd.startswith("/query "):
            q = cmd.split(" ", 1)[1]
            ans = router.query(q.split())
            print(f"-> {ans}")
            
        elif cmd.startswith("/predict "):
            try:
                parts = cmd.split(" ")
                steps = int(parts[1]) if len(parts) > 1 else 5
                pred = router.predict_future([], steps=steps)
                print(f"-> {' '.join(pred)}")
            except Exception as e:
                print(f"Prediction Error: {e}")
                
        else:
            # Treat as standard streaming context
            tokens = cmd.split()
            router.stream(tokens)
            print(f"[+] Streamed {len(tokens)} concepts.")

def debate_loop(topic: str, rounds: int = 10):
    """
    Spawns two OmniRouter agents and forces them to predict/debate 
    continuously over the same topic.
    """
    print("===============================================================")
    print(" Uchi v0.2.0 - Multi-Agent Deterministic Debate")
    print(f" Topic: {topic}")
    print(" Engine: OmniRouter (BPE Stream Compression ENABLED)")
    print("===============================================================")
    
    # Initialize two separate deterministic brains
    # They both use BPE compression to prevent infinite RAM explosion
    agent_alpha = OmniRouter(use_bpe=True, memory_window=5)
    agent_beta  = OmniRouter(use_bpe=True, memory_window=5)
    
    # Seed their initial context
    seed_tokens = topic.split()
    agent_alpha.stream(seed_tokens)
    agent_beta.stream(seed_tokens)
    
    current_context = seed_tokens
    
    for i in range(rounds):
        import time
        time.sleep(1) # Slow down for readability
        
        print(f"\n--- Round {i+1} ---")
        
        # Alpha speaks
        alpha_reply = agent_alpha.predict_future(current_context, steps=5)
        print(f"Agent Alpha: {' '.join(alpha_reply)}")
        
        # Stream Alpha's reply into Beta's brain so Beta hears it
        agent_beta.stream(alpha_reply)
        
        # Beta speaks
        beta_reply = agent_beta.predict_future(alpha_reply, steps=5)
        print(f"Agent Beta:  {' '.join(beta_reply)}")
        
        # Stream Beta's reply back into Alpha's brain
        agent_alpha.stream(beta_reply)
        
        current_context = beta_reply
        
    print("\n[+] Debate concluded. Both agents successfully compressed the entire context history via BPE.")

def main():
    parser = argparse.ArgumentParser(description="Uchi Deterministic LLM CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    chat_parser = subparsers.add_parser("chat", help="Start the interactive Deterministic LLM chat loop")
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file to build massive context")
    ingest_parser.add_argument("filepath", type=str, help="Path to the file to ingest")
    
    debate_parser = subparsers.add_parser("debate", help="Spawn two OmniRouter agents to debate infinitely")
    debate_parser.add_argument("topic", type=str, help="The initial context/topic to debate")
    debate_parser.add_argument("--rounds", type=int, default=10, help="Number of debate rounds")
    
    args = parser.parse_args()
    
    if args.command == "chat":
        interactive_chat()
    elif args.command == "ingest":
        router = OmniRouter()
        ingest_file(router, args.filepath)
        print("Done. Use 'uchi chat' and the '/load' command to interact with context dynamically.")
    elif args.command == "debate":
        debate_loop(args.topic, args.rounds)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

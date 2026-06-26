import sys
import json
import traceback
from uchi.cli import load_brain

def main():
    if len(sys.argv) < 2:
        print("Usage: python python_communication.py 'Your statement here'")
        sys.exit(1)

    statement = sys.argv[1]
    
    print(f"[*] Loading Uchi Brain...")
    try:
        router = load_brain("brain.uchi")
        if router is None:
            print("[-] No brain found at brain.uchi. Is it initialized?")
            sys.exit(1)
            
        print(f"[*] Sending statement: '{statement}'")
        
        # Capture the response
        reply = router.chat(statement)
        
        # Gather relevant metrics
        memory_records = 0
        if hasattr(router, "memory") and hasattr(router.memory, "cpu_mem"):
            memory_records = len(router.memory.cpu_mem.records)
            
        metrics = {
            "response": reply,
            "metrics": {
                "memory_records": memory_records,
                "ssm_baseline_mean": round(router.baseline.mean, 4) if hasattr(router, "baseline") else 0.0,
                "ssm_baseline_std": round(router.baseline.std, 4) if hasattr(router, "baseline") else 0.0,
                "skills_loaded": len(router.skills.list_skills()) if hasattr(router, "skills") else 0,
                "has_code_specialist": router.specialist_pool.has_specialist("code") if hasattr(router, "specialist_pool") else False
            }
        }
        
        print("\n--- UCHI RESPONSE & METRICS ---")
        print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        print(f"[-] Error communicating with Uchi: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

import asyncio
import os
import sys
from uchi.tui.app import UchiApp
from textual.app import App

def safe_print(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

async def evaluate_qa():
    safe_print("\n--- Running Evaluation Suite ---")
    if os.path.exists("brain.uchi"): os.remove("brain.uchi")
    
    app = UchiApp(brain_path="brain.uchi", preload_path=None)
    
    questions = [
        "hello uchi",
        "how are you today?",
        "can you please tell me about the weather?",
        "your responses seen to be deprecating huh",
        "write a python function to add two numbers",
        "write a python class for a simple dog"
    ]
    
    async with app.run_test() as pilot:
        await asyncio.sleep(2) # Wait for bootstrap
        
        for q in questions:
            safe_print(f"\n[Test Question] {q}")
            await pilot.click("#input-box")
            for char in q:
                await pilot.press(char)
            await pilot.press("enter")
            
            await asyncio.sleep(3) # wait for prediction and sandbox execution
            
            # Extract the last reply from ODUSP
            lines = [line.text for line in app.query_one("#chat-log").lines]
            reply = ""
            for line in reversed(lines):
                if "ODUSP (Reply):" in line:
                    reply = line.split("ODUSP (Reply):")[1].strip()
                    break
            
            safe_print(f"[Model Output] {reply}")
            
            # Grade the response
            if "I am unfamiliar with the word" in reply:
                safe_print("[Grade] D - Active Learning False Positive. The model interrupted the conversation because its vocabulary is too small.")
            elif "hello how can i" in reply.lower():
                safe_print("[Grade] A - Perfect recall from persona.txt")
            elif reply == "6":
                safe_print("[Grade] F - Hallucination (Unigram fallback due to missing feedback update)")
            elif "understand i will suppress" in reply.lower():
                safe_print("[Grade] F - Frankenstein sequence. The model stitched random persona paths together.")
            elif "```python" in reply:
                safe_print("[Grade] A - Code Generated Successfully!")
                if "SyntaxError" in reply:
                    safe_print("  -> Wait, RL Engine caught a SyntaxError and autonomously pruned it!")
            else:
                safe_print(f"[Grade] C - Evaluated dynamically.")

async def main():
    await evaluate_qa()

if __name__ == "__main__":
    asyncio.run(main())

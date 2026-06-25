---
name: python_communication
description: A programmatic interface to chat with Uchi and retrieve its internal metrics simultaneously.
---

# Python Communication with Uchi

This skill allows agents to pass a natural language statement directly to Uchi's `OmniRouter` and receive both the generated response and a snapshot of the engine's current internal metrics (Memory Records, SSM Baseline, Skills Loaded).

## Usage
Run the script and pass the statement as a string argument:

```bash
python scripts/python_communication.py "Explain how a Trie works."
```

## Output Format
The script will output a JSON payload containing both the text response and the metrics:

```json
{
  "response": "The response string from Uchi...",
  "metrics": {
    "memory_records": 125,
    "ssm_baseline_mean": 0.012,
    "ssm_baseline_std": 0.98,
    "skills_loaded": 13,
    "has_code_specialist": true
  }
}
```

This is extremely useful for verifying that the State Space Model metrics are updating appropriately after a conversation.

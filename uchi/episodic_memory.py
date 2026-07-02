import time
from typing import List, Dict

class EpisodicMemory:
    """
    Episodic Memory for Uchi. 
    Stores conversational history (User and Uchi exchanges) to maintain 
    context across sessions without polluting the factual Semantic Index.
    """
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_interaction(self, user_msg: str, uchi_reply: str) -> None:
        """Log a single turn of conversation."""
        self.history.append({
            "timestamp": time.time(),
            "user": user_msg,
            "uchi": uchi_reply
        })
        # Prune to keep only recent history
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_string(self, n_turns: int = 3) -> str:
        """Format the most recent history as a context string for FLUX."""
        recent = self.history[-n_turns:]
        if not recent:
            return ""
        
        lines = ["--- Recent Conversation Context ---"]
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Uchi: {turn['uchi']}")
        lines.append("-----------------------------------")
        return "\n".join(lines)

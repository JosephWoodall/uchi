"""
ontology.py
===========
Native Ontological Semantic Mapping for Uchi.

This dictionary maps conversational slang, synonyms, and variations
to the canonical words used in `persona.txt`. This allows the deterministic
trie to handle massive combinatorial variance in human speech without
needing a massive pre-trained brain or an LLM parser.
"""

ONTOLOGY = {
    # Greetings
    "hi": "hello",
    "hey": "hello",
    "greetings": "hello",
    "sup": "hello",
    "yo": "hello",
    "howdy": "hello",
    "hiya": "hello",
    "morning": "hello",
    "afternoon": "hello",
    "evening": "hello",
    "salutations": "hello",
    
    # Gratitude
    "thx": "thanks",
    "ty": "thanks",
    "appreciate": "thanks",
    "grateful": "thanks",
    "cheers": "thanks",
    "thankyou": "thanks",
    "thnx": "thanks",
    
    # Farewells
    "bye": "goodbye",
    "cya": "goodbye",
    "peace": "goodbye",
    "farewell": "goodbye",
    "later": "goodbye",
    "adios": "goodbye",
    "night": "goodbye",
    "goodnight": "goodbye",
    
    # Actions & Verbs
    "assist": "help",
    "aid": "help",
    "support": "help",
    "guide": "help",
    
    "explain": "tell",
    "describe": "tell",
    "elaborate": "tell",
    "clarify": "tell",
    "show": "tell",
    
    "define": "what",
    "meaning": "what",
    
    "teach": "learn",
    "study": "learn",
    "understand": "learn",
    "grasp": "learn",
    
    # Identity & Creator
    "purpose": "do",
    "capabilities": "do",
    "function": "do",
    "features": "do",
    "abilities": "do",
    
    "make": "created",
    "build": "created",
    "built": "created",
    "creator": "joseph",
    "maker": "joseph",
    "author": "joseph",
    "developer": "joseph",
    
    "bot": "uchi",
    "ai": "uchi",
    "assistant": "uchi",
    "system": "uchi",
    "program": "uchi",
    
    # Conversational Fillers & Pronouns
    "ya": "you",
    "u": "you",
    "ur": "your",
    "yours": "your",
    
    "im": "i",
    "id": "i",
    "ive": "i",
    
    "yeah": "yes",
    "yep": "yes",
    "yup": "yes",
    "sure": "yes",
    "ok": "yes",
    "okay": "yes",
    "alright": "yes",
    "definitely": "yes",
    "absolutely": "yes",
    "y": "yes",
    
    "nah": "no",
    "nope": "no",
    "never": "no",
    "n": "no",
    
    # Technical Concepts
    "smart": "intelligence",
    "clever": "intelligence",
    "intelligent": "intelligence",
    
    "brain": "memory",
    "mind": "memory",
    "storage": "memory",
    
    "model": "trie",
    "network": "trie",
    "algorithm": "trie",
    "structure": "trie",
    "graph": "trie",
    
    "llm": "llms",
    "chatgpt": "llms",
    "gpt": "llms",
    "claude": "llms",
    "gemini": "llms",
}

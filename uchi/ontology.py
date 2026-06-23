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
    # ── GREETINGS ──
    "hi": "hello", "hey": "hello", "greetings": "hello", "sup": "hello", "yo": "hello",
    "howdy": "hello", "hiya": "hello", "morning": "hello", "afternoon": "hello", 
    "evening": "hello", "salutations": "hello", "ello": "hello", "heya": "hello",
    "aloha": "hello", "hola": "hello", "bonjour": "hello", "hallo": "hello",
    "welcome": "hello",

    # ── GRATITUDE ──
    "thx": "thanks", "ty": "thanks", "appreciate": "thanks", "grateful": "thanks",
    "cheers": "thanks", "thankyou": "thanks", "thnx": "thanks", "tysm": "thanks",
    "gracias": "thanks", "merci": "thanks", "danke": "thanks", "arigato": "thanks",
    
    # ── FAREWELLS ──
    "bye": "goodbye", "cya": "goodbye", "peace": "goodbye", "farewell": "goodbye",
    "later": "goodbye", "adios": "goodbye", "night": "goodbye", "goodnight": "goodbye",
    "ciao": "goodbye", "sayonara": "goodbye", "ttyl": "goodbye", "brb": "goodbye",
    "leaving": "goodbye", "departing": "goodbye", "quit": "goodbye", "exit": "goodbye",
    
    # ── ACTIONS / VERBS ──
    # Help
    "assist": "help", "aid": "help", "support": "help", "guide": "help", "rescue": "help",
    "save": "help", "serve": "help", "facilitate": "help", "accommodate": "help",
    
    # Tell/Explain
    "explain": "tell", "describe": "tell", "elaborate": "tell", "clarify": "tell",
    "show": "tell", "instruct": "tell", "demonstrate": "tell",
    "illustrate": "tell", "reveal": "tell", "inform": "tell", "notify": "tell",
    "communicate": "tell", "state": "tell", "express": "tell", "say": "tell",
    "speak": "tell", "articulate": "tell", "outline": "tell", "summarize": "tell",
    
    # What/Define (Inquiries)
    "define": "what", "meaning": "what", "who": "what", "when": "what", 
    "which": "what", "where": "what", "why": "what", "whose": "what",
    
    # Learn/Understand
    "study": "learn", "understand": "learn", "grasp": "learn", "comprehend": "learn",
    "master": "learn", "absorb": "learn", "acquire": "learn", "discover": "learn",
    "realize": "learn", "recognize": "learn",
    
    # Do/Purpose
    "purpose": "do", "capabilities": "do", "function": "do", "features": "do",
    "abilities": "do", "perform": "do", "execute": "do", "achieve": "do", 
    "accomplish": "do", "act": "do", "operate": "do", "tasks": "do", "skills": "do",
    
    # Create
    "make": "created", "build": "created", "built": "created", "produce": "created",
    "generate": "created", "construct": "created", "fabricate": "created",
    "design": "created", "invent": "created", "develop": "created", "formulate": "created",
    "originate": "created", "establish": "created",
    
    # ── IDENTITY / ENTITIES ──
    # Joseph Woodall
    "creator": "joseph", "maker": "joseph", "author": "joseph", "developer": "joseph",
    "programmer": "joseph", "engineer": "joseph", "architect": "joseph",
    "founder": "joseph", "inventor": "joseph", "father": "joseph",
    
    # Uchi
    "bot": "uchi", "ai": "uchi", "assistant": "uchi", "system": "uchi", 
    "program": "uchi", "software": "uchi", "application": "uchi", "app": "uchi",
    "tool": "uchi", "machine": "uchi", "robot": "uchi", "computer": "uchi",
    "algorithm": "uchi",
    
    # Intelligence
    "smart": "intelligence", "clever": "intelligence", "intelligent": "intelligence",
    "brilliant": "intelligence", "genius": "intelligence", "conscious": "intelligence",
    "sentient": "intelligence", "aware": "intelligence", "thinking": "intelligence",
    "cognition": "intelligence", "wisdom": "intelligence", "knowledge": "intelligence",
    
    # Memory
    "brain": "memory", "mind": "memory", "storage": "memory", "database": "memory",
    "recall": "memory", "remember": "memory", "retention": "memory", "cache": "memory",
    
    # Trie / Geometric
    "model": "trie", "network": "trie", "structure": "trie", "graph": "trie",
    "tree": "trie", "geometry": "geometric", "math": "geometric", "mathematics": "geometric",
    "mathematical": "geometric", "calculus": "geometric", "algebra": "geometric",
    "topology": "geometric",
    
    # LLMs
    "llm": "llms", "chatgpt": "llms", "gpt": "llms", "claude": "llms", 
    "gemini": "llms", "llama": "llms", "transformers": "llms", "neural": "llms",
    "nn": "llms", "deeplearning": "llms",
    
    # Sequence/Prediction
    "patterns": "sequence", "series": "sequence", "order": "sequence", 
    "chain": "sequence", "progression": "sequence", "succession": "sequence",
    "predict": "prediction", "forecast": "prediction", "guess": "prediction",
    "anticipate": "prediction", "foresee": "prediction", "project": "prediction",
    "estimate": "prediction",
    
    # ── CONVERSATIONAL FILLERS & PRONOUNS ──
    # You
    "ya": "you", "u": "you", "thou": "you", "yall": "you", "ye": "you",
    "ur": "your", "yours": "your", "thy": "your", "thine": "your",
    
    # I
    "im": "i", "id": "i", "ive": "i", "me": "i", "my": "i", "mine": "i", "myself": "i",
    
    # Yes
    "yeah": "yes", "yep": "yes", "yup": "yes", "sure": "yes", "ok": "yes", 
    "okay": "yes", "alright": "yes", "definitely": "yes", "absolutely": "yes", 
    "y": "yes", "certainly": "yes", "indeed": "yes", "affirmative": "yes",
    "agreed": "yes", "roger": "yes", "cool": "yes", "fine": "yes", "gladly": "yes",
    "course": "yes", "always": "yes",
    
    # No
    "nah": "no", "nope": "no", "never": "no", "n": "no", "negative": "no",
    "nay": "no", "false": "no", "incorrect": "no", "wrong": "no", "disagree": "no",
    "deny": "no", "refuse": "no", "reject": "no",
    
    # Unknowns
    "idk": "what", "dunno": "what", "maybe": "what", "perhaps": "what", 
    "possibly": "what", "unsure": "what", "confused": "what", "lost": "what",
}

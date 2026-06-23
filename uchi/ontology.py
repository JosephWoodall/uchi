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
    
    # Gratitude
    "thx": "thanks",
    "ty": "thanks",
    "appreciate": "thanks",
    "thx!": "thanks",
    "ty!": "thanks",
    
    # Farewells
    "bye": "goodbye",
    "cya": "goodbye",
    "peace": "goodbye",
    "farewell": "goodbye",
    
    # Actions & Verbs
    "assist": "help",
    "aid": "help",
    "support": "help",
    
    "explain": "tell",
    "describe": "tell",
    "define": "what",
    
    "teach": "learn",
    
    # Identity & Metaphysics
    "purpose": "do",
    "capabilities": "do",
    "who": "what",
}

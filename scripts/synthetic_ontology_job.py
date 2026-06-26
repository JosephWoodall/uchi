import os
import re
import sys
from pathlib import Path

# Add the parent directory to the path so we can import uchi
sys.path.insert(0, str(Path(__file__).parent.parent))

from uchi.ontology_manager import OntologyManager

def generate_typos(word: str) -> set[str]:
    """Deterministically generates common typographical errors for a given word."""
    typos = set()
    if len(word) < 4:
        return typos
        
    # 1. Dropped vowels
    for i, c in enumerate(word):
        if c in 'aeiou':
            typos.add(word[:i] + word[i+1:])
            
    # 2. Transposed adjacent letters
    for i in range(len(word) - 1):
        typos.add(word[:i] + word[i+1] + word[i] + word[i+2:])
        
    # 3. Accidental double letters
    for i in range(len(word)):
        typos.add(word[:i] + word[i] * 2 + word[i+1:])
        
    return typos

def generate_synonyms(word: str) -> set[str]:
    """Extracts conservative synonyms using NLTK WordNet."""
    try:
        from nltk.corpus import wordnet
        try:
            wordnet.ensure_loaded()
        except Exception:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
        synonyms = set()
        synsets = wordnet.synsets(word)
        if not synsets:
            return synonyms
            
        # Only use the top synset to remain highly conservative
        top_synset = synsets[0]
        for lemma in top_synset.lemmas():
            syn = lemma.name().lower().replace('_', ' ')
            # Only accept single-word synonyms for simple 1-to-1 mapping
            if ' ' not in syn and syn != word:
                synonyms.add(syn)
        return synonyms
    except ImportError:
        print("[!] NLTK not installed. Skipping synonym generation.")
        return set()

def main():
    print("==================================================")
    print(" Synthetic Ontology Data Generation Job")
    print("==================================================")
    
    # 1. Load the Core Vocabulary
    persona_path = Path(__file__).parent.parent / "uchi" / "persona.txt"
    if not persona_path.exists():
        print(f"[-] Could not find persona file at {persona_path}")
        return
        
    with open(persona_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
        
    core_vocab = set(re.findall(r'\b[a-z]+\b', text))
    print(f"[*] Extracted {len(core_vocab)} core words from persona.txt")
    
    # 2. Load the Ontology Manager
    manager = OntologyManager("ontology.json")
    initial_count = len(manager.mapping)
    print(f"[*] Loaded initial ontology with {initial_count} mappings")
    
    # 3. Generate Data
    new_mappings = 0
    
    for word in core_vocab:
        # Generate Typos
        typos = generate_typos(word)
        for typo in typos:
            # CRITICAL: Prevent branch collisions! Never map a word that is already in the core vocab.
            if typo not in core_vocab and typo not in manager.mapping:
                manager.add_mapping(typo, word)
                new_mappings += 1
                
        # Generate Synonyms
        syns = generate_synonyms(word)
        for syn in syns:
            if syn not in core_vocab and syn not in manager.mapping:
                manager.add_mapping(syn, word)
                new_mappings += 1
                
    print(f"[+] Successfully generated {new_mappings} new synthetic mappings.")
    print(f"[+] Ontology size is now {len(manager.mapping)} mappings.")

if __name__ == "__main__":
    main()

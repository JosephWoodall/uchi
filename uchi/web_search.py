import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

import spacy

# Lazy load spacy model
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Performs a DuckDuckGo HTML search and extracts SVO triples using SpaCy.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all('a', class_='result__snippet')
        
        extracted_text = []
        for i, res in enumerate(results):
            if i >= max_results:
                break
            text = res.get_text(separator=" ", strip=True)
            extracted_text.append(text)
            
        if not extracted_text:
            return ""

        combined = " ".join(extracted_text).lower()

        # SpaCy SVO triples augment but do not replace the raw text.
        # Passive-voice facts ("energy cannot be created or destroyed") have no
        # SVO subject and are silently dropped by the dependency parser, causing
        # >90% information loss on exactly the factual sentences we need most.
        try:
            nlp = get_nlp()
            doc = nlp(combined)
            triples = []
            for sentence in doc.sents:
                subject = None
                verb = None
                obj = None
                for token in sentence:
                    if "subj" in token.dep_:
                        subject = token.text.lower()
                    elif "obj" in token.dep_:
                        obj = token.text.lower()
                    elif token.pos_ == "VERB":
                        verb = token.lemma_.lower()
                if subject and verb and obj:
                    triples.append(f"{subject} {verb} {obj}")
            if triples:
                combined = combined + " " + " ".join(triples)
        except Exception:
            pass

        return combined
        
    except Exception as e:
        return ""

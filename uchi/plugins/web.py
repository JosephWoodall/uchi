import requests
from bs4 import BeautifulSoup
import re

def fetch_web_context(query: str) -> str:
    """
    Autonomous Web Sourcing Hook.
    If the ODUSP has a gap in its geometric trie, it fires this hook
    to scrape Wikipedia for the missing concept, returns the plain text,
    which is then fed directly into the BPE stream compressor.
    """
    print(f"\n[!] Knowledge gap detected for '{query}'. Engaging Autonomous Web Sourcing...")
    try:
        # We use a simple Wikipedia search as a safe, deterministic source of truth.
        url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            print(f"[-] Web Sourcing failed: No deterministic truth found for '{query}'.")
            return ""
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the main paragraphs
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        
        # Clean up citations like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Limit to first 2000 characters to prevent huge BPE blasts on simple queries
        summary = text[:2000].strip()
        print(f"[+] Sourced {len(summary)} bytes of structural truth from the web.")
        return summary
    except Exception as e:
        print(f"[-] Web Sourcing error: {e}")
        return ""

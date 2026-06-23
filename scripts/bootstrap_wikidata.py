import os
import sys
import logging
import pickle
import wikipedia
import spacy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_triples(text):
    nlp = get_nlp()
    doc = nlp(text)
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
            triples.append([subject, verb, obj])
    return triples

def bootstrap_wikidata(brain_path="brain.uchi"):
    logging.info("Starting Wikipedia Fact Bootstrapping")

    from uchi.cli import load_brain, save_brain
    from uchi.omni_router import OmniRouter

    router = load_brain(brain_path)
    if router is None:
        logging.info("Creating new brain for bootstrapping")
        router = OmniRouter(use_bpe=False)
        
    from uchi.neuro_symbolic import get_ssm
    ssm = get_ssm()
    ssm.train()
    
    import torch
    optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)
    
    topics = [
        "Paris", "France", "Earth", "Sun", "Moon", "Albert Einstein", 
        "Isaac Newton", "Quantum mechanics", "Computer", "Internet",
        "Python (programming language)", "Artificial intelligence", "World War II",
        "Mathematics", "Physics", "Biology", "Chemistry", "Oxygen", "Water",
        "DNA", "United States", "New York City", "London", "Tokyo", "Rome"
    ]
    
    facts_processed = 0
    
    for topic in topics:
        try:
            logging.info(f"Fetching summary for {topic}")
            summary = wikipedia.summary(topic, sentences=3)
            triples = extract_triples(summary)
            
            for subject, verb, obj in triples:
                user_prompt = f"Tell me about {subject} and {obj}."
                assistant_response = f"{subject} {verb} {obj}."
                
                tokens = ["<|user|>"] + router.tokenizer.tokenize(user_prompt.split(), is_inference=False) + \
                         ["<|assistant|>"] + router.tokenizer.tokenize(assistant_response.split(), is_inference=False)
                
                router.stream(tokens)
                
                optimizer.zero_grad()
                v_loss = ssm.update_value(tokens, reward=1.0)
                d_loss = ssm.train_dynamics(tokens)
                loss = v_loss + d_loss
                loss.backward()
                optimizer.step()
                
                facts_processed += 1
                
        except Exception as e:
            logging.warning(f"Failed to process topic {topic}: {e}")
            
    logging.info(f"Finished Wikipedia bootstrap. Processed {facts_processed} facts.")
    save_brain(router, brain_path)
    torch.save(ssm.state_dict(), "ssm_dynamics.pt")
    logging.info("World knowledge ingested.")

def run(router, progress_callback=None):
    """
    Ingests Wikipedia fact triples into the trie.
    Accepts an existing router instance — does not load or save brain.uchi.
    """
    from uchi.neuro_symbolic import get_ssm
    import torch

    ssm = get_ssm()
    ssm.train()
    optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)

    topics = [
        "Paris", "France", "Earth", "Sun", "Moon", "Albert Einstein",
        "Isaac Newton", "Quantum mechanics", "Computer", "Internet",
        "Python (programming language)", "Artificial intelligence", "World War II",
        "Mathematics", "Physics", "Biology", "Chemistry", "Oxygen", "Water",
        "DNA", "United States", "New York City", "London", "Tokyo", "Rome"
    ]

    total = len(topics)

    for idx, topic in enumerate(topics):
        try:
            summary = wikipedia.summary(topic, sentences=3)
            triples = extract_triples(summary)

            for subject, verb, obj in triples:
                user_prompt = f"Tell me about {subject} and {obj}."
                assistant_response = f"{subject} {verb} {obj}."

                tokens = ["<|user|>"] + router.tokenizer.tokenize(user_prompt.split(), is_inference=False) + \
                         ["<|assistant|>"] + router.tokenizer.tokenize(assistant_response.split(), is_inference=False)

                router.stream(tokens)

                optimizer.zero_grad()
                v_loss = ssm.update_value(tokens, reward=1.0)
                d_loss = ssm.train_dynamics(tokens)
                loss = v_loss + d_loss
                loss.backward()
                optimizer.step()

        except Exception:
            pass

        if progress_callback:
            progress_callback(idx + 1, total)


if __name__ == "__main__":
    bootstrap_wikidata()

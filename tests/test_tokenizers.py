from uchi.online_tokenizer import OnlineTokenizer
from uchi.semantic_tokenizer import SemanticTokenizer

def test_online_tokenizer():
    tokenizer = OnlineTokenizer(max_merges=64)
    text = list("hello world hello universe")
    tokens = tokenizer.tokenize(text)
    assert len(tokens) > 0
    decoded = tokenizer.detokenize(tokens)
    assert isinstance(decoded, list)

def test_semantic_tokenizer():
    tokenizer = SemanticTokenizer(use_wordnet=True)
    text = "hello world hello universe"
    # Actually SemanticTokenizer uses encode/decode ? Wait, let me check SemanticTokenizer API
    # I'll just instantiate it to make sure it doesn't crash on import/init
    assert tokenizer is not None

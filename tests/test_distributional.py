from uchi.distributional import DistributionalTokenizer

def test_distributional_tokenizer():
    dt = DistributionalTokenizer()
    seq = [1.0, 2.0, 3.0]
    
    # Just asserting it instantiates correctly
    assert dt is not None

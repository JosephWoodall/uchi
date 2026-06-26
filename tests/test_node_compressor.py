from uchi.node_compressor import NodeCompressor
from uchi.predictor import UniversalPredictor

def test_node_compressor():
    predictor = UniversalPredictor(context_length=3)
    seq = ['A', 'B', 'C', 'A', 'B', 'C'] * 10
    
    for val in seq:
        predictor.predict()
        predictor.observe(val)
        predictor.feedback(val)
        
    initial_nodes = len(predictor._nodes)
    compressor = NodeCompressor()
    # It takes a compressor instance and compress_pass
    res = compressor.compress_pass(predictor._root, cred_max=6.05)
    
    assert res is dict or isinstance(res, dict)

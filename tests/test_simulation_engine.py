from uchi.simulation_engine import LifelongSimulationEngine

def test_simulation_engine():
    engine = LifelongSimulationEngine(n_instances=2, context_length=2)
    # Stream different sequences to different instances
    seq1 = ['A', 'B', 'A', 'B'] * 10
    seq2 = ['C', 'D', 'C', 'D'] * 10
    
    engine.stream_parallel([seq1, seq2])
    
    # Check that it returns a valid prediction
    pred, conf = engine.vote_plural()
    assert pred is not None
    assert conf >= 0.0

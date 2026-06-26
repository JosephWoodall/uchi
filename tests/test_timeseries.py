from uchi.timeseries import MultivariateTSPredictor

def test_timeseries_predictor():
    ts = MultivariateTSPredictor(context_length=3, n_bins=5)
    seq = [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]] * 10
    
    ts.fit(seq)
    
    for val in seq:
        pred = ts.predict()
        ts.observe(val)
        ts.feedback(val)
        
    pred = ts.predict()
    assert pred is not None
    assert isinstance(pred, list)

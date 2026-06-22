from uchi.dual_predictor import DualPredictor

def test_dual_predictor():
    dp = DualPredictor(context_length=2)
    seq = ['hello', 'world', 'hello', 'world'] * 10
    
    for val in seq:
        pred, conf = dp.predict()
        dp.observe(val)
        dp.feedback(val)
        
    pred, conf = dp.predict()
    assert pred is not None
    assert isinstance(conf, float)

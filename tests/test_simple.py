from uchi.predictor import UniversalPredictor

def test_simple():
    
    p = UniversalPredictor(context_length=2)
    for i in range(50):
        p.observe('A')
        p.predict()
        p.observe('B')
        p.feedback('B')
    
        p.observe('C')
        p.predict()
        p.observe('D')
        p.feedback('D')
    
    p.observe('A')
    pred, conf = p.predict()
    print(f"Given A, predict: {pred} (conf {conf})")
    print("Dist:", dict(p._last_distribution))
    
    p.observe('C')
    pred, conf = p.predict()
    print(f"Given C, predict: {pred} (conf {conf})")
    print("Dist:", dict(p._last_distribution))

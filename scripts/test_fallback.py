import pickle
with open("brain.uchi", "rb") as f:
    router = pickle.load(f)

p = router.predictor
p.history = []
p.observe("what")
p.observe("is")
p.observe("the")
p.observe("[UNKNOWN:meaning]")
p.predict()
print("Dist length:", len(p._last_distribution))
print("Last depth:", p._last_prediction_depth)

import sys
from uchi.omni_router import OmniRouter

router = OmniRouter()

good_seq = ["<|user|>", "hello", "uchi", "<|assistant|>", "hello", "how", "can", "i", "assist", "you", "today"]
bad_seq = ["<|user|>", "what", "<|assistant|>", "can", "algorithm", "is", "a", "step", "okay", "goodbye", "up"]

score_good = router.predictor.score(good_seq)
score_bad = router.predictor.score(bad_seq)

print(f"Good sequence score: {score_good}")
print(f"Bad sequence score: {score_bad}")

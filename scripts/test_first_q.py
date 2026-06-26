import pickle
with open("brain.uchi", "rb") as f:
    router = pickle.load(f)

if router:
    print("User: hello uchi")
    reply = router.chat("hello uchi")
    print("ODUSP:", reply)
    
    print("User: what is the meaning of life")
    reply2 = router.chat("what is the meaning of life")
    print("ODUSP:", reply2)

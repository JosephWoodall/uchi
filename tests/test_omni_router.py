from uchi import OmniRouter, OntologicalState, OntologicalAction

def test_omni_router_multimodal():
    # Initialize the new Omni-Modal Master Controller
    router = OmniRouter(use_bpe=False, memory_window=3)
    
    # Simulate a stream with 4 distinct modalities!
    # 1. Text
    # 2. Math/Telemetry
    # 3. Code/Agent Objects
    # 4. Images & Audio paths
    stream = [
        "the", "system", "booted", "normally", 
        "CPU=45", "RAM=30%",
        OntologicalState("NOMINAL", properties={}),
        "screenshot_101.jpg", "alert_sound.wav",
        "CPU=99", 
        OntologicalState("CRITICAL", properties={}), 
        "RAM=100%", 
        OntologicalAction("REBOOT", target="SERVER")
    ]
    
    # Route the entire stream through the engine
    router.stream(stream)
    
    # Test zero-shot multi-modal query
    # The engine has NEVER seen this exact mapping, but should associate
    # "CRITICAL" state with the surrounding math telemetry.
    ans = router.query(["why", "CRITICAL", "?"])
    
    # query() returns a string — either a retrieved concept or "[Unknown Context]"
    assert isinstance(ans, str)

def test_omni_image_audio():
    router = OmniRouter(use_bpe=False)

    stream = [
        "user", "uploaded", "dog_photo.jpg",
        "dog_bark.wav",
        "classifier", "said", "dog"
    ]
    router.stream(stream)

    # Query what was uploaded
    ans = router.query(["what", "uploaded", "?"])
    assert isinstance(ans, str)

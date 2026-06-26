from uchi.process import ProcessPredictor, OntologicalState, OntologicalAction

def test_process():
    print("Initializing Process Predictor...")
    predictor = ProcessPredictor(context_length=4)
    
    # Define a simple ontology sequence
    # State -> Sort required -> Action: QuickSort
    # State -> Search required -> Action: BinarySearch
    
    s_sort = OntologicalState(name="NeedSort", properties=("unordered_list",))
    a_sort = OntologicalAction(name="ExecuteSort", target="QuickSort")
    
    s_search = OntologicalState(name="NeedSearch", properties=("sorted_list",))
    a_search = OntologicalAction(name="ExecuteSearch", target="BinarySearch")
    
    print("Streaming process observations...")
    for _ in range(50):
        predictor.observe_state(s_sort)
        predictor.predict_next_action() # Trigger prediction for credibility
        predictor.feedback_action(a_sort)
        
        predictor.observe_state(s_search)
        predictor.predict_next_action()
        predictor.feedback_action(a_search)
        
    print("\nTesting prediction...")
    print(predictor.forest.node_stats())
    predictor.observe_state(s_sort)
    pred, conf = predictor.predict_next_action()
    print(f"Given state {s_sort.name}, predicted action: {pred.name} (target={pred.target}), conf={conf:.2f}")

    predictor.observe_state(s_search)
    pred, conf = predictor.predict_next_action()
    print(f"Given state {s_search.name}, predicted action: {pred.name} (target={pred.target}), conf={conf:.2f}")

if __name__ == "__main__":
    main()

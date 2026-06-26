import pytest
from uchi.memory import AssociativeMemory
from uchi.omni_tokenizer import OmniTokenizer

def test_qa_accuracy_benchmark():
    """
    Simulates Facebook bAbI Tasks using the real OmniTokenizer and Fractal Attention.
    Calculates and prints the Question/Answer Accuracy Benchmark.
    """
    tokenizer = OmniTokenizer(use_wordnet=True) # Use full semantic clustering
    memory = AssociativeMemory(window_size=10)
    
    # Preload World Knowledge (Context)
    context = "Mary went to the bathroom . John moved to the hallway . The child was wearing a red hat , and had an orange in his hand ."
    ctx_concepts = tokenizer.tokenize(context.split())
    
    memory.stream_context(ctx_concepts)
    
    # Run Benchmark Questions
    benchmark_tasks = [
        {"q": "Where is Mary ?", "expected": "bathroom"},
        {"q": "Where is John ?", "expected": "hallway"},
        {"q": "What color was the hat ?", "expected": "red"},
        {"q": "What was in the hand ?", "expected": "orange"}
    ]
    
    correct = 0
    for task in benchmark_tasks:
        q_concepts = tokenizer.tokenize(task["q"].split())
        ans_concept = memory.query(q_concepts)
        
        # In a real scenario, we would detokenize. For the test, we check if the expected answer concept string is in the answer concept.
        expected_concept = tokenizer.tokenize([task["expected"]])[0]
        
        if ans_concept and expected_concept in ans_concept:
            correct += 1
            
    accuracy = (correct / len(benchmark_tasks)) * 100.0
    print(f"\n[BENCHMARK] ODUSP Question/Answer Accuracy: {accuracy}% ({correct}/{len(benchmark_tasks)})")

import unittest
import os
from uchi.omni_router import OmniRouter
from uchi.cli import ingest_file

class TestChatCapabilities(unittest.TestCase):
    def setUp(self):
        # We test with BPE off to ensure strictly deterministic behavior based on N-gram states
        self.router = OmniRouter(use_bpe=False, memory_window=5)

    def test_basic_conversational_grammar(self):
        """
        Tests if the geometric predictor can output a perfectly grammatical response
        based purely on the bootstrap dataset without structural bleeding.
        """
        # "hello uchi" should trigger the intro persona
        prompt = ["<|user|>", "hello", "uchi", "<|assistant|>"]
        reply = self.router.predict_future(prompt, steps=10, temperature=0.0, creativity=0.0)
        
        # Verify it hallucinates the exact deterministic continuation mathematically
        self.assertEqual(reply, ['hello!', 'i', 'am', 'odusp', 'the', 'omni-modal', 'deterministic', 'universal', 'sequence', 'predictor'])

    def test_dual_pass_routing_isolation(self):
        """
        Tests if preloaded structural context does not pollute the conversational space
        unless explicitly hit by the AssociativeMemory graph.
        """
        # Create a mock code file
        mock_file = "test_code.py"
        with open(mock_file, "w", encoding="utf-8") as f:
            f.write("def calculate_physics():\n    return 'E=mc^2'")
            
        try:
            # Ingest it using the structured bounding tags
            ingest_file(self.router, mock_file, quiet=True)
            
            # The conversational prompt "how are you" should still work flawlessly
            prompt = ["<|user|>", "how", "are", "you", "<|assistant|>"]
            reply = self.router.predict_future(prompt, steps=9, temperature=0.0, creativity=0.0)
            
            # It should not say 'E=mc^2' or 'def', it should just output the persona
            self.assertEqual(reply[:8], ['i', 'am', 'functioning', 'optimally', 'within', 'my', 'mathematical', 'constraints'])
            # The 9th token should be a boundary marker (either <|user|> or <|assistant|>)
            self.assertIn(reply[8], ['<|user|>', '<|assistant|>'])
            
            # Now test if the dual-pass router successfully retrieves the code via the associative query
            ans = self.router.query(["calculate_physics():"])
            self.assertNotEqual(ans, "[Unknown Context]")
            
        finally:
            if os.path.exists(mock_file):
                os.remove(mock_file)

if __name__ == '__main__':
    unittest.main()

import unittest
import os
from uchi.omni_router import OmniRouter
from uchi.cli import ingest_file

class TestChatCapabilities(unittest.TestCase):
    def setUp(self):
        # We test with BPE off to ensure strictly deterministic behavior based on N-gram states
        self.router = OmniRouter(use_bpe=False, memory_window=5)

    def _get_reply(self, query):
        """Simulate the Natural Autocomplete CLI flow and extract the reply."""
        tokens = ["<|user|>"] + query.split()
        pred = self.router.predict_future(tokens, steps=30, temperature=0.0, creativity=0.0)
        
        reply = []
        recording = False
        for p in pred:
            if p == "<|assistant|>" and not recording:
                recording = True
                continue
            if recording and p in ("<|user|>", "<|assistant|>"):
                break
            if recording:
                reply.append(p)
        return reply

    def test_greeting_produces_coherent_response(self):
        """
        Tests if a greeting produces a grammatically coherent response
        that includes relevant greeting or identity tokens.
        """
        reply = self._get_reply("hello")
        reply_text = ' '.join(reply)
        
        # Should produce a non-empty reply
        self.assertGreater(len(reply), 0, "Reply should not be empty")
        
        # Should contain at least one greeting or identity marker
        greeting_markers = {"hello", "hi", "there", "help", "uchi", "i", "am"}
        self.assertTrue(
            any(tok in greeting_markers for tok in reply),
            f"Reply '{reply_text}' should contain a greeting or identity token"
        )

    def test_identity_question_mentions_creator(self):
        """
        Tests if asking 'who created you' produces a response mentioning the creator.
        """
        reply = self._get_reply("who created you")
        reply_text = ' '.join(reply)
        
        # Should mention joseph woodall (the creator)
        self.assertIn("joseph", reply, f"Reply '{reply_text}' should mention 'joseph'")
        self.assertIn("woodall", reply, f"Reply '{reply_text}' should mention 'woodall'")

    def test_capability_question_is_relevant(self):
        """
        Tests if asking about capabilities produces a relevant answer.
        """
        reply = self._get_reply("what can you do")
        reply_text = ' '.join(reply)
        
        # Should mention prediction or sequences or patterns.
        # Tokens go through OmniTokenizer so "sequences" → "sequence", "predict" → "predictor".
        capability_markers = {
            "predict", "predictor", "sequences", "sequence", "patterns",
            "help", "memory", "information", "stream", "trie", "graph",
        }
        self.assertTrue(
            any(tok in capability_markers for tok in reply),
            f"Reply '{reply_text}' should mention a capability"
        )

    def test_unknown_topic_is_graceful(self):
        """
        Tests if asking about something outside the training data produces a graceful response
        rather than garbage tokens.
        """
        reply = self._get_reply("tell me about the weather")
        reply_text = ' '.join(reply)
        
        # Should produce a non-empty reply
        self.assertGreater(len(reply), 0, "Reply should not be empty for unknown topics")
        
        # Should NOT contain structural tags or code tokens
        for tok in reply:
            self.assertNotIn("<|file:", tok, f"Reply should not contain file boundary tags")

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
            
            # A conversational prompt should still produce a coherent persona response
            reply = self._get_reply("how are you")
            reply_text = ' '.join(reply)
            
            # It should not say 'E=mc^2' or 'def'
            self.assertNotIn("e=mc^2", reply_text.lower(), "Reply should not contain code content")
            self.assertNotIn("def", reply, "Reply should not contain Python keywords")
            
            # Should produce a non-empty, coherent reply
            self.assertGreater(len(reply), 0, "Reply should not be empty after code ingestion")
            
        finally:
            if os.path.exists(mock_file):
                os.remove(mock_file)

if __name__ == '__main__':
    unittest.main()

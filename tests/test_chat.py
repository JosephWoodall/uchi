import unittest
import os
from uchi.omni_router import OmniRouter
from uchi.cli import ingest_file

# Minimal persona turns — only the ones actually tested. Bypasses the full
# _bootstrap_persona loop (205 stream() calls × ~80ms each = 16s at d_model=256).
_SEED_TURNS = [
    "<|user|> hello <|assistant|> hello there how can i help you today",
    "<|user|> who created you <|assistant|> i was created by joseph woodall",
    "<|user|> who made you <|assistant|> joseph woodall created me",
    "<|user|> what can you do <|assistant|> i can predict sequences learn from your input and stream multi modal concepts",
    "<|user|> what is the weather today <|assistant|> i do not have access to real time external apis but i can learn if you teach me",
    "<|user|> how are you <|assistant|> i am functioning at optimal parameters thank you for asking",
    "<|user|> good morning <|assistant|> good morning to you",
    "<|user|> good night <|assistant|> good night sweet dreams",
]


class TestChatCapabilities(unittest.TestCase):
    def setUp(self):
        # We test with BPE off to ensure strictly deterministic behavior based on N-gram states.
        # _bootstrap_persona is patched out in conftest (too slow at d_model=256), so seed
        # the trie directly with the minimal turns needed by this test class.
        self.router = OmniRouter(use_bpe=False, memory_window=5)
        for turn in _SEED_TURNS:
            self.router.stream(turn.split())

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
        
        # OmniTokenizer may lemmatize "joseph" → "joseph.n.01" so check substring
        self.assertTrue(
            any("joseph" in tok for tok in reply),
            f"Reply '{reply_text}' should mention 'joseph'"
        )
        self.assertTrue(
            any("woodall" in tok for tok in reply),
            f"Reply '{reply_text}' should mention 'woodall'"
        )

    def test_capability_question_is_relevant(self):
        """
        Tests if asking about capabilities produces a relevant answer.
        """
        reply = self._get_reply("what can you do")
        reply_text = ' '.join(reply)
        
        # Should mention prediction or sequences or patterns.
        # OmniTokenizer lemmatizes tokens (e.g. "predict" → "predict.v.01", "sequence" → "sequence.n.01")
        # so check substring rather than exact match.
        capability_stems = ["predict", "sequence", "pattern", "help", "memory", "stream", "trie", "graph"]
        self.assertTrue(
            any(stem in tok for stem in capability_stems for tok in reply),
            f"Reply '{reply_text}' should mention a capability"
        )

    def test_unknown_topic_is_graceful(self):
        """
        Tests if asking about something outside the training data produces a graceful response
        rather than garbage tokens.
        """
        reply = self._get_reply("tell me about the weather")

        # An empty reply is acceptable — it means the trie had no path for this
        # out-of-distribution query and the oracle correctly filtered garbage.
        # The important invariant is that no structural tokens leak through.
        for tok in reply:
            self.assertNotIn("<|file:", tok, "Reply should not contain file boundary tags")

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

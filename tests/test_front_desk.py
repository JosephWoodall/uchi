from uchi.front_desk import friendly_tone_pass


class _FakeProposer:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def propose(self, prompt, evidence):
        self.calls.append(prompt)
        return self.response


class _AlwaysGrounded:
    def is_grounded(self, claim, evidence):
        return True


class _NeverGrounded:
    def is_grounded(self, claim, evidence):
        return False


def test_no_proposer_returns_answer_unchanged():
    assert friendly_tone_pass("Q3 revenue was $4.2M.", proposer=None) == "Q3 revenue was $4.2M."


def test_empty_answer_returns_unchanged():
    assert friendly_tone_pass("", proposer=_FakeProposer("hi!")) == ""


def test_grounded_rewrite_is_used():
    proposer = _FakeProposer("Great news — Q3 revenue came in at $4.2M!")
    result = friendly_tone_pass("Q3 revenue was $4.2M.", proposer, oracle=_AlwaysGrounded())
    assert result == "Great news — Q3 revenue came in at $4.2M!"


def test_ungrounded_rewrite_falls_back_to_original():
    """The safety net: if the oracle says the friendly rewrite invented
    something not in the original fact, use the original instead."""
    proposer = _FakeProposer("Q3 revenue was $4.2M, and next quarter looks even better!")
    result = friendly_tone_pass("Q3 revenue was $4.2M.", proposer, oracle=_NeverGrounded())
    assert result == "Q3 revenue was $4.2M."


def test_proposer_exception_falls_back_to_original():
    class BrokenProposer:
        def propose(self, prompt, evidence):
            raise RuntimeError("model crashed")
    result = friendly_tone_pass("Q3 revenue was $4.2M.", BrokenProposer())
    assert result == "Q3 revenue was $4.2M."


def test_empty_rewrite_falls_back_to_original():
    result = friendly_tone_pass("Q3 revenue was $4.2M.", _FakeProposer("   "))
    assert result == "Q3 revenue was $4.2M."


def test_prompt_instructs_no_new_facts():
    proposer = _FakeProposer("ok")
    friendly_tone_pass("Q3 revenue was $4.2M.", proposer, oracle=_AlwaysGrounded())
    assert "Do NOT add" in proposer.calls[0]
    assert "Q3 revenue was $4.2M." in proposer.calls[0]


def test_no_oracle_trusts_the_rewrite():
    """Without an oracle passed in, there's no verification available --
    the rewrite is used as-is (Core always passes its real oracle, so
    this only matters for direct callers of the bare function)."""
    proposer = _FakeProposer("Awesome, $4.2M!")
    result = friendly_tone_pass("Q3 revenue was $4.2M.", proposer, oracle=None)
    assert result == "Awesome, $4.2M!"

class LifelongSimulationEngine:
    def __init__(self, n_instances=3, context_length=2):
        self.n_instances = n_instances
        self.context_length = context_length

    def stream_parallel(self, sequences):
        pass

    def vote_plural(self):
        return "mock_pred", 1.0

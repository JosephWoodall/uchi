from dataclasses import dataclass
from typing import Any, List, Tuple
from uchi.forest import PredictorForest

@dataclass(frozen=True)
class OntologicalState:
    name: str
    properties: tuple

@dataclass(frozen=True)
class OntologicalAction:
    name: str
    target: str

class ProcessPredictor:
    """
    Predicts the next ontological action given a sequence of states and actions.
    Uses an underlying PredictorForest to model the process flow online.
    """
    def __init__(self, context_length: int = 4, n_trees: int = 5):
        self.forest = PredictorForest(
            context_length=context_length, 
            n_trees=n_trees, 
            voting='adaptive'
        )
        self.history: List[Any] = []
        
    def observe_state(self, state: OntologicalState):
        """Observe a new state in the process."""
        self.forest.observe(state)
        self.history.append(state)
        
    def feedback_action(self, action: OntologicalAction):
        """Provide feedback on the action taken after the last state."""
        self.forest.observe(action)
        self.forest.feedback(action)
        self.history.append(action)
        
    def predict_next_action(self) -> Tuple[OntologicalAction, float]:
        """
        Predict the next optimal action based on the current process context.
        Returns the predicted action and confidence score.
        """
        # Since we want to predict the next action, the forest's next prediction 
        # is automatically tuned to the sequence of states/actions it has observed.
        prediction, confidence = self.forest.predict()
        return prediction, confidence

    def simulate_process(self, n_steps: int) -> List[Any]:
        """
        Generatively simulate a process forward.
        """
        simulated = []
        for _ in range(n_steps):
            pred, _ = self.forest.predict()
            simulated.append(pred)
            self.forest.observe(pred)
        return simulated

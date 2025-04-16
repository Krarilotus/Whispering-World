# agent_framework/core/personality.py
from typing import Dict, List, Optional, Any

DEFAULT_OCEAN = {
    "openness": 0.5,
    "conscientiousness": 0.5,
    "extraversion": 0.5,
    "agreeableness": 0.5,
    "neuroticism": 0.5,
}

class AffectiveState:
    """Represents the agent's current emotional state."""
    def __init__(self, valence: float = 0.0, arousal: float = 0.0, current_emotion: str = "neutral"):
        # Valence: Pleasant (1) to Unpleasant (-1)
        # Arousal: Active/Excited (1) to Passive/Calm (-1)
        self.valence: float = max(-1.0, min(1.0, valence))
        self.arousal: float = max(-1.0, min(1.0, arousal))
        # Discrete emotion label for easier LLM prompting
        self.current_emotion: str = current_emotion

    def update(self, valence_change: float = 0.0, arousal_change: float = 0.0, new_emotion: Optional[str] = None):
        """Updates the emotional state, potentially clamping values."""
        self.valence = max(-1.0, min(1.0, self.valence + valence_change))
        self.arousal = max(-1.0, min(1.0, self.arousal + arousal_change))
        if new_emotion:
            self.current_emotion = new_emotion
        # TODO: Could add decay towards neutral over time

    def get_state_description(self) -> str:
        """Returns a simple text description of the current emotion."""
        # Could be more nuanced based on valence/arousal values
        return f"Feeling {self.current_emotion}."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "current_emotion": self.current_emotion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AffectiveState':
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            current_emotion=data.get("current_emotion", "neutral")
        )

class Personality:
    """Defines the agent's stable personality traits and characteristics."""
    def __init__(self,
                 ocean_scores: Optional[Dict[str, float]] = None,
                 traits: Optional[List[str]] = None,
                 flaws: Optional[List[str]] = None):
        self.ocean_scores: Dict[str, float] = ocean_scores or DEFAULT_OCEAN.copy()
        # Ensure all OCEAN keys are present
        for key in DEFAULT_OCEAN:
            if key not in self.ocean_scores:
                self.ocean_scores[key] = DEFAULT_OCEAN[key]
            # Clamp values
            self.ocean_scores[key] = max(0.0, min(1.0, self.ocean_scores[key]))

        self.traits: List[str] = traits or [] # Positive or neutral distinguishing features
        self.flaws: List[str] = flaws or [] # Negative traits, weaknesses, fears

    def get_description(self) -> str:
        """Generates a textual description of the personality."""
        desc = "Personality Profile:\n"
        desc += "  OCEAN Scores:\n"
        for trait, score in self.ocean_scores.items():
            desc += f"    - {trait.capitalize()}: {score:.2f}\n" # (0=Low, 1=High)
        if self.traits:
            desc += "  Traits:\n"
            for trait in self.traits:
                desc += f"    - {trait}\n"
        if self.flaws:
            desc += "  Flaws:\n"
            for flaw in self.flaws:
                desc += f"    - {flaw}\n"
        return desc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocean_scores": self.ocean_scores,
            "traits": self.traits,
            "flaws": self.flaws,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Personality':
        return cls(
            ocean_scores=data.get("ocean_scores"),
            traits=data.get("traits"),
            flaws=data.get("flaws")
        )
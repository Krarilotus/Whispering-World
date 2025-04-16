# agent_framework/core/motivation.py
from typing import Dict, List, Optional, Any
from .personality import Personality, AffectiveState

# --- Psychological Needs (SDT inspired) ---
DEFAULT_NEEDS = {
    "autonomy": 0.7,  # Sense of control, acting aligned with values
    "competence": 0.7, # Feeling effective, mastering challenges
    "relatedness": 0.7, # Feeling connected, belonging
    # --- Basic Needs (Maslow/Drive inspired) ---
    "physiological_safety": 0.8, # Basic survival, security (combined for simplicity)
    "stimulation": 0.6 # Need for novelty, exploration (links to Openness)
}

class Goal:
    """Represents an agent's objective or intention."""
    def __init__(self, description: str, source: str, urgency: float = 5.0, target_state: Optional[str] = None):
        self.description: str = description
        self.source: str = source # e.g., "need:relatedness", "ideal:justice", "bond:family", "flaw:greedy"
        self.urgency: float = max(1.0, min(10.0, urgency)) # Scale 1-10
        self.target_state: Optional[str] = target_state # Optional description of desired outcome
        self.status: str = "active" # active, completed, failed

    def __repr__(self) -> str:
        return f"Goal(Desc='{self.description}', Source='{self.source}', Urgency={self.urgency:.1f}, Status='{self.status}')"

    def to_dict(self) -> Dict[str, Any]:
         return {
            "description": self.description,
            "source": self.source,
            "urgency": self.urgency,
            "target_state": self.target_state,
            "status": self.status,
         }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        goal = cls(
            description=data["description"],
            source=data["source"],
            urgency=data.get("urgency", 5.0),
            target_state=data.get("target_state")
        )
        goal.status = data.get("status", "active")
        return goal


class MotivationSystem:
    """Manages the agent's needs, drives, ideals, bonds, and goals."""
    def __init__(self,
                 needs: Optional[Dict[str, float]] = None,
                 ideals: Optional[List[str]] = None,
                 bonds: Optional[List[str]] = None):
        self.needs: Dict[str, float] = needs or DEFAULT_NEEDS.copy()
         # Ensure all default needs are present and clamped 0-1
        for key in DEFAULT_NEEDS:
            if key not in self.needs:
                self.needs[key] = DEFAULT_NEEDS[key]
            self.needs[key] = max(0.0, min(1.0, self.needs[key]))

        self.ideals: List[str] = ideals or [] # Core beliefs, values
        self.bonds: List[str] = bonds or [] # Connections to people, places, items

        self.active_goals: List[Goal] = []

    def update_needs(self, changes: Dict[str, float]):
        """Applies changes to need satisfaction levels."""
        for need, change in changes.items():
            if need in self.needs:
                self.needs[need] = max(0.0, min(1.0, self.needs[need] + change))
        self.generate_goals_from_state() # Re-evaluate goals after need changes

    def add_goal(self, goal: Goal, check_duplicates: bool = True):
        """Adds a new goal if it's not already active."""
        if check_duplicates:
             for existing_goal in self.active_goals:
                  if existing_goal.description == goal.description and existing_goal.status == "active":
                       # Optionally update urgency if the new goal is more urgent
                       existing_goal.urgency = max(existing_goal.urgency, goal.urgency)
                       return # Don't add duplicate
        self.active_goals.append(goal)

    def generate_goals_from_state(self, personality: Optional[Personality] = None):
        """Generates or updates goals based on needs, ideals, bonds, flaws."""
        # --- Need-Based Goals ---
        for need, level in self.needs.items():
            urgency = (1.0 - level) * 10.0 # Higher urgency for lower satisfaction
            if urgency > 3.0: # Only generate goal if need is sufficiently low
                goal_desc = ""
                if need == "autonomy" and urgency > 6: goal_desc = "Assert independence or make own choice"
                elif need == "competence" and urgency > 6: goal_desc = "Seek a challenge or demonstrate skill"
                elif need == "relatedness" and urgency > 5: goal_desc = "Connect with someone or strengthen a bond"
                elif need == "physiological_safety" and urgency > 7: goal_desc = "Ensure safety or acquire basic resources"
                elif need == "stimulation" and urgency > 4: goal_desc = "Explore surroundings or seek novelty"

                if goal_desc:
                    self.add_goal(Goal(description=goal_desc, source=f"need:{need}", urgency=urgency))

        # --- Ideal-Based Goals ---
        for ideal in self.ideals:
             # Ideals often translate to standing goals or trigger goals in specific situations
             # Simplified: Add as a persistent, moderately urgent goal
             self.add_goal(Goal(description=f"Uphold ideal: '{ideal}'", source=f"ideal:{ideal}", urgency=6.0), check_duplicates=True)

        # --- Bond-Based Goals ---
        for bond in self.bonds:
             # Bonds often trigger goals when the bonded entity is relevant/threatened
             # Simplified: Add as a persistent, moderately urgent goal
             self.add_goal(Goal(description=f"Maintain/protect bond: '{bond}'", source=f"bond:{bond}", urgency=7.0), check_duplicates=True)

        # --- Flaw-Based Goals (Optional) ---
        if personality:
            for flaw in personality.flaws:
                 # Flaws might generate counter-productive or specific situational goals
                 # Example: Flaw "Greedy" -> Goal "Acquire valuable items"
                 # This requires more specific mapping logic based on flaw text
                 # Simplified: Add as a lower urgency background goal
                 self.add_goal(Goal(description=f"Act according to flaw: '{flaw}'", source=f"flaw:{flaw}", urgency=3.0), check_duplicates=True)

        # Remove completed/failed goals
        self.active_goals = [g for g in self.active_goals if g.status == "active"]


    def prioritize_goals(self, personality: Optional[Personality] = None, affective_state: Optional[AffectiveState] = None) -> List[Goal]:
        """Ranks active goals based on urgency, personality, and affect."""
        # Simple prioritization: Primarily by urgency
        # TODO: Enhance with personality biases (e.g., Conscientious -> task goals)
        # TODO: Enhance with affect biases (e.g., Fear -> safety goals)
        return sorted(self.active_goals, key=lambda g: g.urgency, reverse=True)

    def get_highest_priority_goal(self, personality: Optional[Personality] = None, affective_state: Optional[AffectiveState] = None) -> Optional[Goal]:
        """Returns the single most important goal."""
        prioritized = self.prioritize_goals(personality, affective_state)
        return prioritized[0] if prioritized else None

    def set_goal_status(self, goal_description: str, status: str):
        """Sets the status ('completed' or 'failed') of a goal by its description."""
        for goal in self.active_goals:
            if goal.description == goal_description:
                goal.status = status
                break

    def get_state_description(self) -> str:
        """Generates a textual description of the motivational state."""
        desc = "Motivational State:\n"
        desc += "  Needs (Satisfaction 0-1):\n"
        for need, level in self.needs.items():
            desc += f"    - {need.capitalize()}: {level:.2f}\n"
        if self.ideals:
            desc += "  Ideals:\n"
            for ideal in self.ideals: desc += f"    - {ideal}\n"
        if self.bonds:
            desc += "  Bonds:\n"
            for bond in self.bonds: desc += f"    - {bond}\n"
        desc += "  Active Goals (Highest Priority First):\n"
        prioritized_goals = self.prioritize_goals()
        if prioritized_goals:
            for goal in prioritized_goals[:5]: # Show top 5
                 desc += f"    - {goal}\n"
        else:
            desc += "    - None\n"
        return desc

    def to_dict(self) -> Dict[str, Any]:
         return {
             "needs": self.needs,
             "ideals": self.ideals,
             "bonds": self.bonds,
             "active_goals": [goal.to_dict() for goal in self.active_goals],
         }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MotivationSystem':
        system = cls(
            needs=data.get("needs"),
            ideals=data.get("ideals"),
            bonds=data.get("bonds")
        )
        goal_data_list = data.get("active_goals", [])
        system.active_goals = [Goal.from_dict(g_data) for g_data in goal_data_list]
        return system
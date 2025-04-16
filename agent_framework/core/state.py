# agent_framework/core/state.py
from typing import Optional, List, Dict, Any
from .personality import AffectiveState # Use relative import within the package
from .motivation import Goal # Use relative import

class CurrentState:
    """Tracks the agent's immediate context and status."""
    def __init__(self,
                 location: str = "Unknown",
                 inventory: Optional[List[str]] = None,
                 affective_state: Optional[AffectiveState] = None,
                 current_action: Optional[str] = None,
                 action_target: Optional[str] = None,
                 action_start_time: Optional[float] = None,
                 action_duration: float = 0.0,
                 active_goal: Optional[Goal] = None # Link to the current driving goal
                ):
        self.location: str = location
        self.inventory: List[str] = inventory or []
        self.affective_state: AffectiveState = affective_state or AffectiveState()
        self.current_action: Optional[str] = current_action
        self.action_target: Optional[str] = action_target
        self.action_start_time: Optional[float] = action_start_time
        self.action_duration: float = action_duration # Estimated duration
        self.active_goal: Optional[Goal] = active_goal

    def is_busy(self, current_time: float) -> bool:
        """Checks if the agent is currently performing an action."""
        if self.current_action and self.action_start_time:
            return current_time < (self.action_start_time + self.action_duration)
        return False

    def start_action(self, action: str, duration: float, target: Optional[str] = None, start_time: Optional[float] = None):
        """Sets the current action."""
        import time # Local import
        self.current_action = action
        self.action_duration = duration
        self.action_target = target
        self.action_start_time = start_time if start_time is not None else time.time()

    def finish_action(self):
        """Clears the current action."""
        self.current_action = None
        self.action_duration = 0.0
        self.action_target = None
        self.action_start_time = None

    def get_state_description(self) -> str:
        """Generates a textual description of the current state."""
        import time # Local import
        desc = "Current State:\n"
        desc += f"  Location: {self.location}\n"
        desc += f"  Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}\n"
        desc += f"  Emotion: {self.affective_state.get_state_description()}\n"
        if self.active_goal:
            desc += f"  Current Goal: {self.active_goal.description} (Urgency: {self.active_goal.urgency:.1f})\n"
        else:
            desc += "  Current Goal: None\n"

        if self.current_action and self.action_start_time:
            elapsed = time.time() - self.action_start_time
            remaining = max(0, self.action_duration - elapsed)
            desc += f"  Action: {self.current_action}"
            if self.action_target: desc += f" (Target: {self.action_target})"
            desc += f" (Remaining: {remaining:.1f}s)\n"
        else:
            desc += "  Action: Idle\n"
        return desc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "inventory": self.inventory,
            "affective_state": self.affective_state.to_dict(),
            "current_action": self.current_action,
            "action_target": self.action_target,
            "action_start_time": self.action_start_time,
            "action_duration": self.action_duration,
            "active_goal": self.active_goal.to_dict() if self.active_goal else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurrentState':
        affective_state_data = data.get("affective_state")
        active_goal_data = data.get("active_goal")
        return cls(
            location=data.get("location", "Unknown"),
            inventory=data.get("inventory"),
            affective_state=AffectiveState.from_dict(affective_state_data) if affective_state_data else None,
            current_action=data.get("current_action"),
            action_target=data.get("action_target"),
            action_start_time=data.get("action_start_time"),
            action_duration=data.get("action_duration", 0.0),
            active_goal=Goal.from_dict(active_goal_data) if active_goal_data else None
        )
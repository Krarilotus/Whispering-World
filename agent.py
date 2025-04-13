# agent.py (Simple Agent Representation)

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GameAgent:
    """
    Represents an interactive character in the game powered by an OpenAI Assistant.
    Manages the agent's identity (Assistant ID) and conversation state (Thread ID).
    """

    def __init__(
        self,
        name: str,
        assistant_id: str,
        thread_id: Optional[str] = None,
        # Add any other game-specific state for this agent
        knowledge_summary: Optional[str] = None,
        relationship_level: int = 0,
    ):
        """
        Initializes the GameAgent.

        Args:
            name: The in-game name of the agent.
            assistant_id: The OpenAI Assistant ID backing this agent.
            thread_id: The OpenAI Thread ID for the ongoing conversation with the player.
                       Starts as None if no conversation has begun.
            knowledge_summary: Externally stored summary of past interactions.
            relationship_level: Example game state associated with the agent.
        """
        self.name = name
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.knowledge_summary = knowledge_summary
        self.relationship_level = relationship_level
        # Add other state variables as needed (e.g., current mood, objectives)

        logger.info(f"GameAgent '{self.name}' initialized. Assistant ID: {self.assistant_id}, Thread ID: {self.thread_id}")

    def update_state(self, summary: Optional[str] = None, relationship_delta: int = 0):
        """Updates the agent's game-specific state."""
        if summary is not None:
             self.knowledge_summary = summary
             logger.info(f"Updated knowledge summary for agent '{self.name}'.")
        if relationship_delta != 0:
             self.relationship_level += relationship_delta
             logger.info(f"Updated relationship level for agent '{self.name}' to {self.relationship_level}.")

    def get_current_game_state_context(self) -> str:
         """Generates a context string based on the agent's current game state."""
         # Example: Customize based on what the AI should know
         context = f"Agent Name: {self.name}\n"
         context += f"Relationship with Player: {self.relationship_level} (Scale: -10 to +10)\n"
         if self.knowledge_summary:
              context += f"Summary of Past Interactions:\n{self.knowledge_summary}\n"
         # Add other relevant game state (mood, location, immediate goal, etc.)
         context += f"Current Goal: Guard the cellar key, prevent player escape.\n"
         context += f"Personal Weakness: Easily bored, susceptible to flattery or interesting stories.\n"

         return context

    def set_thread_id(self, thread_id: str):
        """Assigns a thread ID to this agent's conversation."""
        self.thread_id = thread_id
        logger.info(f"Assigned Thread ID {thread_id} to agent '{self.name}'.")
# agent_framework/core/agent.py
import time
import logging
import json
import re
import asyncio
import os # Needed for file path manipulation
from typing import Dict, Any, Optional, List, TYPE_CHECKING # Use modern types

# Relative imports
from .memory import MemoryStream, MemoryObject, Observation
from .personality import Personality, AffectiveState
from .motivation import MotivationSystem, Goal
from .state import CurrentState
from ..llm.llm_interface import LLM_API_Interface
if TYPE_CHECKING: from ..world.world import World
from ..file_utils.yaml_handler import save_agent_state_to_yaml, load_agent_config_from_yaml
from ..api.agent_api import AgentManager, DEFAULT_MODEL
from ..llm import prompts
import openai

logger = logging.getLogger(__name__)

class Agent:
    """ Agent class using consolidated interface, corrected context handling, and ID saving. """
    def __init__(self, config: dict[str, Any], agent_manager: AgentManager, world: Optional['World'] = None, config_file_path: Optional[str] = None):
        """ Initializes agent. Stores config path for saving back Assistant ID. """
        self._config_data = config # Store original config data
        self._config_file_path = config_file_path # Store path to save back to
        profile = self._config_data.get('profile', {})
        self.name: str = profile.get('name', 'Unnamed Agent')
        self.description: str = profile.get('description', 'An AI agent.')
        self.agent_manager_ref = agent_manager
        self.world = world
        self.assistant_id: Optional[str] = profile.get('assistant_id') # Load existing ID if present
        self.thread_id: Optional[str] = None
        self.llm_interface : Optional[LLM_API_Interface] = None # Must be set externally

        logger.info(f"Initializing Agent: {self.name}")
        self.personality = Personality.from_dict(self._config_data.get('personality', {}))
        self.motivation = MotivationSystem.from_dict(self._config_data.get('motivation', {}))
        self.memory = MemoryStream.from_dict(self._config_data.get('memory', {}))
        self.current_state = CurrentState.from_dict(self._config_data.get('current_state', {}))

        # Add initial memories
        if not self.memory._memories and 'initial_memory' in self._config_data:
             initial_mems = self._config_data['initial_memory']
             if isinstance(initial_mems, list):
                  count = 0
                  for mem_text in initial_mems:
                       if isinstance(mem_text, str): self.memory.add_memory(Observation(mem_text, importance=7.0)); count+=1
                  if count > 0: logger.info(f"Added {count} initial memories.")

        self.motivation.generate_goals_from_state(self.personality)
        logger.info(f"Agent {self.name} init complete. World: {'Yes' if self.world else 'No'}")
        self._log_state(logging.DEBUG)

    def set_llm_interface(self, llm_interface: LLM_API_Interface):
        """ Links the shared LLM Interface. """
        self.llm_interface = llm_interface
        logger.info(f"LLM Interface linked for Agent {self.name}")

    async def perceive(self, observation_text: str, source: str = "external", importance: float = 5.0):
        """Adds observation to memory stream."""
        # (Implementation remains the same)
        obs = Observation(f"[{source.upper()}] {observation_text}", importance=importance)
        self.memory.add_memory(obs)
        logger.info(f"Agent {self.name} perceived: {observation_text[:100]}...")
        self._log_state(logging.DEBUG)

    async def think_and_respond(self, user_input: str) -> dict[str, Any]:
        """ Main agent processing loop: Perceive -> Generate Context -> Add User Msg -> Run LLM -> Update State """
        start_time = time.time()
        logger.info(f">>> Agent {self.name} processing input: '{user_input[:100]}...'")
        if not self.llm_interface: logger.error("LLM Interface missing!"); return {"action": {}, "dialogue": "[ERR: No Interface]", "reasoning":"No LLM Interface"}
        if not self.thread_id: # Ensure thread exists before adding message
             if not await self.initialize_conversation(): return {"action":{}, "dialogue":"[ERR: Thread Init Fail]", "reasoning":"Failed to create/verify thread"}

        # 1. Perceive Input
        await self.perceive(user_input, source="PlayerInput", importance=6.0)

        # 2. Simplified Verification / Context Generation
        verification_notes = self._perform_simplified_verification(user_input)

        # 3. Construct Dynamic Prompt (Context + Task) - SEPARATE from user_input
        base_context_prompt = self._construct_base_prompt()
        task_prompt = prompts.get_react_to_input_prompt(
             user_input=user_input, # Pass user input here for analysis instruction
             verification_notes=verification_notes)
        # This 'additional_instructions' prompt holds the dynamic context + task for the LLM run
        additional_instructions_prompt = f"{base_context_prompt}\n{task_prompt}"
        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Additional Instructions Prompt:\n{additional_instructions_prompt}")

        # 4. Add ONLY the actual user input to the thread
        msg_added_ok = await self.agent_manager_ref.add_message_to_thread(self.thread_id, user_input, role="user")
        if not msg_added_ok:
             logger.error(f"Failed to add user message '{user_input[:50]}...' to thread {self.thread_id}");
             return {"action": {}, "dialogue": "[ERR: Message Send Fail]", "reasoning":"Failed add_message_to_thread"}

        # 5. Run LLM with dynamic prompt via 'additional_instructions'
        llm_response = await self.llm_interface.generate_json_response(
            assistant_id=self.assistant_id,
            thread_id=self.thread_id,
            additional_instructions=additional_instructions_prompt, # Pass dynamic prompt here
            task_description=f"Agent {self.name} response generation"
        )

        # 6. Process Response & Update State
        if not llm_response: # Fallback
            logger.error(f"Agent {self.name}: Failed get valid response from LLM Interface.")
            return {"action": {"type": "stay_in_conversation"}, "dialogue": "[Internal processing error]", "reasoning": "LLM Interface Error"}

        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Response Dict: {json.dumps(llm_response, indent=2)}")
        self._update_internal_state_from_llm(llm_response) # Refactored state update
        action_data = self._determine_action_from_llm(llm_response) # Refactored action handling

        # 7. Timing & Return
        processing_time = time.time() - start_time
        action_type = action_data.get('type', 'unknown')
        logger.info(f"<<< Agent {self.name} responded in {processing_time:.2f}s. Action: {action_type}")
        if processing_time > 45: logger.warning(f"Agent response took >45s ({processing_time:.1f}s)")
        return {"action": action_data, "dialogue": llm_response.get("dialogue", ""), "reasoning": llm_response.get("reasoning", "N/A")}

    # --- Helper methods for think_and_respond ---
    def _construct_base_prompt(self) -> str:
        """Constructs the DYNAMIC context part of the prompt for additional_instructions."""
        # Focus on DYNAMIC state and RELEVANT memories
        world_summary = "World context unavailable."
        if self.world:
             try: world_summary = self.world.state.get_summary(max_entities=10, max_rels=3, max_len=400) # Even more concise
             except Exception as e: logger.error(f"Error getting world summary: {e}"); world_summary = "Error retrieving world context."

        # Use vector search for relevant memories
        memory_summary = "No relevant memories found for current input."
        last_observation = self.memory.get_memory_summary(max_memories=1, max_length=150) # Get latest observation text
        query_context = last_observation
        if self.current_state.active_goal: query_context += f" Goal: {self.current_state.active_goal.description}"

        if self.memory.faiss_index and query_context:
             relevant_memories = self.memory.retrieve_relevant_memories(query_context.strip(), top_k=3) # TOP 3 Relevant
             if relevant_memories:
                 mem_summary_lines = [f"- {mem}" for mem in relevant_memories]
                 memory_summary = "Relevant Memories:\n" + "\n".join(mem_summary_lines)
        else: logger.debug("Vector search unavailable or no query context for relevant memories.")

        # Combine dynamic parts for the prompt
        prompt = prompts.get_base_context_prompt(
            agent_name=self.name,
            agent_description=None, # Base description is in Assistant instructions
            personality_description=None, # Base personality is in Assistant instructions
            motivation_description=self.motivation.get_state_description(), # Include current goals/needs
            current_state_description=self.current_state.get_state_description(), # Include location, emotion etc.
            memory_summary=memory_summary, # Include only relevant memories
            world_context_summary=world_summary) # Include concise world state
        return prompt

    def _log_state(self, level=logging.DEBUG):
        """Logs the current state of all components."""
        # (Implementation remains the same)
        if logger.isEnabledFor(level):
            max_len=1500; log_str=f"--- {self.name} State ---\n"; log_str+=self.personality.get_description()+"\n"; log_str+=self.motivation.get_state_description()+"\n"; log_str+=self.current_state.get_state_description()+"\n"; log_str+=f"Memory (Count:{len(self.memory._memories)}):\n{self.memory.get_memory_summary(max_memories=5, max_length=300)}\n"; log_str+="-"*20; logger.log(level, log_str[:max_len]+('...' if len(log_str)>max_len else ''))

    def _perform_simplified_verification(self, user_input: str) -> str:
        """ Performs minimal local checks for context. """
        # (Implementation remains the same - basic entity check)
        logger.debug(f"--- Running Simplified Verification ---")
        notes = []
        potential_names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", user_input))
        if potential_names and self.world:
            found = []; unknown = []
            known_lower = {p.get('name','').lower():i for i,p in self.world.state.entities.items() if p.get('name')}
            for name in potential_names:
                if name.lower() in known_lower: found.append(f"'{name}'")
                elif name.lower()!=self.name.lower() and name.lower()!="adventurer": unknown.append(f"'{name}'") # Crude filter
            if found: notes.append(f"Known: {', '.join(found)}.")
            if unknown: notes.append(f"Unknown?: {', '.join(unknown)}.")
        final = "Verification Notes: " + (" ".join(notes) if notes else "None.")
        logger.debug(f"Simplified Verification: {final}")
        return final

    def _update_internal_state_from_llm(self, llm_response: dict):
        """Updates agent's internal state based on LLM response."""
        internal_update = llm_response.get("internal_state_update", {})
        if isinstance(internal_update, dict):
            # Emotion
            new_emotion = internal_update.get("new_emotion")
            if new_emotion and isinstance(new_emotion, str):
                self.current_state.affective_state.update(new_emotion=new_emotion.lower())
                logger.info(f"Agent Emotion -> {new_emotion}")
            # Memory
            new_memory = internal_update.get("new_memory")
            if new_memory and isinstance(new_memory, str):
                self.memory.add_memory(Observation(f"[Self Thought] {new_memory}", importance=5.0))
                logger.info(f"Agent Added Memory: {new_memory[:60]}...")
            # Goal
            goal_update = internal_update.get("goal_update", {})
            if isinstance(goal_update, dict):
                 desc = goal_update.get("goal_description"); status = goal_update.get("new_status")
                 if desc and status:
                      # TODO: Implement set_or_add_goal in MotivationSystem for robustness
                      self.motivation.set_goal_status(desc, status)
                      logger.info(f"Goal '{desc[:50]}' -> '{status}'.")

    def _determine_action_from_llm(self, llm_response: dict) -> dict:
        """Determines the final action, potentially handling world queries."""
        action_data = llm_response.get("action", {"type": "stay_in_conversation"})
        if not isinstance(action_data, dict): action_data = {"type": "stay_in_conversation"}

        # Handle world query (if agent decides to)
        if action_data.get("type") == "query_world" and self.world:
             details = action_data.get("details"); target = action_data.get("target")
             if details and target and isinstance(details, str) and isinstance(target, str):
                 logger.info(f"Agent action: Querying world -> '{target}' about '{details}'")
                 # World query logic now simpler, maybe just direct lookup or LLM call
                 world_answer = "Unknown" # Placeholder
                 # TODO: Implement world query logic if needed, e.g.:
                 # world_answer = await self.world.query_entity_property_or_llm(target, details)
                 answer_text = f"Checked on '{target}'. Regarding '{details}': {world_answer}."
                 # Use asyncio.create_task to not block response generation
                 asyncio.create_task(self.perceive(answer_text, source="WorldQuery", importance=6.0))
                 action_data = {"type": "stay_in_conversation", "details": "Processed world query."} # Default after query
             else: logger.warning("'query_world' missing target/details."); action_data = {"type": "stay_in_conversation"}

        # Update displayed active goal
        self.current_state.active_goal = self.motivation.get_highest_priority_goal(self.personality, self.current_state.affective_state)
        self._log_state(logging.DEBUG)
        return action_data

    # --- Setup & Persistence Methods ---
    def set_world(self, world: 'World'): self.world = world; logger.info(f"Agent {self.name} linked to World.")

    async def ensure_assistant_exists(self, default_model: str = DEFAULT_MODEL, temperature: float = 0.7) -> bool:
         """Ensures OpenAI Assistant exists, creates if needed, and saves ID back to config."""
         if not self.agent_manager_ref: logger.error("AgentManager ref missing!"); return False
         if self.assistant_id:
             logger.debug(f"Verifying existing assistant ID: {self.assistant_id}")
             details = await self.agent_manager_ref.load_agent(self.assistant_id)
             if details: logger.info(f"Confirmed assistant {self.assistant_id}."); return True
             else: logger.warning(f"Assistant ID {self.assistant_id} invalid. Creating new."); self.assistant_id = None
         if not self.assistant_id:
             logger.info(f"Creating new assistant for {self.name}...")
             instructions = self.description # Use initial description as base instructions
             if not instructions: logger.error(f"Agent {self.name} needs description."); return False
             new_id = await self.agent_manager_ref.create_agent(
                 name=self.name, instructions=instructions, model=default_model,
                 temperature=temperature, metadata={"agent_name": self.name})
             if new_id:
                 self.assistant_id = new_id
                 logger.info(f"New assistant created: {self.assistant_id}. SAVING ID to config dict.")
                 # --- Save ID back to the loaded config dictionary ---
                 if self._config_data:
                      self._config_data.setdefault('profile', {})['assistant_id'] = new_id
                      # Optionally save the YAML file immediately (can cause issues if run in parallel)
                      # if self._config_file_path:
                      #      logger.info(f"Saving updated config with new ID to {self._config_file_path}")
                      #      save_agent_state_to_yaml(self._config_data, self._config_file_path)
                 # ----------------------------------------------------
                 return True
             else: logger.error(f"Failed create assistant for {self.name}."); return False
         return True # Should already have returned True if ID existed

    async def initialize_conversation(self) -> bool:
         """Creates or validates OpenAI thread."""
         # (Implementation remains the same - uses self.agent_manager_ref)
         if not self.agent_manager_ref: logger.error("AgentManager ref missing!"); return False
         if self.thread_id:
             logger.info(f"Verifying thread: {self.thread_id}.")
             try: await self.agent_manager_ref.client.beta.threads.retrieve(self.thread_id); logger.info("Thread valid."); return True
             except Exception as e: logger.warning(f"Thread invalid ({e}). Creating new."); self.thread_id = None
         if not self.assistant_id:
             logger.critical(f"Agent {self.name} needs assistant_id for thread.");
             # Try creating assistant if missing? Risky if it fails repeatedly.
             if not await self.ensure_assistant_exists(): return False # Try creating if missing
             if not self.assistant_id: return False # Still missing after create attempt

         logger.info(f"Creating new thread for agent {self.name} ({self.assistant_id})...");
         meta = {"agent_name": self.name, "assistant_id": self.assistant_id}
         thread_id = await self.agent_manager_ref.create_thread(metadata=meta)
         if thread_id: self.thread_id = thread_id; logger.info(f"Thread {self.thread_id} created."); return True
         else: logger.error(f"Failed create thread."); return False

    async def end_conversation(self, delete_thread: bool = True):
        """Ends conversation, optionally deleting thread."""
        # (Implementation remains the same - uses self.agent_manager_ref)
        if not self.agent_manager_ref: logger.error("AgentManager ref missing!"); return
        if not self.thread_id: logger.info(f"No active thread for {self.name} to end."); return
        tid = self.thread_id; self.thread_id = None
        logger.info(f"Ending conversation. Cleared thread ref '{tid}'.")
        if delete_thread:
            logger.warning(f"Requesting deletion of thread {tid}...")
            deleted = await self.agent_manager_ref.delete_thread(tid)
            logger.info(f"Thread {tid} deletion result: {deleted}")

    def get_state_dict(self) -> dict[str, Any]:
         """Returns agent state (uses internal config dict which now includes saved ID)."""
         # Update the dict with current dynamic state before returning
         self._config_data['personality'] = self.personality.to_dict()
         self._config_data['motivation'] = self.motivation.to_dict()
         self._config_data['memory'] = self.memory.to_dict() # Includes all memories
         self._config_data['current_state'] = self.current_state.to_dict()
         self._config_data.setdefault('profile', {})['assistant_id'] = self.assistant_id # Ensure ID is there
         return self._config_data

    def save_state_to_yaml(self, file_path: Optional[str] = None) -> bool:
         """Saves agent state to YAML (uses stored path if available)."""
         target_path = file_path or self._config_file_path
         if not target_path:
             logger.error(f"Cannot save agent state for {self.name}: No file path provided or stored.")
             return False
         state_dict = self.get_state_dict() # Get the updated internal dict
         logger.info(f"Saving agent '{self.name}' state to {target_path}")
         # Ensure profile exists before accessing assistant_id for logging
         profile_dict = state_dict.get('profile', {})
         logger.debug(f"Saving state with assistant_id: {profile_dict.get('assistant_id')}")
         return save_agent_state_to_yaml(state_dict, target_path)

    @classmethod
    def load_from_yaml(cls, file_path: str, agent_manager: AgentManager, world: Optional['World'] = None) -> Optional['Agent']:
         """Loads agent config, passes file path for potential saving."""
         logger.info(f"Attempting load agent config: {file_path}")
         config = load_agent_config_from_yaml(file_path)
         if config:
             # Pass the file_path to the constructor
             try: return cls(config, agent_manager, world, config_file_path=file_path)
             except Exception as e: logger.error(f"Failed init Agent from {file_path}: {e}", exc_info=True); return None
         else: logger.error(f"Failed load config dict from {file_path}"); return None

    async def delete_persistent_assistant(self) -> bool:
        """Deletes the associated OpenAI assistant permanently."""
        # (Implementation remains the same - uses self.agent_manager_ref)
        if not self.agent_manager_ref: logger.error("AgentManager ref missing!"); return False
        if not self.assistant_id: logger.warning(f"Agent {self.name} has no assistant_id."); return False
        logger.warning(f"Requesting PERMANENT deletion of assistant {self.assistant_id}...")
        deleted = await self.agent_manager_ref.delete_assistant(self.assistant_id)
        if deleted: logger.info(f"Assistant deleted."); self.assistant_id = None; return True
        else: logger.error(f"Failed delete assistant."); return False
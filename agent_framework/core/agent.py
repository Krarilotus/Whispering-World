# agent_framework/core/agent.py
import time
import logging
import json
import re
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Set, TYPE_CHECKING # Use list, dict etc.

# Relative imports
from .memory import MemoryStream, MemoryObject, Observation # Removed GeneratedFact for now
from .personality import Personality, AffectiveState
from .motivation import MotivationSystem, Goal
from .state import CurrentState
from ..llm.llm_interface import LLM_API_Interface # Consolidated interface
if TYPE_CHECKING: from ..world.world import World
from ..file_utils.yaml_handler import save_agent_state_to_yaml, load_agent_config_from_yaml
from ..api.agent_api import AgentManager, DEFAULT_MODEL # Import manager for setup/teardown
from ..llm import prompts
import openai

logger = logging.getLogger(__name__)

class Agent:
    """ Agent class using consolidated interface, simplified verification, explicit rules in prompt. """
    def __init__(self, config: dict[str, Any], agent_manager: AgentManager, world: Optional['World'] = None, config_file_path: Optional[str] = None):
        self._config_data = config # Store original config data
        self._config_file_path = config_file_path # Store path to save back to
        profile = self._config_data.get('profile', {})
        self.name: str = profile.get('name', 'Unnamed Agent')
        self.description: str = profile.get('description', 'An AI agent.') # Base instructions
        self.agent_manager_ref = agent_manager # For assistant/thread management
        self.world = world
        self.assistant_id: Optional[str] = profile.get('assistant_id') # Load existing ID
        self.thread_id: Optional[str] = None
        self.llm_interface : Optional[LLM_API_Interface] = None # Must be set via set_llm_interface

        logger.info(f"Initializing Agent: {self.name}")
        self.personality = Personality.from_dict(self._config_data.get('personality', {}))
        self.motivation = MotivationSystem.from_dict(self._config_data.get('motivation', {}))
        # Assumes memory.py handles vector index init/load
        self.memory = MemoryStream.from_dict(self._config_data.get('memory', {}))
        self.current_state = CurrentState.from_dict(self._config_data.get('current_state', {}))

        # Store last user input for rule checks
        self.last_user_input: str = "[Start of conversation]"

        # Add initial memories
        if not self.memory._memories and 'initial_memory' in self._config_data:
             initial_mems = self._config_data['initial_memory']; count = 0
             if isinstance(initial_mems, list):
                  for mem_text in initial_mems:
                      if isinstance(mem_text, str): self.memory.add_memory(Observation(mem_text, importance=7.0)); count+=1
                  if count > 0: logger.info(f"Added {count} initial memories.")

        self.motivation.generate_goals_from_state(self.personality)
        logger.info(f"Agent {self.name} init complete. World: {'Yes' if self.world else 'No'}")
        self._log_state(logging.DEBUG)

    def set_llm_interface(self, llm_interface: LLM_API_Interface):
        self.llm_interface = llm_interface; logger.info(f"LLM Interface linked for Agent {self.name}")

    # --- Core Methods ---
    async def perceive(self, observation_text: str, source: str = "external", importance: float = 5.0):
        """Adds observation to memory stream."""
        obs = Observation(f"[{source.upper()}] {observation_text}", importance=importance)
        self.memory.add_memory(obs) # Handles embedding internally
        logger.info(f"Agent {self.name} perceived: {observation_text[:100]}...")
        self._log_state(logging.DEBUG)

    async def think_and_respond(self, user_input: str) -> dict[str, Any]:
        """ Main agent processing loop: Perceive -> Minimal Verify -> Run LLM -> Update State """
        start_time = time.time()
        logger.info(f">>> Agent {self.name} processing input: '{user_input[:100]}...'")
        if not self.llm_interface: logger.error("LLM Interface missing!"); return {"action": {}, "dialogue": "[ERR]", "reasoning":"No Interface"}
        if not self.thread_id:
             if not await self.initialize_conversation(): return {"action":{}, "dialogue":"[ERR]", "reasoning":"Thread Fail"}

        # 1. Perceive Input
        await self.perceive(user_input, source="PlayerInput", importance=6.0)

        # 2. Simplified Verification / Context Generation
        verification_notes = self._perform_simplified_verification(user_input)

        # 3. Construct Dynamic Prompt (Context + Task) - Passed as additional_instructions
        context_prompt = self._construct_dynamic_context_prompt(user_input) # Get focused context
        task_prompt = prompts.get_react_to_input_prompt(
             user_input=user_input,
             verification_notes=verification_notes,
             previous_user_input=self.last_user_input # Pass previous input for rule checks
        )
        additional_instructions_prompt = f"{context_prompt}\n{task_prompt}"
        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Additional Instructions Prompt (start):\n{additional_instructions_prompt[:1000]}...")

        # 4. Add ONLY the actual user input to the thread history via API
        msg_added_ok = await self.agent_manager_ref.add_message_to_thread(self.thread_id, user_input, role="user")
        if not msg_added_ok:
             logger.error(f"Failed add user message '{user_input[:50]}' to thread {self.thread_id}");
             return {"action": {}, "dialogue": "[ERR: Msg Send]", "reasoning":"Failed add_message"}

        # 5. Run LLM (passing dynamic context+task via additional_instructions)
        llm_response = await self.llm_interface.generate_json_response(
            assistant_id=self.assistant_id,
            thread_id=self.thread_id,
            additional_instructions=additional_instructions_prompt, # Pass dynamic prompt here
            task_description=f"Agent {self.name} response generation"
        )

        # 6. Process Response & Update State
        if not llm_response: # Fallback
            logger.error(f"Agent {self.name}: Failed get valid response from LLM Interface.")
            return {"action": {"type": "stay_in_conversation"}, "dialogue": "[Internal error occurred.]", "reasoning": "LLM Interface Error / Invalid JSON"}

        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Response Dict: {json.dumps(llm_response, indent=2)}")
        self._update_internal_state_from_llm(llm_response)
        action_data = self._determine_action_from_llm(llm_response)

        # 7. Update last input *after* processing response
        self.last_user_input = user_input

        # 8. Timing & Return
        processing_time = time.time() - start_time
        action_type = action_data.get('type', 'unknown')
        logger.info(f"<<< Agent {self.name} responded in {processing_time:.2f}s. Action: {action_type}")
        if processing_time > 30: logger.warning(f"Agent response took >30s ({processing_time:.1f}s)") # Reduced threshold
        return {"action": action_data, "dialogue": llm_response.get("dialogue", ""), "reasoning": llm_response.get("reasoning", "N/A")}

    # --- Internal Logic Methods ---
    def _construct_dynamic_context_prompt(self, current_input: str) -> str:
        """Constructs a *leaner* context prompt focusing on dynamic state and relevant memories."""
        # World Summary (Concise)
        world_summary = "World unavailable."
        if self.world:
             try: world_summary = self.world.state.get_summary(max_entities=10, max_rels=3, max_len=400)
             except Exception: world_summary = "Error getting world context."

        # Relevant Memory Summary (using Vector Search)
        memory_summary = "No relevant memories triggered by input."
        query_context = current_input # Use current input to find relevant memories
        if self.current_state.active_goal: query_context += f" | Goal: {self.current_state.active_goal.description}" # Add goal context
        query_context = query_context.strip()

        if self.memory.faiss_index and query_context: # Check if vector search ready
             try:
                 relevant_memories = self.memory.retrieve_relevant_memories(query_context, top_k=3) # TOP 3-5 Relevant
                 if relevant_memories:
                     # Use __repr__ for formatting to include timestamp/importance easily
                     mem_summary_lines = [f"- {mem}" for mem in relevant_memories]
                     memory_summary = "Most Relevant Memories:\n" + "\n".join(mem_summary_lines)
             except Exception as e:
                 logger.error(f"Error during memory retrieval: {e}", exc_info=True)
                 memory_summary = "Error retrieving relevant memories."
        else:
             # Fallback if vector search not ready (provide *recent* instead)
             memory_summary = "Recent Memories (Fallback):\n" + self.memory.get_memory_summary(max_memories=5, max_length=400)
             logger.debug("Using recent memories for context (vector search unavailable/no query).")

        # Call the prompt function with ONLY DYNAMIC and RELEVANT info
        # Base persona/description lives in Assistant instructions, not repeated here
        prompt = prompts.get_base_context_prompt(
            agent_name=self.name,
            agent_description=self.description, # Provide base description as reminder
            personality_description=self.personality.get_description(), # Include personality traits
            motivation_description=self.motivation.get_state_description(), # Current needs/goals are dynamic
            current_state_description=self.current_state.get_state_description(), # Location, emotion etc. are dynamic
            memory_summary=memory_summary, # Focused memories
            world_context_summary=world_summary # Focused world state
        )
        return prompt

    def _log_state(self, level=logging.DEBUG):
        """Logs the current state of all components."""
        # (Implementation remains the same)
        if logger.isEnabledFor(level):
             max_len=1500; log_str=f"--- {self.name} State ---\n"; log_str+=self.personality.get_description()+"\n"; log_str+=self.motivation.get_state_description()+"\n"; log_str+=self.current_state.get_state_description()+"\n"; log_str+=f"Memory (Count:{len(self.memory._memories)}):\n{self.memory.get_memory_summary(max_memories=5, max_length=300)}\n"; log_str+="-"*20; logger.log(level, log_str[:max_len]+('...' if len(log_str)>max_len else ''))

    def _perform_simplified_verification(self, user_input: str) -> str:
        """ Performs minimal local checks and generates brief verification notes. """
        logger.debug(f"--- Running Simplified Verification ---")
        notes = []
        # Basic Entity Spotting (crude)
        potential_names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", user_input))
        if potential_names and self.world:
            found = []; unknown = []
            known_names_lower = {p.get('name','').lower():i for i,p in self.world.state.entities.items() if p.get('name')}
            for name in potential_names:
                if name.lower() in known_names_lower: found.append(f"'{name}'")
                elif name.lower() not in [self.name.lower(), "adventurer"]: unknown.append(f"'{name}'") # Filter self/player
            if found: notes.append(f"Known entities mentioned: {', '.join(found)}.")
            if unknown: notes.append(f"Potentially unknown entities mentioned: {', '.join(unknown)}.")
        # Memory contradiction check could be added here if needed, but keep simple
        final_notes = "Verification Notes: " + (" ".join(notes) if notes else "No specific items flagged.")
        logger.debug(f"Simplified Verification Result: {final_notes}")
        return final_notes

    def _update_internal_state_from_llm(self, llm_response: dict):
        """Updates agent's internal state based on LLM response."""
        internal_update = llm_response.get("internal_state_update", {})
        if not isinstance(internal_update, dict): return # Ignore if not a dict

        # Emotion
        new_emotion = internal_update.get("new_emotion")
        if new_emotion and isinstance(new_emotion, str):
            self.current_state.affective_state.update(new_emotion=new_emotion.lower())
            logger.info(f"Agent Emotion -> {new_emotion}")
        # Memory
        new_memory = internal_update.get("new_memory")
        if new_memory and isinstance(new_memory, str):
            # Assign importance based on context? Default for now.
            self.memory.add_memory(Observation(f"[Self Thought] {new_memory}", importance=5.0))
            logger.info(f"Agent Added Memory: {new_memory[:60]}...")
        # Goal
        goal_update = internal_update.get("goal_update", {})
        if isinstance(goal_update, dict):
             goal_desc = goal_update.get("goal_description"); goal_status = goal_update.get("new_status")
             if goal_desc and goal_status:
                  if hasattr(self.motivation, 'set_or_add_goal'):
                       self.motivation.set_or_add_goal(goal_desc, goal_status, source="llm_suggestion")
                  else: # Basic fallback if method not implemented
                       logger.warning("MotivationSystem lacks set_or_add_goal, using set_goal_status.")
                       self.motivation.set_goal_status(goal_desc, goal_status) # Only updates existing
                  logger.info(f"Goal Suggestion: '{goal_desc[:50]}' -> '{goal_status}'.")

    def _determine_action_from_llm(self, llm_response: dict) -> dict:
        """Determines the final action, potentially handling world queries."""
        action_data = llm_response.get("action", {"type": "stay_in_conversation"})
        if not isinstance(action_data, dict): action_data = {"type": "stay_in_conversation"}

        # Handle world query action (remains potentially useful)
        if action_data.get("type") == "query_world" and self.world:
             details = action_data.get("details"); target = action_data.get("target")
             if details and target:
                 logger.info(f"Agent action: Querying world -> '{target}' about '{details}'")
                 # Use world's direct lookup first
                 entity_props = self.world.get_entity_properties(target)
                 world_answer = entity_props.get(details, "Unknown Property") if entity_props else "Unknown Entity"
                 # If direct fails, optionally use LLM query (can be slow)
                 # if "Unknown" in world_answer:
                 #     world_answer = await self.world.query_world_via_llm(details, target) or "Unknown (LLM)"

                 answer_text = f"Checked world about '{details}' for '{target}'. Result: {world_answer if world_answer is not None else 'Unknown'}."
                 # Run perceive in background to avoid blocking return
                 asyncio.create_task(self.perceive(answer_text, source="WorldQuery", importance=6.0))
                 action_data = {"type": "stay_in_conversation", "details": "Processed world query."}
             else: logger.warning("'query_world' action missing target/details."); action_data = {"type": "stay_in_conversation"}

        # Update agent's displayed active goal
        self.current_state.active_goal = self.motivation.get_highest_priority_goal(self.personality, self.current_state.affective_state)
        self._log_state(logging.DEBUG)
        return action_data

    # --- Setup & Persistence Methods ---
    def set_world(self, world: 'World'): self.world = world; logger.info(f"Agent {self.name} linked.")

    async def ensure_assistant_exists(self, default_model: str = DEFAULT_MODEL, temperature: float = 0.7) -> bool:
         """Ensures OpenAI Assistant exists, verifies ID, or creates new and SAVES ID."""
         if not self.agent_manager_ref: logger.error("AgentManager ref missing!"); return False
         assistant_needs_saving = False
         if self.assistant_id:
             logger.debug(f"Verifying assistant ID: {self.assistant_id}")
             details = await self.agent_manager_ref.load_agent(self.assistant_id)
             if details: logger.info(f"Confirmed assistant {self.assistant_id}."); return True
             else: logger.warning(f"Assistant ID {self.assistant_id} invalid. Creating new."); self.assistant_id = None; self._config_data.setdefault('profile', {})['assistant_id'] = None # Clear from config dict too

         if not self.assistant_id:
             logger.info(f"Creating new assistant for {self.name}...")
             instr = self.description; # Use base description from config
             if not instr: logger.error(f"Agent {self.name} needs description."); return False
             new_id = await self.agent_manager_ref.create_agent(
                 name=self.name, instructions=instr, model=default_model,
                 temperature=temperature, metadata={"agent_name": self.name})
             if new_id:
                 self.assistant_id = new_id; logger.info(f"New assistant: {self.assistant_id}.");
                 # --- Store new ID in config dict for saving ---
                 self._config_data.setdefault('profile', {})['assistant_id'] = new_id
                 assistant_needs_saving = True # Flag that config should be saved later
                 # ---------------------------------------------
                 return True
             else: logger.error(f"Failed create assistant."); return False
         # We don't save YAML here, agent state is saved at end of game loop normally
         return True # Should have an ID

    async def initialize_conversation(self) -> bool:
        """Creates or validates OpenAI thread."""
        # (Implementation remains the same - uses self.agent_manager_ref)
        if not self.agent_manager_ref: logger.error("AgentManager ref missing!"); return False
        if self.thread_id:
            logger.info(f"Verifying thread: {self.thread_id}.")
            try: await self.agent_manager_ref.client.beta.threads.retrieve(self.thread_id); logger.info("Thread valid."); return True
            except Exception as e: logger.warning(f"Thread invalid ({e}). Creating new."); self.thread_id = None
        if not self.assistant_id: logger.critical(f"Agent needs assistant_id for thread."); return False
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
         tid = self.thread_id; self.thread_id = None; logger.info(f"Ending conversation. Cleared thread ref '{tid}'.")
         if delete_thread:
             logger.warning(f"Requesting deletion of thread {tid}...")
             deleted = await self.agent_manager_ref.delete_thread(tid)
             logger.info(f"Thread {tid} deletion result: {deleted}")

    def get_state_dict(self) -> dict[str, Any]:
         """Returns agent state (uses internal config dict which includes assistant_id)."""
         # Update the dict with current dynamic state before returning
         self._config_data['personality'] = self.personality.to_dict()
         self._config_data['motivation'] = self.motivation.to_dict()
         self._config_data['memory'] = self.memory.to_dict()
         self._config_data['current_state'] = self.current_state.to_dict()
         # Ensure profile and assistant_id are correctly represented
         profile_dict = self._config_data.setdefault('profile', {})
         profile_dict['name'] = self.name
         profile_dict['description'] = self.description
         profile_dict['assistant_id'] = self.assistant_id
         return self._config_data

    def save_state_to_yaml(self, file_path: Optional[str] = None) -> bool:
         """Saves agent state to YAML (uses stored path if available)."""
         target_path = file_path or self._config_file_path
         if not target_path: logger.error(f"Cannot save {self.name}: No file path."); return False
         state_dict = self.get_state_dict() # Get updated dict
         logger.info(f"Saving agent '{self.name}' state to {target_path}")
         return save_agent_state_to_yaml(state_dict, target_path)

    @classmethod
    def load_from_yaml(cls, file_path: str, agent_manager: AgentManager, world: Optional['World'] = None) -> Optional['Agent']:
         """Loads agent config, passes file path for potential saving."""
         logger.info(f"Attempting load agent config: {file_path}")
         config = load_agent_config_from_yaml(file_path)
         if config:
             if not config.get('profile', {}).get('assistant_id'): logger.warning(f"Config {file_path} missing 'profile.assistant_id'.")
             try: # Pass file path to constructor
                  return cls(config, agent_manager, world, config_file_path=file_path)
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
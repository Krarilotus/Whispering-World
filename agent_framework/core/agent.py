# agent_framework/core/agent.py
import time
import logging
import json # Import json for formatting assertion strings
from typing import Dict, Any, Optional, Tuple, List, Set, TYPE_CHECKING # Use modern types if preferred

# Relative imports for components within the package
from .memory import MemoryStream, MemoryObject, Observation, GeneratedFact
from .personality import Personality, AffectiveState
from .motivation import MotivationSystem, Goal
from .state import CurrentState
from ..llm.llm_interface import LLMInterface
# Import World conditionally to avoid definite circular dependency if World imports Agent
# although the current design shouldn't require that. TYPE_CHECKING is safer.
if TYPE_CHECKING:
    from ..world.world import World
from ..file_utils.yaml_handler import save_agent_state_to_yaml, load_agent_config_from_yaml
# Import AgentManager from adjusted path
from ..api.agent_api import AgentManager

logger = logging.getLogger(__name__)

class Agent:
    """
    The main AI Agent class orchestrating personality, motivation, memory,
    state, world interaction, and LLM communication via the consistency barrier.
    """
    def __init__(self, config: dict[str, Any], agent_manager: AgentManager, world: Optional['World'] = None):
        """
        Initializes the agent from a configuration dictionary.

        Args:
            config: Dictionary containing agent setup data (usually loaded from YAML).
            agent_manager: An instance of AgentManager for API calls.
            world: An optional instance of the World object for interaction.
        """
        profile = config.get('profile', {})
        self.name: str = profile.get('name', 'Unnamed Agent')
        self.description: str = profile.get('description', 'An AI agent.')
        self.agent_manager = agent_manager

        # OpenAI specific IDs - Managed externally but stored here
        self.assistant_id: Optional[str] = profile.get('assistant_id')
        self.thread_id: Optional[str] = None # Thread is usually created per conversation

        self.world = world # Store reference to the world object

        logger.info(f"Initializing Agent: {self.name}")

        # Initialize Core Components from config
        self.personality = Personality.from_dict(config.get('personality', {}))
        self.motivation = MotivationSystem.from_dict(config.get('motivation', {}))
        self.memory = MemoryStream.from_dict(config.get('memory', {})) # Load initial memories if provided
        self.current_state = CurrentState.from_dict(config.get('current_state', {}))

        # Add initial memories from config if not already loaded via memory dict
        if not self.memory._memories and 'initial_memory' in config:
             initial_mems = config['initial_memory']
             if isinstance(initial_mems, list):
                  mem_count = 0
                  for mem_text in initial_mems:
                      if isinstance(mem_text, str):
                          # Initial memories are often core beliefs/backstory, treat as important
                          imp = 7.0 + (2.0 * self.personality.ocean_scores.get('conscientiousness', 0.5)) # Slightly more important if conscientious
                          self.memory.add_memory(Observation(mem_text, importance=imp))
                          mem_count += 1
                  logger.info(f"Added {mem_count} initial memories from config.")

        # Initialize LLM Interface (needs a reference back to self to access state)
        self.llm_interface = LLMInterface(agent_manager, self)

        # Generate initial goals based on loaded state
        self.motivation.generate_goals_from_state(self.personality)
        logger.info(f"Agent {self.name} initialization complete. World link: {'Yes' if self.world else 'No'}")
        self._log_state(logging.DEBUG) # Log initial state if debugging

    # --- Core Methods ---

    async def perceive(self, observation_text: str, source: str = "external", importance: float = 5.0):
        """Adds a new observation to the agent's memory."""
        obs = Observation(f"[{source.upper()}] {observation_text}", importance=importance)
        self.memory.add_memory(obs)
        logger.info(f"Agent {self.name} perceived: {observation_text[:100]}...")
        # Future: Could trigger immediate emotional reaction based on perception rules
        # self._update_affect_from_observation(obs)
        self._log_state(logging.DEBUG) # Log state after perception if debugging

    async def think_and_respond(self, user_input: str) -> dict[str, Any]:
        """
        The main agent processing loop for reacting to an external input.
        Includes perception, consistency check, decision, and action generation.
        """
        start_time = time.time()
        logger.info(f">>> Agent {self.name} processing input: '{user_input[:100]}...'")

        # 1. Perceive the input
        # Determine importance based on source? Player input moderately important.
        await self.perceive(user_input, source="PlayerInput", importance=6.0)

        # 2. Consistency Barrier & Contextualization
        verification_context = await self._verify_and_contextualize_input(user_input)

        # 3. Decide and Act (includes LLM call for response generation)
        response_data = await self._decide_and_act(user_input, verification_context)

        # Optional: Trigger reflection periodically (e.g., based on time or memory importance sum)
        # if self.should_reflect(): # Implement this check
        #    await self.memory.reflect(self.llm_interface)

        end_time = time.time()
        processing_time = end_time - start_time
        action_type = response_data.get('action', {}).get('type', 'unknown')
        logger.info(f"<<< Agent {self.name} responded in {processing_time:.2f}s. Action: {action_type}")
        if processing_time > 45: # Log if processing takes unusually long
            logger.warning(f"Agent {self.name} response took >45s ({processing_time:.1f}s)")

        return response_data

    # --- Internal Logic Methods ---

    def _construct_base_prompt(self) -> str:
        """Constructs the common context part using the prompt function from prompts.py."""
        # Ensure prompts module is available (should be via llm_interface import structure)
        from ..llm import prompts
        # Call the prompt function from prompts.py
        prompt = prompts.get_base_context_prompt(
            agent_name=self.name,
            agent_description=self.description,
            personality_description=self.personality.get_description(),
            motivation_description=self.motivation.get_state_description(),
            current_state_description=self.current_state.get_state_description(),
            memory_summary=self.memory.get_memory_summary(max_memories=15, max_length=1000) # Pass memory summary
        )
        # World context is typically added specifically where needed, like in generate_agent_response
        return prompt

    def _log_state(self, level=logging.DEBUG):
        """Logs the current state of all components."""
        if logger.isEnabledFor(level):
             # Limit logging length to avoid flooding
             max_len = 1500
             log_str = f"--- Agent {self.name} State Snapshot ---\n"
             log_str += self.personality.get_description() + "\n"
             log_str += self.motivation.get_state_description() + "\n"
             log_str += self.current_state.get_state_description() + "\n"
             log_str += f"Memory Stream (Recent {min(10, len(self.memory._memories))}):\n{self.memory.get_memory_summary(max_memories=10, max_length=500)}\n"
             log_str += "------------------------------------"
             logger.log(level, log_str[:max_len] + ('...' if len(log_str) > max_len else ''))

    def _resolve_entity_description(self, description: str, verified_entities: dict[str, str]) -> Optional[str]:
        """
        Tries to resolve a description to a known entity ID (world state or just verified).
        Returns entity ID if resolved, None otherwise. (Helper for verification)
        """
        if not description: return None
        # Check if description itself is already a verified ID in this loop
        if description in verified_entities.values(): return description
        # Check if description maps to a verified ID in this loop
        if description in verified_entities: return verified_entities[description]

        # Check world state if available
        if self.world:
            # Call the world's resolution helper (assumes world object has this method)
            resolved_id = self.world._resolve_entity(description)
            if resolved_id: return resolved_id

        logger.debug(f"Could not resolve description '{description}' to known entity ID.")
        return None

    def _remap_assertion_descriptions(self, assertion: dict, resolved_ids: dict[str, str]) -> Optional[dict]:
         """
         Helper to replace description keys (e.g., entity_desc) with resolved ID keys (e.g., entity_id)
         in an assertion dictionary. Returns None if a required remapping fails.
         """
         remapped = assertion.copy()
         mapping_success = True

         # --- Define mappings ---
         desc_to_id_map = {
             "entity_desc": "entity_id",
             "subject_desc": "subject_id",
             "object_desc": "object_id",
             # Add value_desc -> value ONLY IF we know value_desc refers to an entity
             # For now, handle value_desc separately
         }

         # --- Perform Mappings ---
         for desc_key, id_key in desc_to_id_map.items():
             if desc_key in remapped:
                 desc = remapped[desc_key]
                 if desc in resolved_ids:
                     remapped[id_key] = resolved_ids[desc]
                 else:
                     logger.error(f"CRITICAL: Could not find resolved ID for required description '{desc}' ({desc_key}) in assertion {assertion}. Aborting remapping.")
                     mapping_success = False
                     break # Cannot proceed without resolving required entities
                 remapped.pop(desc_key, None) # Remove original key

         if not mapping_success: return None

         # --- Handle value_desc specifically ---
         if "value_desc" in remapped:
              desc = remapped["value_desc"]
              if desc in resolved_ids:
                   # It resolved to an entity ID, use that as the value
                   remapped["value"] = resolved_ids[desc]
              else:
                   # Assume it's a literal value if not resolved
                   remapped["value"] = desc
                   logger.debug(f"Assuming value_desc '{desc}' is literal in {assertion}")
              remapped.pop("value_desc", None) # Remove original key

         return remapped

    async def _verify_and_contextualize_input(self, user_input: str) -> str:
        """
        Implements the Consistency Barrier using atomic assertions and world interaction.
        Returns formatted verification context string for the main LLM prompt.
        """
        logger.info(f"--- Running Consistency Barrier (Atomic Assertions) for input: '{user_input[:100]}' ---")
        verification_context_lines: list[str] = ["\n--- Input Verification Context ---"]

        # 1. Extract Atomic Assertions
        atomic_assertions = await self.llm_interface.extract_atomic_assertions(user_input)
        if not atomic_assertions: # Handles None or empty list
            logger.info("No atomic assertions extracted from input.")
            verification_context_lines.append("No specific assertions extracted to verify.")
            verification_context_lines.append("--- End Verification Context ---")
            return "\n".join(verification_context_lines)

        # Log extracted assertions clearly
        try:
            assertions_str = json.dumps(atomic_assertions, indent=2)
        except TypeError:
            assertions_str = str(atomic_assertions)
        verification_context_lines.append(f"Extracted Assertions:\n{assertions_str}")
        logger.debug(f"Extracted Assertions:\n{assertions_str}")

        # Track entities verified/created within this loop to handle dependencies
        verified_entities_this_loop: dict[str, str] = {} # Map description -> entity_id

        # 2. Process Assertions Sequentially
        for i, assertion in enumerate(atomic_assertions):
            assertion_str = json.dumps(assertion) # For logging/context
            logger.debug(f"Processing Assertion {i+1}/{len(atomic_assertions)}: {assertion_str}")
            verification_entry = f"\nAssertion {i+1}: {assertion_str}\n"
            prerequisites_met = True
            resolved_ids_for_assertion: dict[str, str] = {} # Map desc -> ID for *this* assertion

            # --- 2a. Identify & Resolve Prerequisite Entities ---
            required_descs: list[tuple[str, Optional[str]]] = [] # list of (description, type_hint)
            assertion_type = assertion.get("type")

            # Extract necessary descriptions based on assertion type
            if assertion_type == "existence":
                 desc = assertion.get("entity_desc")
                 if desc: required_descs.append((desc, assertion.get("entity_type_hint")))
            elif assertion_type == "property":
                 desc = assertion.get("entity_desc")
                 if desc: required_descs.append((desc, assertion.get("entity_type_hint")))
                 # We don't automatically assume value_desc is an entity here
            elif assertion_type == "relationship":
                 subj_desc = assertion.get("subject_desc")
                 obj_desc = assertion.get("object_desc")
                 if subj_desc: required_descs.append((subj_desc, assertion.get("subject_type_hint")))
                 if obj_desc: required_descs.append((obj_desc, assertion.get("object_type_hint")))

            # Check world connection
            if not self.world:
                 verification_entry += "  Status: Skipped (World connection missing).\n"
                 prerequisites_met = False
            else:
                # Resolve each required description
                for desc, type_hint in required_descs:
                    resolved_id = self._resolve_entity_description(desc, verified_entities_this_loop)

                    if resolved_id:
                        resolved_ids_for_assertion[desc] = resolved_id
                        logger.debug(f"  Prerequisite '{desc}': Resolved to existing/verified ID '{resolved_id}'.")
                        continue # Prerequisite exists

                    # --- Prerequisite Missing: Check Plausibility & Attempt Creation ---
                    logger.info(f"  Prerequisite entity '{desc}' not found. Checking world plausibility...")
                    # Frame the plausibility check clearly
                    plausibility_claim = f"An entity described as '{desc}' (type hint: {type_hint or 'unknown'}) exists in this world."
                    pl_result = await self.llm_interface.check_claim_plausibility(plausibility_claim, assessment_target="world")

                    if pl_result and pl_result[0]: # is_plausible
                        logger.info(f"  Entity '{desc}' deemed plausible by agent. Requesting world creation.")
                        # World generates defaults & adds entity
                        new_id = await self.world.add_entity(desc, type_hint)
                        if new_id:
                             verification_entry += f"  Prerequisite '{desc}': UNKNOWN, plausible -> CREATED by world as ID '{new_id}'.\n"
                             verified_entities_this_loop[desc] = new_id # Track globally
                             resolved_ids_for_assertion[desc] = new_id # Track locally
                        else:
                             verification_entry += f"  Prerequisite '{desc}': UNKNOWN, plausible, but world FAILED to create.\n"
                             prerequisites_met = False; break # Critical failure
                    else:
                        reason = pl_result[1] if pl_result else "Plausibility check failed"
                        verification_entry += f"  Prerequisite '{desc}': UNKNOWN and deemed IMPLAUSIBLE by agent. Reason: {reason}. Cannot proceed.\n"
                        prerequisites_met = False; break # Stop

            # --- 2b. Verify Assertion if Prerequisites Met ---
            if prerequisites_met and self.world:
                # Remap assertion to use resolved IDs before proposing
                mapped_assertion = self._remap_assertion_descriptions(assertion, resolved_ids_for_assertion)

                if not mapped_assertion: # Check if remapping failed critically
                     verification_entry += "  Status: Skipped (Failed to resolve critical entity descriptions to IDs).\n"
                     logger.error(f"Critical failure in remapping descriptions for assertion: {assertion}")
                else:
                    # Check if assertion is agent-internal or world-objective (could be refined)
                    # Simple heuristic: existence, world-object properties, world-object relationships are objective.
                    # Agent properties (mood, thoughts) might be internal. For now, assume all extracted are objective if prerequisites met.
                    is_objective_assertion = True # Default assumption for now

                    if is_objective_assertion:
                        logger.debug(f"  Proposing remapped objective assertion to world: {mapped_assertion}")
                        accepted = await self.world.propose_assertion(mapped_assertion, self.name)
                        status = "Applied to World" if accepted else "Rejected by World (e.g., inconsistent)"
                        verification_entry += f"  World Verification Status: {status}\n"
                        if not accepted:
                            self.memory.add_memory(Observation(f"[World Event] My assertion '{json.dumps(mapped_assertion)}' was rejected by the world state.", importance=6.0))
                            logger.warning(f"World rejected proposed assertion: {mapped_assertion}")
                        # else: self.memory.add_memory(Observation(f"[World Event] My assertion '{json.dumps(mapped_assertion)}' was accepted.", importance=4.0))
                    # else: # Handle internal assertion verification (using memory.check_consistency)
                    #     logger.debug(f" Verifying internal assertion: {mapped_assertion}")
                    #     # ... internal verification logic ...

            elif not prerequisites_met:
                 verification_entry += f"  World Verification Status: Skipped (Prerequisites not met).\n"

            # Append entry for this assertion
            verification_context_lines.append(verification_entry.strip())
            if not prerequisites_met: break # Stop processing further assertions if one fails critically

        verification_context_lines.append("--- End Verification Context ---")
        final_context = "\n".join(verification_context_lines)
        logger.info(f"--- Consistency Barrier Complete. Final Context Length: {len(final_context)} ---")
        logger.debug(f"Final Verification Context:\n{final_context}")
        return final_context

    async def _decide_and_act(self, user_input: str, verification_context: str) -> dict[str, Any]:
        """
        Gets LLM response based on state and VERIFIED input context, parses, updates state, returns action.
        """
        logger.debug("--- Decision and Action Phase ---")
        llm_response = await self.llm_interface.generate_agent_response(user_input, verification_context)
        if not llm_response: # Fallback...
            return {"action": {"type": "stay_in_conversation"}, "dialogue": "[Confused]", "reasoning": "LLM Error"}

        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Response Dict: {json.dumps(llm_response, indent=2)}")

        # Update internal state... (emotion, memory, goals)
        internal_update = llm_response.get("internal_state_update", {})
        if isinstance(internal_update, dict):
            new_emotion = internal_update.get("new_emotion")
            if new_emotion and isinstance(new_emotion, str): self.current_state.affective_state.update(new_emotion=new_emotion.lower()); logger.info(f"Emotion -> {new_emotion}")
            new_memory_text = internal_update.get("new_memory")
            if new_memory_text and isinstance(new_memory_text, str): self.memory.add_memory(Observation(f"[Self] {new_memory_text}")); logger.info(f"Added memory: {new_memory_text[:50]}...")
            goal_update = internal_update.get("goal_update", {})
            if isinstance(goal_update, dict):
                 goal_desc = goal_update.get("goal_description"); goal_status = goal_update.get("new_status")
                 if goal_desc and goal_status in ["completed", "failed"]: self.motivation.set_goal_status(goal_desc, goal_status); logger.info(f"Goal '{goal_desc[:30]}' -> {goal_status}")

        # Determine Action...
        action_data = llm_response.get("action", {"type": "stay_in_conversation"})
        if not isinstance(action_data, dict): action_data = {"type": "stay_in_conversation"}

        # Handle query_world action...
        if action_data.get("type") == "query_world" and self.world:
             query_details = action_data.get("details") # What property to query
             target_desc = action_data.get("target") # Description of entity to query
             if query_details and target_desc and isinstance(query_details, str) and isinstance(target_desc, str):
                  logger.info(f"Agent action: Querying world -> '{target_desc}' about '{query_details}'")
                  entity_id = self._resolve_entity_description(target_desc, {}) # Resolve target
                  world_answer = f"Could not resolve entity '{target_desc}'"
                  if entity_id:
                       world_answer = await self.world.query_entity_property(entity_id, query_details) # Query specific property
                  answer_text = f"Queried world about '{query_details}' for entity '{target_desc}' ({entity_id}). Answer: {world_answer if world_answer is not None else 'Unknown'}"
                  await self.perceive(answer_text, source="WorldQuery")
                  action_data = {"type": "stay_in_conversation", "details": "Processed world query result"} # Modify action
             else: logger.warning("'query_world' action missing target/details."); action_data = {"type": "stay_in_conversation"}

        self.current_state.active_goal = self.motivation.get_highest_priority_goal(self.personality, self.current_state.affective_state)
        self._log_state(logging.DEBUG)
        return {
            "action": action_data,
            "dialogue": llm_response.get("dialogue", ""),
            "reasoning": llm_response.get("reasoning", "N/A")
        }

    # --- Setup & Persistence Methods ---

    async def ensure_assistant_exists(self, default_model: str = "gpt-4o-mini", temperature: float = 0.7) -> bool:
        """Ensures OpenAI Assistant exists, verifies ID, or creates new."""
        if self.assistant_id:
            logger.debug(f"Verifying existing assistant ID: {self.assistant_id}")
            details = await self.agent_manager.load_agent(self.assistant_id)
            if details: logger.info(f"Confirmed assistant {self.assistant_id} for {self.name}."); return True
            else: logger.warning(f"Assistant ID {self.assistant_id} not found. Creating new."); self.assistant_id = None
        if not self.assistant_id:
            logger.info(f"Creating new assistant for {self.name}...")
            instructions = self.description
            if not instructions: logger.error(f"Agent {self.name} has no description. Cannot create."); return False
            logger.info(f"Creating assistant '{self.name}' model '{default_model}'...")
            new_id = await self.agent_manager.create_agent(name=self.name, instructions=instructions, model=default_model, temperature=temperature, metadata={"agent_name": self.name})
            if new_id: self.assistant_id = new_id; logger.info(f"New assistant for {self.name}: {self.assistant_id}"); return True
            else: logger.error(f"Failed to create assistant for {self.name}."); return False
        return True # Should have an ID if logic worked

    async def initialize_conversation(self) -> bool:
        """Creates or validates OpenAI thread."""
        if self.thread_id:
            logger.info(f"Agent {self.name} init conversation, has thread: {self.thread_id}. Verifying.")
            try: await self.agent_manager.client.beta.threads.retrieve(self.thread_id); logger.info(f"Reusing valid thread {self.thread_id}."); return True
            except Exception as e: logger.warning(f"Thread {self.thread_id} invalid: {e}. Creating new."); self.thread_id = None
        if not self.assistant_id: logger.critical(f"Agent {self.name} needs assistant_id to create thread."); return False
        logger.info(f"Creating new thread for agent {self.name} ({self.assistant_id})...");
        thread_metadata = {"agent_name": self.name, "assistant_id": self.assistant_id}
        thread_id = await self.agent_manager.create_thread(metadata=thread_metadata)
        if thread_id: self.thread_id = thread_id; logger.info(f"Thread {self.thread_id} created for {self.name}."); return True
        else: logger.error(f"Failed create thread for {self.name}."); return False

    async def end_conversation(self, delete_thread: bool = True):
         """Ends conversation, optionally deleting thread."""
         if not self.thread_id: logger.info(f"No active thread for {self.name} to end."); return
         thread_to_delete = self.thread_id; self.thread_id = None
         logger.info(f"Ending conversation for {self.name}. Cleared thread ref.")
         if delete_thread:
             logger.warning(f"Requesting deletion of thread {thread_to_delete}...")
             deleted = await self.agent_manager.delete_thread(thread_to_delete)
             if deleted: logger.info(f"Thread {thread_to_delete} deleted.")
             else: logger.warning(f"Failed delete thread {thread_to_delete} (may already be gone).")

    def get_state_dict(self) -> dict[str, Any]:
         """Returns agent state as dictionary for saving."""
         return {
             "profile": {"name": self.name, "description": self.description, "assistant_id": self.assistant_id},
             "personality": self.personality.to_dict(), "motivation": self.motivation.to_dict(),
             "memory": self.memory.to_dict(), "current_state": self.current_state.to_dict(),}

    def save_state_to_yaml(self, file_path: str) -> bool:
         """Saves agent state to YAML."""
         state_dict = self.get_state_dict()
         logger.info(f"Saving agent '{self.name}' state to {file_path}")
         return save_agent_state_to_yaml(state_dict, file_path)

    @classmethod
    def load_from_yaml(cls, file_path: str, agent_manager: AgentManager, world: Optional['World'] = None) -> Optional['Agent']:
         """Loads agent config from YAML, creates instance, links world."""
         logger.info(f"Attempting load agent config: {file_path}")
         config = load_agent_config_from_yaml(file_path)
         if config:
             if not config.get('profile', {}).get('assistant_id'): logger.warning(f"Loaded config {file_path} missing 'profile.assistant_id'.")
             return cls(config, agent_manager, world)
         else: logger.error(f"Failed load config dict from {file_path}"); return None

    async def delete_persistent_assistant(self) -> bool:
         """Deletes the associated OpenAI assistant permanently."""
         if not self.assistant_id: logger.warning(f"Agent {self.name} has no assistant_id to delete."); return False
         logger.warning(f"Requesting PERMANENT deletion of assistant {self.assistant_id} for agent {self.name}...")
         deleted = await self.agent_manager.delete_assistant(self.assistant_id)
         if deleted: logger.info(f"Assistant {self.assistant_id} deleted."); self.assistant_id = None; return True
         else: logger.error(f"Failed delete assistant {self.assistant_id}."); return False
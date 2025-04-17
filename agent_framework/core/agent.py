# agent_framework/core/agent.py
import time
import logging
import json
import re
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Set, TYPE_CHECKING # Use modern types

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

# Constants for checks
NON_SUBSTANTIVE_WORDS = {'ok', 'hmm', 'hm', '...', '.', 'yes', 'no', 'got it'}
NON_SUBSTANTIVE_LENGTH = 2 # Inputs with 2 or fewer words are likely non-substantive

class Agent:
    """ Agent class using framework-guided state updates before LLM call. """
    def __init__(self, config: dict[str, Any], agent_manager: AgentManager, world: Optional['World'] = None, config_file_path: Optional[str] = None):
        """ Initializes agent. Stores config path for saving back Assistant ID. """
        self._config_data = config
        self._config_file_path = config_file_path
        profile = self._config_data.get('profile', {})
        self.name: str = profile.get('name', 'Unnamed Agent')
        self.description: str = profile.get('description', 'An AI agent.')
        self.agent_manager_ref = agent_manager
        self.world = world
        self.assistant_id: Optional[str] = profile.get('assistant_id')
        self.thread_id: Optional[str] = None
        self.llm_interface : Optional[LLM_API_Interface] = None

        logger.info(f"Initializing Agent: {self.name}")
        self.personality = Personality.from_dict(self._config_data.get('personality', {}))
        self.motivation = MotivationSystem.from_dict(self._config_data.get('motivation', {}))
        # Assumes memory.py handles vector store init/load
        self.memory = MemoryStream.from_dict(self._config_data.get('memory', {}))
        self.current_state = CurrentState.from_dict(self._config_data.get('current_state', {}))

        # --- State for internal logic ---
        self.previous_input_was_non_substantive: bool = False
        # Find core vulnerability memories (e.g., daughter) for faster checks
        self.vulnerability_triggers: list[str] = self._identify_vulnerability_keywords()
        # --------------------------------

        # Add initial memories
        if not self.memory._memories and 'initial_memory' in self._config_data:
             initial_mems = self._config_data['initial_memory']
             if isinstance(initial_mems, list):
                  count = 0; # ... loop and add memories ...
                  for mem_text in initial_mems:
                      if isinstance(mem_text, str): self.memory.add_memory(Observation(mem_text, importance=7.0)); count+=1
                  if count > 0: logger.info(f"Added {count} initial memories.")

        self.motivation.generate_goals_from_state(self.personality)
        logger.info(f"Agent {self.name} init complete. Vulnerability triggers: {self.vulnerability_triggers}")
        self._log_state(logging.DEBUG)

    def _identify_vulnerability_keywords(self) -> list[str]:
        """ Simple heuristic to find keywords related to core vulnerabilities from flaws/bonds. """
        keywords = set()
        # Example: Look for 'daughter', 'Elara', 'injustice' in flaws/bonds text
        trigger_phrases = ["daughter", "elara", "unjust", "injustice", "locket"] # Example triggers for Warden
        if self.personality and self.personality.flaws:
             for flaw in self.personality.flaws:
                  flaw_lower = flaw.lower()
                  for phrase in trigger_phrases:
                       if phrase in flaw_lower: keywords.add(phrase)
        if self.motivation and self.motivation.bonds:
             for bond in self.motivation.bonds:
                  bond_lower = bond.lower()
                  for phrase in trigger_phrases:
                       if phrase in bond_lower: keywords.add(phrase)
        # Add keywords from important initial memories?
        for mem in self.memory.get_memories(): # Check all initially
             if mem.importance_score >= 8.0: # High importance
                mem_lower = mem.description.lower()
                for phrase in trigger_phrases:
                     if phrase in mem_lower: keywords.add(phrase)

        return list(keywords)

    def set_llm_interface(self, llm_interface: LLM_API_Interface): self.llm_interface = llm_interface; logger.info(f"LLM Interface linked for Agent {self.name}")
    async def perceive(self, observation_text: str, source: str = "external", importance: float = 5.0): # ... same ...
    def _log_state(self, level=logging.DEBUG): # ... same ...

    def _is_input_non_substantive(self, text: str) -> bool:
        """ Checks if input text is likely non-substantive based on keywords or length. """
        if text == "[Player remains silent]": return True
        text_lower = text.lower()
        if text_lower in NON_SUBSTANTIVE_WORDS: return True
        if len(text.split()) <= NON_SUBSTANTIVE_LENGTH: return True
        return False

    def _check_vulnerability_trigger(self, text: str) -> bool:
         """ Checks if input text contains vulnerability keywords. """
         text_lower = text.lower()
         for trigger in self.vulnerability_triggers:
              if trigger in text_lower:
                   logger.info(f"Vulnerability trigger '{trigger}' detected in input.")
                   return True
         return False

    async def _pre_llm_state_update(self, user_input: str):
        """ Updates agent state based on rules BEFORE the main LLM call. """
        logger.debug("--- Running Pre-LLM State Update ---")
        state_changed = False

        # Rule: Impatience -> Generate Leave Goal
        current_input_non_sub = self._is_input_non_substantive(user_input)
        if current_input_non_sub and self.previous_input_was_non_substantive:
            logger.info("Impatience Rule Triggered: Two non-substantive inputs.")
            self.motivation.set_or_add_goal(
                description="Leave conversation due to boredom/impatience",
                status="active",
                source="internal_rule:impatience",
                urgency=10.0 # Max urgency
            )
            state_changed = True
        # Update history *after* checking
        self.previous_input_was_non_substantive = current_input_non_sub

        # Rule: Vulnerability Trigger -> Change Emotion
        if self._check_vulnerability_trigger(user_input):
             logger.info("Vulnerability Rule Triggered: Setting emotion.")
             # Example: Make warden distressed/suspicious
             self.current_state.affective_state.update(new_emotion='distressed', valence_change=-0.3, arousal_change=0.1)
             state_changed = True
             # Optionally add a high-urgency goal like "Protect self from manipulation"
             # self.motivation.set_or_add_goal(...)

        # Add more rules here based on agent state, retrieved memories etc.
        # E.g., if low health -> add goal "Seek safety"

        if state_changed:
             logger.info("Internal state updated by pre-LLM rules.")
             self._log_state(logging.INFO) # Log updated state if it changed

    async def think_and_respond(self, user_input: str) -> dict[str, Any]:
        """ Main loop: Perceive -> Update State (Internal) -> Minimal Verify -> Run LLM -> Update State (LLM) """
        start_time = time.time()
        logger.info(f">>> Agent {self.name} processing input: '{user_input[:100]}...'")
        if not self.llm_interface: return {"action": {}, "dialogue": "[ERR: No Interface]", "reasoning":"No Interface"}
        if not self.thread_id:
             if not await self.initialize_conversation(): return {"action":{}, "dialogue":"[ERR: Thread Fail]", "reasoning":"Thread Init Fail"}

        # 1. Perceive Input
        await self.perceive(user_input, source="PlayerInput", importance=6.0)

        # 2. Pre-LLM State Update based on Rules/Triggers
        await self._pre_llm_state_update(user_input) # Updates internal state (emotion, goals)

        # 3. Simplified Verification (Local Checks)
        verification_notes = self._perform_simplified_verification(user_input)

        # 4. Construct Dynamic Prompt (using potentially updated state)
        dynamic_context_prompt = self._construct_dynamic_context_prompt(user_input) # Includes relevant mems, world, current state

        # 5. Construct Task Prompt (with updated rules)
        task_prompt = prompts.get_react_to_input_prompt(
             user_input=user_input,
             verification_notes=verification_notes,
             previous_user_input=self.last_user_input # Still useful for LLM analysis context
        )
        additional_instructions = f"{dynamic_context_prompt}\n{task_prompt}"
        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Additional Instructions (start):\n{additional_instructions[:1000]}...")

        # 6. Add *Actual User Input* to Thread History
        msg_added_ok = await self.agent_manager_ref.add_message_to_thread(self.thread_id, user_input, role="user")
        if not msg_added_ok: logger.error(f"Failed add user message to thread {self.thread_id}"); return {"action": {}, "dialogue": "[ERR: Msg Send]", "reasoning":"Failed add_message"}

        # 7. Run LLM (Single Call)
        llm_response = await self.llm_interface.generate_json_response(
            assistant_id=self.assistant_id,
            thread_id=self.thread_id,
            additional_instructions=additional_instructions, # Pass dynamic context + task
            task_description=f"Agent {self.name} response generation"
        )

        # 8. Process Response & Update State (from LLM suggestions)
        if not llm_response: # Fallback
            logger.error(f"Agent {self.name}: Failed get valid structured response from LLM Interface.")
            return {"action": {"type": "stay_in_conversation"}, "dialogue": "[Internal error]", "reasoning": "LLM Interface Error"}

        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"LLM Response Dict: {json.dumps(llm_response, indent=2)}")
        self._update_internal_state_from_llm(llm_response) # Apply LLM's suggested state changes
        action_data = self._determine_action_from_llm(llm_response) # Determine final action

        # 9. Update History & Timing
        self.last_user_input = user_input # Update AFTER processing
        processing_time = time.time() - start_time
        action_type = action_data.get('type', 'unknown')
        logger.info(f"<<< Agent {self.name} responded in {processing_time:.2f}s. Action: {action_type}")
        if processing_time > 30: logger.warning(f"Agent response took >30s ({processing_time:.1f}s)")
        return {"action": action_data, "dialogue": llm_response.get("dialogue", ""), "reasoning": llm_response.get("reasoning", "N/A")}

    def _construct_dynamic_context_prompt(self, current_input: str) -> str:
         # (Implementation remains the same as Response #55 - uses vector search etc.)
         # ... Gets world_summary, memory_summary (via vector search) ...
         # ... Calls prompts.get_base_context_prompt(...) ...
         world_summary="World unavailable.";
         if self.world: try: world_summary=self.world.state.get_summary(max_entities=15,max_rels=5,max_len=500) except Exception: world_summary="Error getting world."
         memory_summary="No relevant memories found."
         query_context=current_input; goal=self.current_state.active_goal; if goal: query_context+=f"|Goal:{goal.description}"
         if self.memory.faiss_index and query_context:
             try: relevant=self.memory.retrieve_relevant_memories(query_context.strip(),top_k=3);
             if relevant: memory_summary="Relevant Memories:\n"+"\n".join([f"- {m}" for m in relevant])
             except Exception as e: logger.error(f"Memory retrieval error: {e}"); memory_summary="Error retrieving memories."
         else: memory_summary = "Recent Memories (Fallback):\n" + self.memory.get_memory_summary(max_memories=5, max_length=400); logger.debug("Using recent mems for context.")
         prompt=prompts.get_base_context_prompt(agent_name=self.name,agent_description=self.description,personality_description=self.personality.get_description(),motivation_description=self.motivation.get_state_description(),current_state_description=self.current_state.get_state_description(),memory_summary=memory_summary,world_context_summary=world_summary)
         return prompt

    def _perform_simplified_verification(self, user_input: str) -> str:
         # (Implementation remains the same as Response #55 - basic entity check)
         logger.debug(f"--- Simplified Verification ---"); notes = []; pot_names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", user_input));
         if pot_names and self.world: found=[]; unknown=[]; known={p.get('name','').lower():i for i,p in self.world.state.entities.items() if p.get('name')}
         for n in pot_names:
             if n.lower() in known: found.append(f"'{n}'")
             elif n.lower() not in [self.name.lower(), "adventurer"]: unknown.append(f"'{n}'")
         if found: notes.append(f"Known: {','.join(found)}.")
         if unknown: notes.append(f"Unknown?: {','.join(unknown)}.")
         final = "Verification Notes: " + (" ".join(notes) if notes else "None.")
         logger.debug(f"Verification Result: {final}"); return final

    def _update_internal_state_from_llm(self, llm_response: dict):
        # (Implementation remains the same as Response #55)
         update = llm_response.get("internal_state_update", {});
         if isinstance(update, dict):
             emo = update.get("new_emotion"); mem = update.get("new_memory"); goal = update.get("goal_update", {})
             if emo: self.current_state.affective_state.update(new_emotion=emo.lower()); logger.info(f"Emotion->{emo}")
             if mem: self.memory.add_memory(Observation(f"[Self Thought] {mem}", importance=5.0)); logger.info(f"Added Memory:{mem[:50]}...")
             if isinstance(goal,dict): desc=goal.get("goal_description"); status=goal.get("new_status");
             if desc and status: self.motivation.set_or_add_goal(desc, status, source="llm"); logger.info(f"Goal Suggestion: '{desc[:50]}' -> '{status}'.")

    def _determine_action_from_llm(self, llm_response: dict) -> dict:
        # (Implementation remains the same as Response #55)
         action = llm_response.get("action", {"type": STAY_ACTION});
         if not isinstance(action, dict): action={"type": STAY_ACTION}
         if action.get("type")=="query_world" and self.world: #... handle query ...
             details=action.get("details"); target=action.get("target");
             if details and target: props=self.world.get_entity_properties(target); answer=props.get(details,"?") if props else "?"; text=f"Checked '{target}'.'{details}': {answer}"; asyncio.create_task(self.perceive(text,source="WorldQuery")); action={"type":STAY_ACTION,"details":"Queried world."};
             else: action={"type":STAY_ACTION}
         self.current_state.active_goal = self.motivation.get_highest_priority_goal()
         self._log_state(logging.DEBUG); return action

    # --- Setup & Persistence Methods ---
    # (Ensure these are fully implemented using self.agent_manager_ref)
    def set_llm_interface(self, llm_interface: LLM_API_Interface): self.llm_interface = llm_interface; logger.info(f"LLM Interface set for {self.name}")
    async def ensure_assistant_exists(self, default_model: str = DEFAULT_MODEL, temperature: float = 0.7) -> bool:
         if not self.agent_manager_ref: logger.error("Manager ref missing!"); return False
         assistant_needs_saving = False
         if self.assistant_id:
             details = await self.agent_manager_ref.load_agent(self.assistant_id);
             if details: logger.info(f"Confirmed assistant {self.assistant_id}."); return True
             else: logger.warning(f"Assistant ID {self.assistant_id} invalid. Creating new."); self.assistant_id = None; self._config_data.setdefault('profile', {})['assistant_id'] = None;
         if not self.assistant_id:
             logger.info(f"Creating new assistant for {self.name}...")
             instr = self.description; if not instr: logger.error(f"Agent needs description."); return False
             new_id = await self.agent_manager_ref.create_agent(name=self.name, instructions=instr, model=default_model, temperature=temperature, metadata={"agent_name": self.name})
             if new_id:
                 self.assistant_id = new_id; logger.info(f"New assistant: {self.assistant_id}.");
                 self._config_data.setdefault('profile', {})['assistant_id'] = new_id
                 assistant_needs_saving = True # Flag to save config later
                 # --- Don't save file here, save at end of game ---
                 # if self._config_file_path: save_agent_state_to_yaml(self._config_data, self._config_file_path)
                 return True
             else: logger.error(f"Failed create assistant."); return False
         return True # Should have ID

    async def initialize_conversation(self) -> bool:
         if not self.agent_manager_ref: return False
         if self.thread_id: try: await self.agent_manager_ref.client.beta.threads.retrieve(self.thread_id); return True except: self.thread_id = None
         if not self.assistant_id: if not await self.ensure_assistant_exists(): return False; # Try ensure assistant if missing
         if not self.assistant_id: return False # Still missing
         meta = {"agent_name": self.name, "assistant_id": self.assistant_id}; tid = await self.agent_manager_ref.create_thread(metadata=meta); self.thread_id = tid; return bool(tid)

    async def end_conversation(self, delete_thread: bool = True):
         if not self.agent_manager_ref or not self.thread_id: return
         tid = self.thread_id; self.thread_id = None; logger.info(f"Ending convo for {self.name} (Thread: {tid}).");
         if delete_thread: deleted = await self.agent_manager_ref.delete_thread(tid); logger.info(f"Thread deleted: {deleted}")

    def get_state_dict(self) -> dict[str, Any]:
         # Update internal dict before returning
         self._config_data['personality'] = self.personality.to_dict(); self._config_data['motivation'] = self.motivation.to_dict(); self._config_data['memory'] = self.memory.to_dict(); self._config_data['current_state'] = self.current_state.to_dict(); self._config_data.setdefault('profile', {})['assistant_id'] = self.assistant_id; return self._config_data

    def save_state_to_yaml(self, file_path: Optional[str] = None) -> bool:
         target_path = file_path or self._config_file_path
         if not target_path: logger.error(f"No path to save agent {self.name}."); return False
         state_dict = self.get_state_dict(); logger.info(f"Saving agent '{self.name}' state to {target_path}")
         return save_agent_state_to_yaml(state_dict, target_path)

    @classmethod
    def load_from_yaml(cls, file_path: str, agent_manager: AgentManager, world: Optional['World'] = None) -> Optional['Agent']:
         logger.info(f"Loading agent config: {file_path}")
         config = load_agent_config_from_yaml(file_path)
         if config:
             if not config.get('profile', {}).get('assistant_id'): logger.warning(f"Config {file_path} missing assistant_id.")
             try: return cls(config, agent_manager, world, config_file_path=file_path) # Pass path
             except Exception as e: logger.error(f"Failed init Agent from {file_path}: {e}", exc_info=True); return None
         else: logger.error(f"Failed load config dict from {file_path}"); return None

    async def delete_persistent_assistant(self) -> bool:
        if not self.agent_manager_ref: return False;
        if not self.assistant_id: logger.warning(f"{self.name} has no assistant_id."); return False
        logger.warning(f"Requesting PERMANENT deletion of assistant {self.assistant_id}...");
        deleted = await self.agent_manager_ref.delete_assistant(self.assistant_id)
        if deleted: logger.info(f"Assistant deleted."); self.assistant_id = None; return True
        else: logger.error(f"Failed delete assistant."); return False
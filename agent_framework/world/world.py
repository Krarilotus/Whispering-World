# agent_framework/world/world.py
import logging
import uuid
from typing import Optional, Dict, Any, Tuple, List # Use dict, list etc if preferred

from .world_state import WorldState, ENTITY_ID_PREFIX, Location
# Import the consolidated interface
from ..llm.llm_interface import LLM_API_Interface
from ..file_utils.yaml_handler import load_agent_config_from_yaml, save_agent_state_to_yaml
# Import prompts module for potential world queries
from ..llm import prompts

logger = logging.getLogger(__name__)

class World:
    """
    Manages world state (entities, locations). Uses consolidated LLM Interface
    for optional complex queries initiated by agents. Verification/generation logic is simplified.
    """
    def __init__(self,
                 world_state: WorldState,
                 llm_interface: LLM_API_Interface, # Use the single interface
                 world_oracle_assistant_id: Optional[str]):
        self.state = world_state
        self.llm = llm_interface # Store the shared interface
        self.world_oracle_assistant_id = world_oracle_assistant_id # Store the Oracle's ID
        self.world_oracle_thread_id: Optional[str] = None # Manage thread locally
        logger.info(f"World initialized. Oracle Asst ID: {self.world_oracle_assistant_id or 'Not Set'}")

    async def ensure_world_thread(self):
        """Ensures a dedicated thread exists for the World Oracle using the shared manager."""
        if self.world_oracle_thread_id:
            logger.debug(f"World Oracle reusing thread: {self.world_oracle_thread_id}")
            # Optional: Verify thread still exists? Requires manager access
            try: await self.llm.manager.client.beta.threads.retrieve(self.world_oracle_thread_id); return True
            except Exception: logger.warning("Existing world thread invalid, creating new."); self.world_oracle_thread_id = None
        if not self.world_oracle_assistant_id: logger.error("Cannot ensure world thread: Oracle Asst ID missing."); return False
        if not self.llm.manager: logger.error("Cannot ensure world thread: LLM Interface lacks AgentManager."); return False

        logger.info("Creating new thread for World Oracle...")
        self.world_oracle_thread_id = await self.llm.manager.create_thread(
            metadata={"purpose": "world_oracle_queries", "assistant_id": self.world_oracle_assistant_id})
        if not self.world_oracle_thread_id: logger.error("Could not create World Oracle thread."); return False
        logger.info(f"World Oracle assigned thread: {self.world_oracle_thread_id}")
        return True

    # --- Entity Resolution Helper ---
    def _resolve_entity(self, description: str) -> Optional[str]:
        """ Tries to find entity ID by ID or case-insensitive name match. """
        if not description: return None
        desc_lower = description.lower().strip()
        if description.startswith(ENTITY_ID_PREFIX) and description in self.state.entities: return description
        for entity_id, props in self.state.entities.items():
            name = props.get('name')
            if name and name.lower() == desc_lower:
                 logger.debug(f"Resolved '{description}' -> ID '{entity_id}' (Name Match).")
                 return entity_id
        logger.debug(f"Could not resolve description '{description}' to existing entity ID.")
        return None

    # --- Direct State Access/Query ---
    def get_entity_properties(self, entity_id_or_desc: str) -> Optional[dict[str, Any]]:
         """Finds an entity by ID or description and returns its properties."""
         entity_id = self._resolve_entity(entity_id_or_desc)
         if entity_id: return self.state.get_entity(entity_id)
         logger.debug(f"Entity '{entity_id_or_desc}' not found in world state.")
         return None

    # --- World Query via LLM (Called by Agent Action) ---
    async def query_world_via_llm(self, query_property: str, target_entity_desc: str) -> Optional[Any]:
         """Uses the World Oracle LLM for potentially complex queries about entities."""
         if not self.world_oracle_assistant_id: logger.error("No World Oracle Assistant ID set."); return None
         if not await self.ensure_world_thread(): logger.error("Failed ensure World Oracle thread."); return None

         target_entity_id = self._resolve_entity(target_entity_desc)
         if not target_entity_id: return f"Entity description '{target_entity_desc}' not resolved."

         world_context = self.state.get_summary(max_entities=30, max_len=1500) # Provide decent context

         # Use the generic JSON response method of the consolidated interface
         # We need to generate the specific prompt for this task first
         query_params = {
             "entity_id": target_entity_id,
             "property_name": query_property,
             "world_state_context": world_context
         }
         task_description = f"world_query:{target_entity_id}.{query_property}"
         try:
             prompt = prompts.get_query_entity_property_prompt(**query_params)
         except Exception as e:
              logger.error(f"Failed to generate World Oracle prompt for {task_description}: {e}")
              return None

         parsed_response = await self.llm.generate_json_response(
             assistant_id=self.world_oracle_assistant_id,
             thread_id=self.world_oracle_thread_id, # Use world's thread
             full_prompt=prompt,
             task_description=task_description
         )

         if parsed_response and "answer" in parsed_response:
            answer = parsed_response["answer"]
            if isinstance(answer, str) and "unknown based on provided context" in answer.lower(): return None
            return answer
         logger.warning(f"World LLM query failed valid JSON 'answer'. Task: {task_description}")
         return None # Indicate query failed or returned unknown

    # --- Basic State Updates (Called by Agent actions or setup) ---
    def update_entity_property(self, entity_id_or_desc: str, property_name: str, property_value: Any):
         """Directly updates a property for a resolved entity."""
         entity_id = self._resolve_entity(entity_id_or_desc)
         if entity_id: logger.info(f"Updating world state: {entity_id}.{property_name} = {property_value}"); self.state.add_or_update_entity(entity_id, {property_name: property_value})
         else: logger.warning(f"Cannot update property '{property_name}': Entity '{entity_id_or_desc}' not resolved.")

    def update_entity_location(self, entity_id: str, new_location_id_or_desc: Optional[str]):
         """Updates the entity's location property and location lists."""
         new_loc_id : Optional[str] = None
         if new_location_id_or_desc: new_loc_id = self._resolve_entity(new_location_id_or_desc)
         if new_location_id_or_desc and (not new_loc_id or new_loc_id not in self.state.locations): logger.warning(f"Invalid target location '{new_location_id_or_desc}' for entity '{entity_id}'."); return
         logger.info(f"World update: Entity '{entity_id}' location -> '{new_loc_id}' ('{new_location_id_or_desc}').")
         self.state.add_or_update_entity(entity_id, {'location_id': new_loc_id})

    # --- Persistence ---
    def save_state(self, file_path: str) -> bool: # ... same ...
         try: state_dict = self.state.to_dict(); logger.info(f"Saving world state to {file_path}"); return save_agent_state_to_yaml(state_dict, file_path)
         except Exception as e: logger.error(f"Error saving world state: {e}", exc_info=True); return False
    @classmethod
    def load_state(cls, file_path: str, llm_interface: LLM_API_Interface, world_oracle_assistant_id: Optional[str]) -> Optional['World']: # Updated signature
         logger.info(f"Attempting load world state: {file_path}")
         state_dict = load_agent_config_from_yaml(file_path)
         if state_dict:
             try: world_state = WorldState.from_dict(state_dict); logger.info("World state loaded."); return cls(world_state, llm_interface, world_oracle_assistant_id)
             except Exception as e: logger.error(f"Failed instantiate WorldState from {file_path}: {e}", exc_info=True); return None
         else: logger.warning(f"YAML load failed from {file_path}."); return None
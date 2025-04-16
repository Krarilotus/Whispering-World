# agent_framework/world/world.py
import logging
import uuid
from typing import Optional, Dict, Any, Tuple, List # Use dict, list etc if preferred

from .world_state import WorldState, ENTITY_ID_PREFIX
from .world_llm_interface import WorldLLMInterface
from ..file_utils.yaml_handler import load_agent_config_from_yaml, save_agent_state_to_yaml

logger = logging.getLogger(__name__)

class World:
    """Manages the overall simulation world state, entity resolution, and consistency checks."""
    def __init__(self, world_state: WorldState, world_llm_interface: WorldLLMInterface):
        self.state = world_state
        self.llm = world_llm_interface
        logger.info("World initialized.")

    def _resolve_entity(self, description: str) -> Optional[str]:
        """
        Tries to find an existing entity ID based on its name or description.
        Starts with simple name matching, could be expanded with LLM coreference.
        """
        if not description: return None
        desc_lower = description.lower()

        # Priority 1: Exact ID match
        if description.startswith(ENTITY_ID_PREFIX) and description in self.state.entities:
            return description

        # Priority 2: Exact name match (case-insensitive)
        # This assumes 'name' property exists consistently
        for entity_id, props in self.state.entities.items():
            name = props.get('name')
            if name and name.lower() == desc_lower:
                 logger.debug(f"Resolved '{description}' to entity ID '{entity_id}' by name match.")
                 return entity_id

        # Priority 3: Simple substring match in name (can be ambiguous)
        # Use with caution or disable if too broad
        # for entity_id, props in self.state.entities.items():
        #      name = props.get('name')
        #      if name and desc_lower in name.lower():
        #           logger.debug(f"Resolved '{description}' to entity ID '{entity_id}' by substring match (potential ambiguity).")
        #           return entity_id # Returns first match

        # TODO: Add LLM-based coreference resolution for "the king", "him", etc. if needed.

        logger.debug(f"Could not resolve entity description '{description}' to an existing entity ID.")
        return None

    async def add_entity(self, description: str, type_hint: Optional[str] = None) -> Optional[str]:
        """
        Creates a new entity in the world state.
        Generates default properties using the World LLM.
        Returns the new entity ID if successful, None otherwise.
        """
        logger.info(f"Request to add new entity described as '{description}' (Hint: {type_hint}).")
        # Generate a unique ID
        new_id = self.state.generate_new_entity_id(ENTITY_ID_PREFIX)

        # Ask World LLM to generate default properties
        world_context = self.state.get_summary() # Provide context
        default_properties = await self.llm.generate_entity_defaults(description, type_hint, world_context)

        if default_properties:
            # Ensure essentials are present
            default_properties.setdefault('name', description) # Use desc as name if not generated
            default_properties.setdefault('type', type_hint or 'Thing')
            # Add the entity to the state
            self.state.add_or_update_entity(new_id, default_properties)
            return new_id
        else:
            logger.error(f"Failed to generate default properties for new entity '{description}'. Cannot add.")
            return None

    async def query_entity_property(self, entity_id: str, property_name: str) -> Optional[Any]:
        """Queries a specific property of a known entity."""
        entity = self.state.get_entity(entity_id)
        if not entity:
            logger.warning(f"Query for non-existent entity ID: {entity_id}")
            return None

        # Check if property exists locally first
        if property_name in entity:
            return entity[property_name]
        else:
             # If not found locally, maybe ask the World LLM based on general context?
             # Could be useful for inferred properties, but risks inconsistency.
             # For now, only return locally known properties.
             logger.debug(f"Property '{property_name}' not found directly on entity '{entity_id}'.")
             # Optional: Ask World LLM based on context (might be slow/costly)
             # world_context = self.state.get_summary()
             # llm_answer = await self.llm.query_entity_property(entity_id, property_name, world_context)
             # return llm_answer
             return None


    async def propose_assertion(self, assertion: dict, proposing_agent_id: Optional[str] = None) -> bool:
        """
        Processes a proposed atomic assertion (existence, property, relationship).
        Checks consistency and applies the change to the world state if valid.
        Requires assertion dictionary to have descriptions replaced with resolved entity IDs
        (e.g., 'entity_id' instead of 'entity_desc').
        Returns True if the assertion was successfully applied, False otherwise.
        """
        logger.info(f"Agent '{proposing_agent_id}' proposes assertion: {assertion}")
        assertion_type = assertion.get("type")

        # 1. Basic Validation
        if not assertion_type:
             logger.error("Proposed assertion is missing 'type'. Rejected.")
             return False

        # 2. Check Consistency with World State via LLM
        world_context = self.state.get_summary()
        consistency_status = await self.llm.check_assertion_consistency(assertion, world_context)

        if consistency_status == "contradictory":
            logger.warning(f"Proposed assertion {assertion} contradicts existing world state. Rejected.")
            return False
        elif consistency_status == "unknown":
            # If LLM is unsure based on context, should we allow it?
            # Let's be conservative: only apply if explicitly 'consistent'.
            # Alternatively, could have different logic: allow 'unknown' for some types?
            logger.warning(f"Proposed assertion {assertion} consistency is unknown based on context. Rejected.")
            return False
        elif consistency_status != "consistent":
             logger.error(f"Unexpected consistency status '{consistency_status}' for assertion {assertion}. Rejected.")
             return False

        # 3. Apply Consistent Assertion to World State
        logger.debug(f"Assertion {assertion} deemed consistent. Applying to world state.")
        try:
            if assertion_type == "existence":
                # Existence check should ideally happen *before* proposing.
                # If we reach here and it's consistent, it implies the entity *should* exist.
                # Ensure it does, possibly by adding minimal data if needed (though creation should handle this)
                entity_id = assertion.get('entity_id')
                if entity_id and entity_id not in self.state.entities:
                     logger.warning(f"Consistent 'existence' assertion proposed for '{entity_id}' which wasn't previously created. Adding minimal entry.")
                     self.state.add_or_update_entity(entity_id, {"id": entity_id, "name": entity_id, "type": assertion.get("entity_type_hint","Thing")})
                # No state change if entity already exists.
            elif assertion_type == "property":
                entity_id = assertion.get('entity_id')
                prop_name = assertion.get('property')
                prop_value = assertion.get('value') # Note: 'value_desc' should be resolved to ID if it's an entity link
                if entity_id and prop_name is not None and prop_value is not None:
                    self.state.add_or_update_entity(entity_id, {prop_name: prop_value}) # Use update method
                else: raise ValueError("Missing required fields for property assertion.")
            elif assertion_type == "relationship":
                subj_id = assertion.get('subject_id')
                verb = assertion.get('verb')
                obj_id = assertion.get('object_id')
                if subj_id and verb and obj_id:
                     # Simple relationship storage for now
                     self.state.add_relationship(subj_id, verb, obj_id)
                     # Could also update properties, e.g. subject.knows = [..., obj_id]
                else: raise ValueError("Missing required fields for relationship assertion.")
            else:
                raise ValueError(f"Unknown assertion type: {assertion_type}")

            logger.info(f"Successfully applied assertion: {assertion}")
            return True

        except Exception as e:
             logger.error(f"Error applying consistent assertion {assertion}: {e}", exc_info=True)
             return False

    # --- World State Accessors/Mutators (Simplified examples) ---
    def get_agent_location_id(self, agent_entity_id: str) -> Optional[str]:
        """Gets the registered location ID of an agent entity."""
        agent_props = self.state.get_entity(agent_entity_id)
        return agent_props.get('location_id') if agent_props else None

    def update_entity_location(self, entity_id: str, new_location_id: Optional[str]):
         """Updates the entity's location property and location lists."""
         logger.info(f"World state update: Entity '{entity_id}' location changed to '{new_location_id}'.")
         # This uses the logic already built into add_or_update_entity
         self.state.add_or_update_entity(entity_id, {'location_id': new_location_id})


    # --- Persistence ---
    def save_state(self, file_path: str) -> bool:
        """Saves the current world state to YAML."""
        state_dict = self.state.to_dict()
        logger.info(f"Saving world state to {file_path}")
        return save_agent_state_to_yaml(state_dict, file_path) # Reusing agent saver

    @classmethod
    def load_state(cls, file_path: str, world_llm_interface: WorldLLMInterface) -> Optional['World']:
        """Loads world state from YAML and creates a World instance."""
        logger.info(f"Attempting to load world state from: {file_path}")
        state_dict = load_agent_config_from_yaml(file_path) # Reusing agent loader
        if state_dict:
            world_state = WorldState.from_dict(state_dict)
            return cls(world_state, world_llm_interface)
        else:
             # Error logged by load_agent_config_from_yaml
             return None
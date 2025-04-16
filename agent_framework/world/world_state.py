# agent_framework/world/world_state.py
import time
import logging
import uuid
from typing import Dict, Any, Optional, List, Set # Use modern types list, dict etc if preferred

logger = logging.getLogger(__name__)

# --- Constants ---
ENTITY_ID_PREFIX = "ent_"

# --- World Data Structures ---

class Location:
    """Represents a location in the world."""
    def __init__(self, loc_id: str, name: str, description: str):
        self.id: str = loc_id
        self.name: str = name
        self.description: str = description
        self.entities_present: Set[str] = set() # Set of Entity IDs in this location

    def add_entity(self, entity_id: str):
        self.entities_present.add(entity_id)

    def remove_entity(self, entity_id: str):
        self.entities_present.discard(entity_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entities_present": sorted(list(self.entities_present)) # Save sorted list
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Location':
        loc = cls(data["id"], data.get("name","Unknown Location"), data.get("description",""))
        loc.entities_present = set(data.get("entities_present", []))
        return loc

class WorldState:
    """Holds the structured, objective ground truth of the simulation world using entities."""
    def __init__(self):
        self.current_time: float = time.time() # Or a simulated time step
        # Core state: Entities described by properties
        self.entities: dict[str, dict[str, Any]] = {} # Key: Entity ID, Value: Property Dict
        # Locations are also entities, but we keep a separate index for convenience
        self.locations: dict[str, Location] = {} # Key: Location Entity ID, Value: Location Object
        # Track simple relationships (can be expanded)
        self.relationships: list[tuple[str, str, str]] = [] # (subject_id, verb, object_id)

    def get_entity(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Safely retrieves an entity's property dictionary."""
        return self.entities.get(entity_id)

    def add_or_update_entity(self, entity_id: str, properties: dict[str, Any]):
        """Adds a new entity or updates properties of an existing one."""
        if entity_id not in self.entities:
            self.entities[entity_id] = {'id': entity_id} # Ensure ID exists
            logger.info(f"Entity '{entity_id}' created.")
        # Sensible merge: new properties overwrite/add to existing ones
        self.entities[entity_id].update(properties)
        logger.debug(f"Entity '{entity_id}' updated with properties: {properties}")

        # Handle special properties like 'type' and 'location_id'
        entity_type = self.entities[entity_id].get('type')
        if entity_type == 'Location' and entity_id not in self.locations:
            # Create Location index entry
            loc_name = self.entities[entity_id].get('name', entity_id)
            loc_desc = self.entities[entity_id].get('description', '')
            self.locations[entity_id] = Location(entity_id, loc_name, loc_desc)
            logger.info(f"Location index entry created for '{entity_id}'.")

        # If location changed, update Location objects
        if 'location_id' in properties:
            new_loc_id = properties['location_id']
            # Remove from old location(s) if applicable
            for loc in self.locations.values():
                if entity_id in loc.entities_present and loc.id != new_loc_id:
                    loc.remove_entity(entity_id)
                    logger.debug(f"Removed entity '{entity_id}' from location '{loc.id}'.")
            # Add to new location if it exists
            if new_loc_id in self.locations:
                self.locations[new_loc_id].add_entity(entity_id)
                logger.debug(f"Added entity '{entity_id}' to location '{new_loc_id}'.")
            elif new_loc_id is not None: # If location ID set but location doesn't exist
                 logger.warning(f"Attempted to move entity '{entity_id}' to non-existent location '{new_loc_id}'.")


    def add_relationship(self, subj_id: str, verb: str, obj_id: str):
        """Adds a directed relationship between two entities."""
        rel = (subj_id, verb, obj_id)
        if rel not in self.relationships:
            self.relationships.append(rel)
            logger.info(f"Added relationship: {subj_id} --{verb}--> {obj_id}")

    def find_entity_by_property(self, **kwargs) -> list[str]:
        """Finds entity IDs where all specified properties match."""
        found_ids = []
        for entity_id, properties in self.entities.items():
            match = True
            for key, value in kwargs.items():
                if properties.get(key) != value:
                    match = False
                    break
            if match:
                found_ids.append(entity_id)
        return found_ids

    def get_summary(self, max_entities=30, max_rels=30, max_len=2500) -> str:
        """Generates a textual summary of the current world state for LLM context."""
        summary = f"--- World State Summary (Time: {self.current_time:.0f}) ---\n"

        summary += "Key Entities & Properties:\n"
        entity_count = 0
        sorted_entity_ids = sorted(self.entities.keys()) # Sort for consistency
        for entity_id in sorted_entity_ids:
            if entity_count >= max_entities:
                summary += f"  (...and {len(self.entities) - max_entities} more entities)\n"
                break
            props = self.entities[entity_id]
            prop_strs = []
            # Prioritize key properties for summary
            for key in ['id', 'name', 'type', 'location_id', 'state', 'mood', 'title']:
                 if key in props:
                     prop_strs.append(f"{key}: {props[key]}")
            # Add a few other properties if space allows
            other_keys = [k for k in props if k not in ['id', 'name', 'type', 'location_id', 'state', 'mood', 'title']]
            for key in other_keys[:2]: # Limit other props shown
                 prop_strs.append(f"{key}: {props[key]}")

            line = f"  - Entity: {entity_id} ({', '.join(prop_strs)})\n"
            if len(summary) + len(line) > max_len:
                summary += "  (...summary truncated due to length)\n"
                break
            summary += line
            entity_count += 1
        if not self.entities: summary += "  - None\n"

        summary += "\nRelationships (Subject -> Verb -> Object):\n"
        rel_count = 0
        # Consider sorting relationships if needed for consistency
        for rel in self.relationships:
             if rel_count >= max_rels:
                  summary += f"  (...and {len(self.relationships) - max_rels} more relationships)\n"
                  break
             line = f"  - {rel[0]} --{rel[1]}--> {rel[2]}\n"
             if len(summary) + len(line) > max_len:
                summary += "  (...summary truncated due to length)\n"
                break
             summary += line
             rel_count += 1
        if not self.relationships: summary += "  - None\n"

        summary += "--- End World State Summary ---\n"
        return summary


    def to_dict(self) -> dict[str, Any]:
        return {
            "current_time": self.current_time,
            "entities": self.entities,
            # Save location objects separately for easier reconstruction? Or just use entities?
            # Let's save the index for convenience, but ensure locations are also in entities.
            "locations_index": {id: loc.to_dict() for id, loc in self.locations.items()},
            "relationships": self.relationships,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorldState':
        state = cls()
        state.current_time = data.get("current_time", time.time())
        state.entities = data.get("entities", {})
        state.relationships = [tuple(rel) for rel in data.get("relationships", []) if isinstance(rel, list) and len(rel) == 3] # Ensure tuples

        # Rebuild location index from entities or load saved index
        locations_index_data = data.get("locations_index", {})
        if locations_index_data:
             state.locations = {id: Location.from_dict(loc_data) for id, loc_data in locations_index_data.items()}
        else: # Rebuild from entities if index wasn't saved
             for entity_id, props in state.entities.items():
                 if props.get('type') == 'Location':
                      loc_name = props.get('name', entity_id)
                      loc_desc = props.get('description', '')
                      state.locations[entity_id] = Location(entity_id, loc_name, loc_desc)
             # Re-populate entities_present in locations (optional, might be slow)
             for entity_id, props in state.entities.items():
                  loc_id = props.get('location_id')
                  if loc_id and loc_id in state.locations:
                       state.locations[loc_id].add_entity(entity_id)

        return state

    def generate_new_entity_id(self, prefix: str = ENTITY_ID_PREFIX) -> str:
        """Generates a unique entity ID."""
        # Simple approach: prefix + uuid
        return f"{prefix}{uuid.uuid4().hex[:8]}"
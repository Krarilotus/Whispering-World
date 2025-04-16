# agent_framework/world/world_llm_interface.py
import json
import logging
from typing import Optional, Dict, Any, Union # Use dict, list etc if preferred

# Import AgentManager from adjusted path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path: sys.path.append(project_root)
from ..api.agent_api import AgentManager, DEFAULT_MODEL

# Import prompt functions
from ..llm import prompts
from ..file_utils.yaml_handler import load_agent_config_from_yaml

logger = logging.getLogger(__name__)

class WorldLLMInterface:
    """Handles LLM interactions for the World module using function-based prompts."""
    def __init__(self, agent_manager: AgentManager, config_path: Optional[str] = None, assistant_id: Optional[str] = None):
        """
        Initializes the World LLM Interface.

        Args:
            agent_manager: The AgentManager instance.
            config_path: Path to the World Oracle's YAML config file (used if assistant_id is not provided or invalid).
            assistant_id: An optional pre-existing Assistant ID for the World Oracle.
        """
        self.manager = agent_manager
        self.assistant_id = assistant_id
        self.config_path = config_path
        self.thread_id: Optional[str] = None
        self.config: Optional[dict[str, Any]] = None
        if not self.assistant_id and not self.config_path:
            raise ValueError("WorldLLMInterface requires either an assistant_id or a config_path.")
        logger.info("WorldLLMInterface initialized.")

    async def ensure_assistant_and_thread(self, default_model=DEFAULT_MODEL, default_temp=0.2):
        """Ensures the World Oracle assistant and its dedicated thread exist, creating them if necessary from config."""
        logger.debug("Ensuring World Oracle assistant and thread...")
        # --- Check/Verify/Create Assistant ---
        assistant_verified = False
        if self.assistant_id:
            details = await self.manager.load_agent(self.assistant_id)
            if details:
                logger.info(f"Verified existing World Oracle assistant: {self.assistant_id}")
                assistant_verified = True
            else:
                logger.warning(f"World Oracle assistant ID {self.assistant_id} not found or invalid. Will attempt creation from config.")
                self.assistant_id = None # Clear invalid ID

        if not assistant_verified:
            if not self.config_path or not os.path.exists(self.config_path):
                raise ValueError(f"World Oracle config file not found at {self.config_path} and no valid assistant ID provided.")

            logger.info(f"Loading World Oracle config from: {self.config_path}")
            self.config = load_agent_config_from_yaml(self.config_path)
            if not self.config:
                raise ValueError(f"Failed to load World Oracle config from {self.config_path}")

            config_profile = self.config.get('profile', {})
            config_asst_id = config_profile.get('assistant_id') # Check if config provides an ID to reuse

            if config_asst_id:
                logger.info(f"Config specifies assistant ID: {config_asst_id}. Verifying...")
                details = await self.manager.load_agent(config_asst_id)
                if details:
                    self.assistant_id = config_asst_id
                    logger.info(f"Using existing World Oracle assistant from config: {self.assistant_id}")
                    assistant_verified = True
                else:
                    logger.warning(f"Assistant ID {config_asst_id} from config not found/invalid. Creating new.")

            if not assistant_verified: # Create new if needed
                name = config_profile.get('name', 'World Oracle')
                instructions = config_profile.get('description')
                if not instructions: raise ValueError("World Oracle config must contain 'profile.description' (instructions).")

                model_cfg = self.config.get('model_defaults', {})
                model = model_cfg.get('model', default_model)
                temperature = model_cfg.get('temperature', default_temp)

                logger.info(f"Creating new World Oracle assistant '{name}' with model '{model}'...")
                new_id = await self.manager.create_agent(
                    name=name,
                    instructions=instructions,
                    model=model,
                    temperature=temperature,
                    metadata={"purpose": "world_oracle"}
                )
                if new_id:
                    self.assistant_id = new_id
                    assistant_verified = True
                    logger.info(f"Created new World Oracle assistant: {self.assistant_id}")
                    # TODO: Optional: Save the new ID back to the config file?
                else:
                    raise RuntimeError("Failed to create World Oracle assistant via API.")

        # --- Ensure Thread Exists ---
        if not self.thread_id:
            logger.info(f"Creating new thread for World Oracle (Assistant: {self.assistant_id})...")
            # Ensure assistant ID is valid before creating thread
            if not self.assistant_id: raise RuntimeError("Cannot create thread without a valid World Oracle Assistant ID.")
            self.thread_id = await self.manager.create_thread(metadata={"purpose": "world_oracle_queries", "assistant_id": self.assistant_id})
            if not self.thread_id:
                raise RuntimeError("Could not create World Oracle thread via API.")
            logger.info(f"World Oracle using thread: {self.thread_id}")
        else:
             logger.debug(f"World Oracle reusing existing thread: {self.thread_id}")


    def _parse_json_response(self, response_text: Optional[str], task_description: str) -> Optional[dict[str, Any]]:
        """Attempts to parse JSON from the LLM response, with basic cleaning."""
        if not response_text:
            logger.error(f"Received empty response for World task: {task_description}")
            return None
        try:
            # Basic cleaning for potential markdown ```json ... ``` blocks or leading/trailing text
            response_text = response_text.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                 response_text = response_text[json_start:json_end+1]
            else:
                 logger.warning(f"Could not reliably find JSON object markers {{}} in response for {task_description}. Attempting parse anyway. Raw: {response_text[:200]}...")
                 # Attempt parsing anyway, might work if it's just missing backticks

            parsed_json = json.loads(response_text.strip())
            if not isinstance(parsed_json, dict):
                 logger.error(f"Parsed JSON is not a dictionary for World task '{task_description}'. Type: {type(parsed_json)}. Response: {response_text[:200]}")
                 return None
            return parsed_json
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for World task '{task_description}'. Response: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON for World task '{task_description}': {e}. Response: {response_text[:500]}")
            return None

    async def _run_world_query_non_streaming(self, prompt: str, timeout: int = 60) -> Optional[str]:
         """Runs a non-streaming call specifically for world queries using its dedicated thread."""
         # Ensure assistant and thread are ready
         try:
              await self.ensure_assistant_and_thread()
         except Exception as e:
              logger.critical(f"Failed to ensure World Oracle assistant/thread before running query: {e}", exc_info=True)
              return None

         if not self.assistant_id or not self.thread_id:
             logger.error("World assistant/thread still not available after check.")
             return None

         # Add message to the dedicated world thread
         msg_id = await self.manager.add_message_to_thread(self.thread_id, prompt, role="user")
         if not msg_id:
              logger.error(f"Failed to add world query prompt to thread {self.thread_id}.")
              return None

         logger.debug(f"Requesting World Oracle run on thread {self.thread_id}...")
         run_result = await self.manager.run_agent_on_thread_non_streaming(
             assistant_id=self.assistant_id, thread_id=self.thread_id, timeout_seconds=timeout)

         # Process result
         if run_result and run_result.get("status") == "completed":
             run_id = run_result.get("id")
             logger.debug(f"World Oracle run {run_id} completed. Retrieving message...")
             messages = await self.manager.get_thread_messages(self.thread_id, limit=1, order="desc")
             if messages and messages[0].get("role") == "assistant" and messages[0].get("run_id") == run_id:
                 content_list = messages[0].get("content", [])
                 response = "".join(c.get("text", {}).get("value", "") for c in content_list if c.get("type") == "text").strip()
                 logger.debug(f"World Oracle Raw Response ({run_id}): {response[:150]}...")
                 return response
             else:
                 logger.error(f"World Oracle run {run_id} completed, but failed retrieve message from thread {self.thread_id}.")
                 return None
         elif run_result:
             logger.error(f"World Oracle run failed or incomplete. Status: {run_result.get('status')}, Error: {run_result.get('last_error')}")
             return None
         else:
             logger.error(f"World Oracle run failed unexpectedly (no run result).")
             return None

    async def query_entity_property(self, entity_id: str, property_name: str, world_state_context: str) -> Optional[Any]:
        """Asks the World Oracle about a specific property of an entity."""
        # Call prompt function
        prompt = prompts.get_query_entity_property_prompt(
            entity_id=entity_id,
            property_name=property_name,
            world_state_context=world_state_context
        )
        logger.debug(f"--- World Query: Entity '{entity_id}', Property '{property_name}' ---")
        raw_response = await self._run_world_query_non_streaming(prompt, timeout=45)
        parsed = self._parse_json_response(raw_response, f"query_entity_property({entity_id}.{property_name})")

        if parsed and "answer" in parsed:
            answer = parsed["answer"]
            # Treat specific string "Unknown..." as None, otherwise return the value
            if isinstance(answer, str) and "unknown based on provided context" in answer.lower():
                 logger.info(f"World Oracle query for {entity_id}.{property_name}: Unknown")
                 return None
            logger.info(f"World Oracle query for {entity_id}.{property_name}: Answer='{str(answer)[:100]}'")
            return answer # Return the actual value (could be string, number, bool, etc.)
        else:
             logger.warning(f"World Oracle query failed to produce valid JSON with 'answer'. Raw: {raw_response}")
             return None

    async def check_assertion_consistency(self, assertion: dict, world_state_context: str) -> str:
        """Asks World Oracle if a new assertion contradicts the world state. Returns 'consistent', 'contradictory', or 'unknown'."""
        # Call prompt function
        prompt = prompts.get_check_assertion_consistency_prompt(
            assertion=assertion, # Pass the dict, function handles serialization
            world_state_context=world_state_context
        )
        logger.debug(f"--- World Assertion Consistency Check --- \nAssertion: {assertion}\n...")
        raw_response = await self._run_world_query_non_streaming(prompt, timeout=60)
        parsed = self._parse_json_response(raw_response, f"check_assertion_consistency")

        if parsed and isinstance(parsed.get("consistency_status"), str):
            status = parsed["consistency_status"]
            valid_statuses = ["consistent", "contradictory", "unknown"]
            if status in valid_statuses:
                logger.info(f"World Oracle consistency check for {assertion}: {status}")
                return status
            else:
                logger.warning(f"World Oracle consistency check returned invalid status '{status}'. Raw: {raw_response}")
                return "unknown" # Default to unknown on invalid status
        else:
             logger.warning(f"World Oracle consistency check failed valid JSON response with 'consistency_status'. Raw: {raw_response}")
             return "unknown" # Default to unknown on failure

    async def generate_entity_defaults(self, entity_description: str, entity_type_hint: Optional[str], world_state_context: str) -> Optional[dict[str, Any]]:
        """Asks World Oracle to generate plausible default properties for a new entity."""
        # Call prompt function
        prompt = prompts.get_generate_entity_properties_prompt(
            entity_description=entity_description,
            entity_type_hint=entity_type_hint,
            world_state_context=world_state_context
        )
        logger.debug(f"--- World Generate Entity Defaults ---\nDesc: {entity_description}, TypeHint: {entity_type_hint}\n...")
        raw_response = await self._run_world_query_non_streaming(prompt, timeout=90)
        parsed = self._parse_json_response(raw_response, f"generate_entity_defaults({entity_description})")

        if parsed and isinstance(parsed.get("properties"), dict):
            properties = parsed["properties"]
            # Basic validation/defaults - ensure name and type exist
            properties.setdefault('name', entity_description)
            properties.setdefault('type', entity_type_hint or 'Thing')
            logger.info(f"World Oracle generated default properties for '{entity_description}': {properties}")
            return properties
        else:
            logger.error(f"World Oracle failed to generate valid JSON with 'properties' dict for '{entity_description}'. Raw: {raw_response}")
            return None
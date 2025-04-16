# agent_framework/llm/llm_interface.py
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING # Use modern types if preferred

# Import AgentManager from adjusted path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path: sys.path.append(project_root)

from ..api.agent_api import AgentManager, DEFAULT_MODEL, DEFAULT_DM_MODEL
# Import prompt functions from prompts module
from . import prompts

if TYPE_CHECKING: from ..core.agent import Agent

logger = logging.getLogger(__name__)

class LLMInterface:
    """Handles interaction with the Agent's LLM via AgentManager using function-based prompts."""
    def __init__(self, agent_manager: AgentManager, agent_ref: 'Agent'):
        self.manager = agent_manager
        self.agent = agent_ref

    # --- Helper Methods ---
    def _get_agent_api_details(self) -> Optional[tuple[str, str]]:
         """Safely gets assistant_id and thread_id from the agent."""
         if not self.agent.assistant_id or not self.agent.thread_id:
             logger.error(f"Agent {self.agent.name} assistant_id or thread_id is not set in LLMInterface.")
             return None
         return self.agent.assistant_id, self.agent.thread_id

    def _parse_json_response(self, response_text: Optional[str], task_description: str) -> Optional[dict[str, Any]]:
        """Attempts to parse JSON from the LLM response."""
        if not response_text:
            logger.error(f"Received empty response for Agent task: {task_description}")
            return None
        try:
            # Basic cleaning for potential markdown ```json ... ``` blocks or leading/trailing text
            response_text = response_text.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                 response_text = response_text[json_start:json_end+1]
            else:
                 logger.warning(f"Could not reliably find JSON object markers {{}} in response for {task_description}. Raw: {response_text[:200]}...")
                 # Attempt parsing anyway, might work if it's just missing backticks
                 # return None # Stricter approach

            parsed_json = json.loads(response_text.strip())
            if not isinstance(parsed_json, dict):
                 logger.error(f"Parsed JSON is not a dictionary for Agent task '{task_description}'. Type: {type(parsed_json)}. Response: {response_text[:200]}")
                 return None
            return parsed_json
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for Agent task '{task_description}'. Response: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON for Agent task '{task_description}': {e}. Response: {response_text[:500]}")
            return None

    async def _run_llm_non_streaming(self, prompt: str, timeout: int = 120) -> Optional[str]:
         """Runs a non-streaming call to the Agent's LLM."""
         api_details = self._get_agent_api_details()
         if not api_details: return None
         assistant_id, thread_id = api_details

         msg_id = await self.manager.add_message_to_thread(thread_id, prompt, role="user")
         if not msg_id:
             logger.error(f"Agent {self.agent.name}: Failed to add prompt message to thread {thread_id}.")
             return None

         logger.debug(f"Agent {self.agent.name}: Requesting non-streaming run on thread {thread_id}...")
         run_result = await self.manager.run_agent_on_thread_non_streaming(
             assistant_id=assistant_id, thread_id=thread_id, timeout_seconds=timeout)

         if not run_result:
             logger.error(f"Agent {self.agent.name}: Run failed or returned no result on thread {thread_id}.")
             return None
         run_id = run_result.get("id")
         status = run_result.get("status")

         if status == "completed":
             logger.debug(f"Agent {self.agent.name}: Run {run_id} completed. Retrieving message...")
             messages = await self.manager.get_thread_messages(thread_id, limit=1, order="desc")
             if messages and messages[0].get("role") == "assistant" and messages[0].get("run_id") == run_id:
                 content_list = messages[0].get("content", [])
                 resp = "".join(c.get("text", {}).get("value", "") for c in content_list if c.get("type") == "text").strip()
                 logger.debug(f"Agent LLM Raw Response ({run_id}): {resp[:200]}...")
                 return resp
             else:
                 logger.error(f"Agent {self.agent.name}: Run {run_id} completed, but failed retrieve message from thread {thread_id}.")
                 return None
         elif status == "requires_action":
             logger.warning(f"Agent {self.agent.name}: Run {run_id} requires action (tool use), which is not handled in this interface.")
             # TODO: Implement tool handling if needed
             return None
         elif status == "timed_out":
             logger.error(f"Agent {self.agent.name}: Run {run_id} timed out on thread {thread_id}.")
             return None
         else:
             logger.error(f"Agent {self.agent.name}: Run {run_id} ended with unhandled status: {status}. Error: {run_result.get('last_error')}")
             return None

    # --- Core Agent Interaction Methods ---

    def _construct_base_prompt(self) -> str:
        """Constructs the common context part using the prompt function."""
        # Call the prompt function from prompts.py
        prompt = prompts.get_base_context_prompt(
            agent_name=self.agent.name,
            agent_description=self.agent.description,
            personality_description=self.agent.personality.get_description(),
            motivation_description=self.agent.motivation.get_state_description(),
            current_state_description=self.agent.current_state.get_state_description(),
            memory_summary=self.agent.memory.get_memory_summary(max_memories=15, max_length=1000) # Limit summary length
        )
        return prompt

    async def generate_agent_response(self, user_input: str, verification_context: str) -> Optional[dict[str, Any]]:
        """Generates the agent's main response to user input using JSON format."""
        base_prompt = self._construct_base_prompt()
        # Call the prompt function
        task_prompt = prompts.get_react_to_input_prompt(
             user_input=user_input,
             verification_context=verification_context # Pass the detailed verification results
        )
        # Combine base context with task-specific instructions
        full_prompt = f"{base_prompt}\n{task_prompt}"

        logger.debug(f"--- Generating Agent Response --- \nPrompt Start (truncated):\n{full_prompt[:1000]}...\n--- End Prompt Start ---")

        raw_response = await self._run_llm_non_streaming(full_prompt, timeout=150) # Longer timeout for complex response
        return self._parse_json_response(raw_response, "generate_agent_response")

    # --- Verification Support Methods ---

    async def extract_atomic_assertions(self, user_input: str) -> Optional[list[dict[str, Any]]]:
        """Uses LLM to extract atomic assertions from user input."""
        # Call the prompt function
        prompt = prompts.get_extract_atomic_assertions_prompt(user_input)

        logger.debug(f"--- Extracting Atomic Assertions --- \nInput: {user_input}")
        raw_response = await self._run_llm_non_streaming(prompt, timeout=90) # Increase timeout
        parsed_json = self._parse_json_response(raw_response, "extract_atomic_assertions")

        if parsed_json and isinstance(parsed_json.get("assertions"), list):
            # Basic validation: Ensure items in the list are dictionaries
            validated_assertions = [a for a in parsed_json["assertions"] if isinstance(a, dict)]
            if len(validated_assertions) != len(parsed_json["assertions"]):
                 logger.warning(f"Some items in extracted 'assertions' were not dictionaries. Raw: {raw_response}")

            if not validated_assertions and user_input: # If LLM returns empty list for non-empty input
                 logger.info(f"LLM extracted no assertions for input: '{user_input[:100]}...'")
            elif validated_assertions:
                 logger.info(f"Extracted {len(validated_assertions)} atomic assertions.")
                 logger.debug(f"Extracted Assertions: {json.dumps(validated_assertions, indent=2)}")
            return validated_assertions # Return list (can be empty)
        else:
            # Log error if parsing failed or 'assertions' key is missing/not a list
            logger.error(f"Assertion extraction failed to produce valid JSON list 'assertions'. Raw: {raw_response}")
            return None # Indicate failure to extract

    async def check_claim_plausibility(self, claim_or_assertion: Union[str, dict], assessment_target: str = "agent") -> Optional[Tuple[bool, str]]:
        """Uses LLM to check if a claim/assertion is plausible for the agent or the world."""
        # Format claim/assertion for context search and prompt display
        if isinstance(claim_or_assertion, dict):
            assertion_summary = f"Assertion({json.dumps(claim_or_assertion)})"
            claim_for_context_search = claim_or_assertion.get("entity_desc") or claim_or_assertion.get("subject_desc") or assertion_summary
        else:
            assertion_summary = claim_for_context_search = claim_or_assertion

        # Retrieve relevant memories
        relevant_memories = self.agent.memory.retrieve_relevant_memories(query_text=claim_for_context_search, top_k=5)
        mem_summary = "\n".join([f"- {mem}" for mem in relevant_memories]) if relevant_memories else "No specifically relevant memories found."

        # Get world context summary
        world_summary = self.agent.world.state.get_summary(max_entities=10, max_rels=5, max_len=600) if self.agent.world else "N/A"
        relevant_context = f"Relevant Agent Memories:\n{mem_summary}\n\nRelevant World State Snapshot:\n{world_summary}"

        # Determine prompt context details
        if assessment_target == "agent":
            plausibility_context = "for you personally (your history, beliefs, internal state)"
            reasoning_focus = "based on your persona and memories"
            personality_desc = self.agent.personality.get_description()
        elif assessment_target == "world":
             plausibility_context = "within the objective reality of the simulated world"
             reasoning_focus = "based on general world knowledge and context provided"
             personality_desc = "N/A (Assessing objective world)"
        else:
             logger.error(f"Invalid assessment_target '{assessment_target}' for plausibility check.")
             return None

        # Call the prompt function
        prompt = prompts.get_check_plausibility_prompt(
            agent_name=self.agent.name, agent_description=self.agent.description,
            assessment_target=assessment_target, plausibility_context=plausibility_context,
            reasoning_focus=reasoning_focus, personality_description=personality_desc,
            relevant_context=relevant_context, claim_or_assertion=assertion_summary
        )

        logger.debug(f"--- Checking Plausibility ({assessment_target}) for '{assertion_summary[:100]}' ---")

        # Make the LLM call
        raw_response = await self._run_llm_non_streaming(prompt, timeout=75)
        parsed_json = self._parse_json_response(raw_response, f"check_claim_plausibility ({assessment_target})")

        # Process the response
        if parsed_json and isinstance(parsed_json.get("is_plausible"), bool) and isinstance(parsed_json.get("reasoning"), str):
            logger.info(f"Plausibility Check Result: Plausible={parsed_json['is_plausible']}, Reason='{parsed_json['reasoning'][:100]}...'")
            return parsed_json["is_plausible"], parsed_json["reasoning"]
        else:
            logger.error(f"Plausibility check ({assessment_target}) failed valid JSON. Raw: {raw_response}")
            return False, f"LLM response for plausibility check ({assessment_target}) was invalid or malformed."

    async def generate_synthetic_memory(self, assertion: dict) -> Optional[str]:
         """Uses LLM to generate a synthetic AGENT memory for a plausible internal assertion."""
         assertion_summary = f"Assertion({json.dumps(assertion)})"
         # Call the prompt function
         prompt = prompts.get_generate_synthetic_memory_prompt(
              agent_name=self.agent.name, agent_description=self.agent.description,
              personality_description=self.agent.personality.get_description(),
              assertion_summary=assertion_summary
         )
         logger.debug(f"--- Generating Synthetic Agent Memory for '{assertion_summary}' ---")
         raw_response = await self._run_llm_non_streaming(prompt, timeout=90)
         parsed_json = self._parse_json_response(raw_response, "generate_synthetic_agent_memory")
         if parsed_json and isinstance(parsed_json.get("synthetic_memory"), str):
             generated_mem = parsed_json["synthetic_memory"]
             logger.info(f"Generated synthetic memory: {generated_mem[:100]}...")
             return generated_mem
         else:
             logger.error(f"Synthetic memory gen failed valid JSON 'synthetic_memory'. Raw: {raw_response}")
             return None

    # Optional: Keep generate_simple_response if needed for other tasks like reflection summaries
    async def generate_simple_response(self, prompt: str) -> Optional[str]:
         """Generates a simple text response without enforcing JSON structure."""
         logger.debug(f"--- Generating Simple Response --- \nPrompt Start:\n{prompt[:500]}...")
         return await self._run_llm_non_streaming(prompt, timeout=90)
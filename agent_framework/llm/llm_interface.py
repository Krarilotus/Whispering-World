# agent_framework/llm/llm_interface.py
import json
import logging
import asyncio
from typing import Optional, Dict, Any

# Import AgentManager and prompts correctly using relative paths
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path: sys.path.append(project_root)
from agent_framework.api.agent_api import AgentManager # Correct relative import
from agent_framework.llm import prompts # Correct relative import
import openai # For RateLimitError

logger = logging.getLogger(__name__)

class LLM_API_Interface:
    """ Consolidated interface for OpenAI Assistant API calls with retries and parsing. """
    def __init__(self, agent_manager: AgentManager):
        self.manager = agent_manager
        logger.info("LLM_API_Interface initialized.")

    def _parse_json_response(self, response_text: Optional[str], task_description: str) -> Optional[dict[str, Any]]:
        """Attempts to parse JSON from the LLM response, with basic cleaning."""
        if not response_text:
            logger.error(f"Received empty response for task: {task_description}")
            return None
        try:
            # Basic cleaning for ```json ... ``` blocks or text outside {}
            response_text = response_text.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                 response_text = response_text[json_start:json_end+1]
            elif response_text: # Warn if non-empty but no braces
                 logger.warning(f"No JSON object markers {{}} in response for {task_description}. Attempting parse. Raw: {response_text[:200]}...")
            else: # Empty after strip
                 logger.error(f"Empty response after stripping for {task_description}"); return None

            parsed_json = json.loads(response_text.strip())
            if not isinstance(parsed_json, dict):
                 logger.error(f"Parsed JSON is not a dictionary for '{task_description}'. Type: {type(parsed_json)}. Resp: {response_text[:200]}"); return None
            return parsed_json
        except json.JSONDecodeError:
            logger.error(f"Failed parse JSON for '{task_description}'. Resp: {response_text[:500]}"); return None
        except Exception as e:
            logger.error(f"Unexpected JSON parsing error for '{task_description}': {e}. Resp: {response_text[:500]}"); return None

    async def _run_assistant_non_streaming(
        self,
        assistant_id: str,
        thread_id: str,
        # user_message: Optional[str], # <<< REMOVED - message added externally now
        additional_instructions: Optional[str], # <<< ADDED - This holds the dynamic prompt
        task_description: str, # For logging
        timeout: int = 90,
        max_retries: int = 1,
        retry_delay_base: float = 2.0
        ) -> Optional[str]:
         """
         Core function to run assistant non-streamingly, handling polling and retries.
         Assumes the relevant USER message was already added to the thread externally.
         Uses 'additional_instructions' for dynamic, turn-specific context/tasks.
         Returns the raw text response from the assistant's message.
         """
         if not assistant_id or not thread_id:
             logger.error(f"Missing assistant_id ('{assistant_id}') or thread_id ('{thread_id}') for task '{task_description}'."); return None

         # --- Message addition is now handled *before* calling this function ---
         # logger.debug(f"Adding prompt message to thread {thread_id} for task '{task_description}'")
         # msg_id = await self.manager.add_message_to_thread(thread_id, prompt, role="user") # NO LONGER DONE HERE
         # if not msg_id: logger.error(f"Failed add prompt msg to thread {thread_id} for task '{task_description}'."); return None

         for attempt in range(max_retries + 1):
             run_result = None
             try:
                 if attempt > 0: logger.info(f"Retrying LLM call for '{task_description}' (Attempt {attempt+1}/{max_retries+1})")
                 logger.debug(f"Requesting run for {assistant_id} on thread {thread_id} (Attempt {attempt+1})...")

                 # --- *** Pass additional_instructions correctly *** ---
                 run_result = await self.manager.run_agent_on_thread_non_streaming(
                     assistant_id=assistant_id,
                     thread_id=thread_id,
                     additional_instructions=additional_instructions, # Pass dynamic context/task here
                     timeout_seconds=timeout
                 )
                 # ------------------------------------------------------

                 if not run_result:
                     logger.error(f"LLM run initiation failed for '{task_description}' (Attempt {attempt+1})."); # Simplified error
                     if attempt < max_retries: await asyncio.sleep(retry_delay_base * (2 ** attempt)); continue
                     else: return None

                 run_id = run_result.get("id", "N/A"); status = run_result.get("status")
                 logger.debug(f"Run {run_id} for '{task_description}' status: {status}")

                 if status == "completed":
                     logger.debug(f"Run {run_id} completed. Retrieving message...")
                     messages = await self.manager.get_thread_messages(thread_id, limit=1, order="desc")
                     if messages and messages[0].get("role") == "assistant" and messages[0].get("run_id") == run_id:
                         content_list = messages[0].get("content", [])
                         resp = "".join(c.get("text", {}).get("value", "") for c in content_list if c.get("type") == "text").strip()
                         logger.debug(f"LLM Raw Response ({run_id} for {task_description}): {resp[:200]}...")
                         return resp # SUCCESS
                     else: logger.error(f"Run {run_id} completed for '{task_description}', but failed retrieve msg from thread {thread_id}."); return None

                 elif status == "failed":
                      last_error = run_result.get('last_error', {}); code = last_error.get('code'); msg = last_error.get('message', '')
                      logger.error(f"Run {run_id} FAILED for '{task_description}'. Code: {code}, Msg: {msg}")
                      if code == 'rate_limit_exceeded':
                           if attempt < max_retries:
                                wait = retry_delay_base
                                try: wait = float(msg.split('try again in ')[1].split('s.')[0]) + 0.5
                                except: wait = retry_delay_base * (2 ** attempt) # Exponential backoff if parse fails
                                logger.warning(f"Rate limit hit for '{task_description}'. Retrying in {wait:.1f}s..."); await asyncio.sleep(wait); continue
                           else: logger.error(f"Rate limit for '{task_description}', max retries reached."); return None
                      else: return None # Other failure, don't retry
                 elif status in ["expired", "cancelled", "requires_action"]: # Requires_action not handled
                     logger.error(f"Run {run_id} for '{task_description}' ended status: {status}."); return None
                 else: # Should include queued, in_progress if polling failed (unlikely with current agent_api logic)
                     logger.error(f"Run {run_id} for '{task_description}' unexpected status: {status}"); # Simplified error
                     if attempt < max_retries: await asyncio.sleep(retry_delay_base * (2 ** attempt)); continue
                     else: return None
             except openai.RateLimitError as e:
                 if attempt < max_retries:
                     wait = retry_delay_base * (2 ** attempt) # Add check for retry-after header if available
                     logger.warning(f"OpenAI RateLimitError hit during API call for '{task_description}'. Retrying in {wait:.1f}s..."); await asyncio.sleep(wait); continue
                 else: logger.error(f"OpenAI RateLimitError for '{task_description}', max retries reached: {e}"); return None
             except Exception as e: logger.error(f"Unexpected error during LLM run for '{task_description}': {e}", exc_info=True); return None # Don't retry others

         logger.error(f"LLM call failed after {max_retries + 1} attempts for '{task_description}'."); return None

    # --- Specific Use Case Methods ---

    async def generate_json_response(
        self,
        assistant_id: str,
        thread_id: str,
        # REMOVED: user_message: str, # This is added externally now
        additional_instructions: str, # This is the dynamic prompt part
        task_description: str = "generate_json_response",
        timeout: int = 90
        ) -> Optional[dict[str, Any]]:
         """
         Runs an assistant non-streamingly. Assumes USER message added externally.
         Uses 'additional_instructions' for dynamic prompt. Parses JSON response.
         """
         raw_response = await self._run_assistant_non_streaming(
              assistant_id=assistant_id,
              thread_id=thread_id,
              # user_message=None, # Not passed here
              additional_instructions=additional_instructions, # Pass dynamic prompt here
              task_description=task_description,
              timeout=timeout
            )
         return self._parse_json_response(raw_response, task_description)

    # (generate_simple_response could be updated similarly if needed)
    async def generate_simple_response(
        self,
        assistant_id: str,
        thread_id: str,
        # REMOVED: user_message: str,
        additional_instructions: str,
        task_description: str = "generate_simple_response",
        timeout: int = 90
        )-> Optional[str]:
         """Runs an assistant non-streamingly. Assumes USER message added externally. Returns raw text."""
         logger.debug(f"--- Generating Simple Response ({task_description}) ---")
         return await self._run_assistant_non_streaming(
             assistant_id=assistant_id,
             thread_id=thread_id,
             # user_message=None,
             additional_instructions=additional_instructions,
             task_description=task_description,
             timeout=timeout
            )
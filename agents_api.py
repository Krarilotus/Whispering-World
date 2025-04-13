# agent_api.py (Refined Version)

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Type  # Keep necessary typing imports

# Use pip install openai python-dotenv
# Assumes Python 3.10+ for | None syntax etc.
from openai import AsyncOpenAI, AssistantEventHandler, RateLimitError, NotFoundError, APIError
from openai.types.beta.threads import Run, ThreadMessage
from openai.types.beta.threads.runs import RunStep
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Logging Setup ---
# Configure logging for better diagnostics in a larger application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL = "gpt-4-turbo-preview" # Or "gpt-4o", "gpt-3.5-turbo"

# --- Streaming Event Handler ---
# Renamed for clarity and slightly simplified event emission
class StreamingEventHandler(AssistantEventHandler):
    """Handles streaming events and pushes structured data to an asyncio Queue."""

    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self._queue = queue
        self._run_id: str | None = None

    async def _put_event(self, event_type: str, data: Any = None, **kwargs):
        """Helper to put structured events onto the queue."""
        event = {"event_type": event_type, "data": data}
        event.update(kwargs)
        await self._queue.put(event)

    # --- Overridden event handlers ---
    async def on_text_created(self, text) -> None:
        await self._put_event("status", "generating_text")

    async def on_text_delta(self, delta, snapshot) -> None:
        await self._put_event("token", delta.value)

    async def on_tool_call_created(self, tool_call):
        await self._put_event("tool_start", tool_call.type)

    async def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                 await self._put_event("tool_delta", delta.code_interpreter.input, subtype="code_input")
            if delta.code_interpreter.outputs:
                output_str = "\n".join(str(o.logs) for o in delta.code_interpreter.outputs if o.type == "logs")
                if output_str:
                     await self._put_event("tool_delta", output_str, subtype="code_output")

    async def on_run_step_created(self, run_step: RunStep) -> None:
        """Capture the run_id early."""
        self._run_id = run_step.run_id
        await self._put_event("run_start", run_id=self._run_id, step_id=run_step.id)

    async def on_end(self):
        await self._put_event("finished")

    async def on_exception(self, exception: Exception):
        logger.error(f"Streaming exception: {exception}")
        await self._put_event("error", str(exception))

    async def on_timeout(self):
        logger.warning("Streaming timeout occurred.")
        await self._put_event("error", "Stream timed out")


# --- Agent Manager Class ---
class AgentManager:
    """
    Manages interactions with OpenAI Assistants (Agents).

    Provides an async interface for creating agents, managing conversation
    threads, running agents with streaming responses, and handling cleanup.
    Focuses on interacting with the OpenAI API; state persistence (beyond
    OpenAI objects) is handled by the caller.
    """
    def __init__(self, api_key: str | None = None, client: AsyncOpenAI | None = None):
        """
        Initializes the Agent Manager.

        Args:
            api_key: OpenAI API key. If None, attempts to load from OPENAI_API_KEY env var.
            client: An existing AsyncOpenAI client instance (optional).
        """
        if client:
            self.client = client
        else:
            key = api_key or OPENAI_API_KEY
            if not key:
                logger.critical("OpenAI API key not provided or found in environment.")
                raise ValueError("OpenAI API key is required.")
            self.client = AsyncOpenAI(api_key=key)
        logger.info("AgentManager initialized.")

    async def create_agent(
        self,
        name: str,
        instructions: str,
        model: str = DEFAULT_MODEL,
        tools: list[dict[str, Any]] | None = None
    ) -> str | None:
        """
        Creates a new OpenAI Assistant.

        Args:
            name: The name for the Assistant.
            instructions: The system prompt defining the agent's behavior.
            model: The OpenAI model to use.
            tools: Optional list of tools (e.g., [{"type": "code_interpreter"}]).

        Returns:
            The Assistant ID if successful, otherwise None.
        """
        logger.info(f"Creating agent '{name}' with model '{model}'...")
        try:
            assistant = await self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=model,
                tools=tools or [],
            )
            logger.info(f"Agent '{name}' created successfully with ID: {assistant.id}")
            return assistant.id
        except APIError as e:
            logger.error(f"API Error creating agent '{name}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error creating agent '{name}': {e}", exc_info=True)
        return None

    async def load_agent(self, assistant_id: str) -> dict[str, Any] | None:
        """
        Retrieves an existing agent by ID to verify and get details.

        Args:
            assistant_id: The ID of the Assistant.

        Returns:
            A dictionary with the assistant's details if found, otherwise None.
        """
        logger.info(f"Loading agent with ID: {assistant_id}...")
        try:
            assistant = await self.client.beta.assistants.retrieve(assistant_id)
            logger.info(f"Agent '{assistant.name}' ({assistant_id}) loaded successfully.")
            return assistant.model_dump()
        except NotFoundError:
            logger.warning(f"Agent with ID {assistant_id} not found.")
        except APIError as e:
            logger.error(f"API Error loading agent {assistant_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error loading agent {assistant_id}: {e}", exc_info=True)
        return None

    async def create_thread(self) -> str | None:
        """
        Creates a new conversation thread.

        Returns:
            The thread ID if successful, otherwise None.
        """
        logger.info("Creating new conversation thread...")
        try:
            thread = await self.client.beta.threads.create()
            logger.info(f"Thread created successfully with ID: {thread.id}")
            return thread.id
        except APIError as e:
            logger.error(f"API Error creating thread: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error creating thread: {e}", exc_info=True)
        return None

    async def add_message_to_thread(
        self,
        thread_id: str,
        content: str,
        role: str = "user",
        attachments: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """
        Adds a message to a specified thread.

        Args:
            thread_id: The ID of the thread.
            content: The text content of the message.
            role: The role ("user" or "assistant").
            attachments: Optional list of attachments for the message (Assistants v2).

        Returns:
            The message ID if successful, otherwise None.
        """
        logger.debug(f"Adding {role} message to thread {thread_id}: '{content[:50]}...'")
        try:
            message = await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content,
                attachments=attachments or []
            )
            logger.debug(f"Message {message.id} added to thread {thread_id}.")
            return message.id
        except APIError as e:
            logger.error(f"API Error adding message to thread {thread_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error adding message to thread {thread_id}: {e}", exc_info=True)
        return None

    async def run_agent_on_thread_stream(
        self,
        assistant_id: str,
        thread_id: str,
        instructions: str | None = None,
        additional_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[asyncio.Queue[dict[str, Any] | None], asyncio.Task, str | None] | None:
        """
        Runs the agent on the thread and streams the response via an asyncio Queue.

        Handles potential race conditions by using the event handler to get the run ID.

        Args:
            assistant_id: The ID of the Assistant (Agent) to run.
            thread_id: The ID of the thread.
            instructions: Optional override instructions for this specific run.
            additional_messages: Optional list of messages to add before this run (Assistants v2).

        Returns:
            A tuple (event_queue, streaming_task, initial_run_id) if run initiation is successful,
            otherwise None. The queue yields event dictionaries or None when finished/error.
            The caller should await the task or cancel it for interruption.
        """
        logger.info(f"Requesting stream run for agent {assistant_id} on thread {thread_id}...")
        event_queue = asyncio.Queue()
        handler = StreamingEventHandler(event_queue)
        initial_run_id: str | None = None # Will be captured by handler

        async def _stream_run_internal():
            run_completed_normally = False
            try:
                async with self.client.beta.threads.runs.stream(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    instructions=instructions,
                    additional_messages=additional_messages,
                    event_handler=handler,
                ) as stream:
                    # Capture the run_id as soon as the handler provides it
                    while initial_run_id is None:
                         # Peek at the queue to get the run_id from the first relevant event
                         # This assumes the handler emits run_start quickly.
                         try:
                              event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
                              if event and event.get("event_type") == "run_start":
                                   global initial_run_id # Modify outer scope variable
                                   initial_run_id = event.get("run_id")
                              # Put it back or handle it - here we just needed the ID,
                              # the caller will process it properly. For simplicity, we
                              # rely on the caller getting it soon after this function returns.
                              # A more robust way might involve passing a future/event.
                              # Let's simplify: assume the caller gets it via the queue.
                         except asyncio.TimeoutError:
                              logger.error("Timeout waiting for run_id from stream handler.")
                              await handler._put_event("error", "Timeout getting run_id")
                              break # Exit if run_id isn't received quickly
                         except Exception as e:
                              logger.error(f"Error getting initial run_id: {e}")
                              await handler._put_event("error", f"Error getting run_id: {e}")
                              break

                    # Let the stream run to completion
                    await stream.until_done()
                    run_details = await stream.get_final_run()
                    logger.info(f"Stream run {run_details.id} completed with status: {run_details.status}")
                    await handler._put_event("run_completed", status=run_details.status, run_id=run_details.id)
                    run_completed_normally = True

            except APIError as e:
                logger.error(f"API Error during streaming run initiation/execution: {e}", exc_info=True)
                await handler._put_event("error", f"API Error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during streaming run: {e}", exc_info=True)
                await handler._put_event("error", f"Unexpected Error: {e}")
            finally:
                # Signal end, regardless of how we exited
                await handler._put_event(None) # Use None as sentinel value
                logger.debug("Streaming task internal function finished.")


        # Start the background task
        streaming_task = asyncio.create_task(_stream_run_internal())

        # Attempt to get the run_id quickly from the queue after starting the task
        # This helps the caller get the ID for potential cancellation before waiting for tokens
        # Note: There's still a slight chance the task finishes before the caller gets the ID if the run is instant
        initial_run_id_from_q = None
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=2.0)
            if event and event.get("event_type") == "run_start":
                initial_run_id_from_q = event.get("run_id")
            # Put the event back for the actual consumer if needed, or structure consumer to handle it
            # Here, we just extract the ID for the return value
            if event:
                event_queue.put_nowait(event) # Put it back quickly

        except asyncio.TimeoutError:
            logger.warning("Did not receive run_id quickly from queue after starting task.")
        except asyncio.QueueEmpty:
             logger.warning("Queue was empty when trying to peek for run_id.")


        return event_queue, streaming_task, initial_run_id_from_q


    async def cancel_run(self, thread_id: str, run_id: str) -> bool:
        """
        Attempts to cancel an ongoing run on OpenAI's side.

        Args:
            thread_id: The ID of the thread the run is on.
            run_id: The ID of the run to cancel.

        Returns:
            True if cancellation request was accepted, False otherwise.
        """
        logger.warning(f"Attempting to cancel run {run_id} on thread {thread_id}...")
        try:
            run = await self.client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
            logger.info(f"Cancel request for run {run_id} sent. Current status: {run.status}")
            # Note: Status might be 'cancelling' or already 'cancelled'/'failed' etc.
            return run.status in ["cancelling", "cancelled"]
        except APIError as e:
            logger.error(f"API Error cancelling run {run_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error cancelling run {run_id}: {e}", exc_info=True)
        return False

    async def get_thread_messages(self, thread_id: str, limit: int = 100, order: str = "desc") -> list[dict[str, Any]]:
        """
        Retrieves messages from a thread, ordered as specified.

        Args:
            thread_id: The ID of the thread.
            limit: Maximum number of messages.
            order: "asc" for oldest first, "desc" for newest first.

        Returns:
            A list of message dictionaries, or an empty list on error.
        """
        logger.debug(f"Retrieving messages for thread {thread_id} (limit {limit}, order {order})...")
        try:
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit,
                order=order # type: ignore - Literal type hint can cause issues sometimes
            )
            return [msg.model_dump() for msg in messages.data]
        except APIError as e:
            logger.error(f"API Error retrieving messages for thread {thread_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error retrieving messages for thread {thread_id}: {e}", exc_info=True)
        return []

    # --- Context/World Knowledge Injection ---
    async def inject_world_knowledge(self, thread_id: str, knowledge: str) -> str | None:
        """
        Injects world knowledge as a specific user message before the next agent run.
        Best used *before* calling run_agent_on_thread_stream.

        Args:
            thread_id: The thread to add the knowledge to.
            knowledge: The world knowledge string.

        Returns:
            The message ID if successful, None otherwise.
        """
        logger.info(f"Injecting world knowledge into thread {thread_id}...")
        # Format it clearly so the LLM recognizes it as context, not direct user speech
        context_message = f"--- Relevant World Knowledge Update ---\n{knowledge}\n--- End Update ---"
        return await self.add_message_to_thread(thread_id, context_message, role="user")


    # --- Cleanup ---
    async def delete_assistant(self, assistant_id: str) -> bool:
        """Deletes an Assistant from OpenAI."""
        logger.warning(f"Requesting deletion of assistant {assistant_id}...")
        try:
            await self.client.beta.assistants.delete(assistant_id)
            logger.info(f"Assistant {assistant_id} deleted successfully.")
            return True
        except NotFoundError:
            logger.warning(f"Assistant {assistant_id} not found for deletion.")
            return False
        except APIError as e:
            logger.error(f"API Error deleting assistant {assistant_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error deleting assistant {assistant_id}: {e}", exc_info=True)
        return False

    async def delete_thread(self, thread_id: str) -> bool:
        """Deletes a thread from OpenAI."""
        logger.warning(f"Requesting deletion of thread {thread_id}...")
        try:
            await self.client.beta.threads.delete(thread_id)
            logger.info(f"Thread {thread_id} deleted successfully.")
            return True
        except NotFoundError:
            logger.warning(f"Thread {thread_id} not found for deletion.")
            return False
        except APIError as e:
            logger.error(f"API Error deleting thread {thread_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error deleting thread {thread_id}: {e}", exc_info=True)
        return False


# --- Example Usage (Async Main Function) ---
async def main_example():
    """Demonstrates the usage of the refined AgentManager."""
    manager = AgentManager() # Uses env var for API key by default
    assistant_id = None
    thread_id = None
    current_stream_task: asyncio.Task | None = None
    current_run_id: str | None = None

    # --- Task manager for handling interruption ---
    input_task = None
    stream_consumer_task = None

    try:
        # --- 1. Create or Load Agent ---
        logger.info("STEP 1: Create or Load Agent")
        # Let's assume we create one for this example
        assistant_id = await manager.create_agent(
            name="Adventure Guide",
            instructions="You are a slightly sarcastic guide in a text adventure. Provide hints but don't give away answers directly. Keep responses relatively short."
        )
        if not assistant_id: return # Exit if creation failed

        # --- 2. Create Thread ---
        logger.info("STEP 2: Create Thread")
        thread_id = await manager.create_thread()
        if not thread_id: return # Exit if creation failed

        # --- 3. Conversation Loop ---
        logger.info("STEP 3: Start Conversation Loop")
        print("\n--- Starting Conversation ---")
        print("Type your message, 'quit' to exit, or 'interrupt' WHILE the assistant is responding.")

        while True:
            # --- Get User Input Asynchronously ---
            # Use asyncio.to_thread for non-blocking input
            input_task = asyncio.create_task(asyncio.to_thread(input, "You: "))
            done, pending = await asyncio.wait(
                {input_task, stream_consumer_task} if stream_consumer_task else {input_task},
                return_when=asyncio.FIRST_COMPLETED
            )

            user_input = None
            if input_task in done:
                 user_input = await input_task
            else:
                 # If input task is not done, the stream consumer must have finished/errored
                 # We let the loop continue to handle potential stream errors or completion
                 pass

            # If stream finished while waiting for input, handle its completion
            if stream_consumer_task and stream_consumer_task in done:
                 try:
                     await stream_consumer_task # Check for exceptions from the consumer task
                 except Exception as e:
                      logger.error(f"Stream consumer task ended with error: {e}")
                 stream_consumer_task = None # Reset consumer task
                 current_run_id = None

            if user_input is None and not stream_consumer_task:
                 # This happens if stream finished AND user didn't input anything (shouldn't normally occur with input())
                 continue


            # --- Handle User Commands ---
            if user_input:
                if user_input.lower() == 'quit':
                    # If a stream is running, cancel it before quitting
                    if stream_consumer_task and not stream_consumer_task.done():
                        logger.info("Interrupting active stream before quitting...")
                        stream_consumer_task.cancel()
                        if current_run_id:
                            await manager.cancel_run(thread_id, current_run_id)
                        await asyncio.sleep(0.5) # Give cancellation a moment
                    break # Exit the main loop

                if user_input.lower() == 'interrupt':
                    if stream_consumer_task and not stream_consumer_task.done():
                        logger.warning("User requested interruption.")
                        stream_consumer_task.cancel() # Cancel the consumer task first
                        if current_run_id:
                            await manager.cancel_run(thread_id, current_run_id) # Attempt OpenAI cancel
                        stream_consumer_task = None # Reset
                        current_run_id = None
                        print("\n[Interruption requested. Previous response cancelled.]")
                    else:
                        print("[No active response to interrupt.]")
                    continue # Get next user input immediately

                # --- Add message and start new run ---
                if stream_consumer_task and not stream_consumer_task.done():
                     print("[Please wait for the current response or use 'interrupt'.]")
                     # Reschedule input task if it was the one that finished
                     if input_task in done:
                         input_task = None # Avoid race condition
                     continue


                logger.info(f"Adding user message: '{user_input[:50]}...'")
                msg_id = await manager.add_message_to_thread(thread_id, user_input)
                if not msg_id:
                     print("[Error adding message to thread. Please try again.]")
                     continue

                # Example: Inject dynamic world knowledge before running
                knowledge = f"The player is currently in room '{player.get('location', 'unknown')}'. Door locked: {game_state.get('door_locked', 'unknown')}."
                # You could inject this using manager.inject_world_knowledge(thread_id, knowledge)
                # OR pass it as run instructions:
                run_instructions = f"World State Context: {knowledge}\n\nRespond to the user's last message."

                logger.info("Starting agent run...")
                stream_result = await manager.run_agent_on_thread_stream(
                    assistant_id,
                    thread_id,
                    instructions=run_instructions
                )

                if stream_result:
                    event_queue, task, initial_run_id = stream_result
                    current_run_id = initial_run_id # Store the run ID we got back
                    logger.info(f"Run started with ID: {current_run_id}")

                    # --- Define the concurrent stream consumer task ---
                    async def consume_stream():
                        assistant_response = ""
                        print("Assistant: ", end="")
                        try:
                            while True:
                                event = await event_queue.get()
                                if event is None: # End signal
                                    logger.debug("Stream consumer received None sentinel.")
                                    break

                                event_type = event.get("event_type")
                                if event_type == "token":
                                    print(event["data"], end="", flush=True)
                                    assistant_response += event["data"]
                                elif event_type == "error":
                                    print(f"\n[STREAM ERROR: {event.get('data')}]")
                                    break # Stop processing on error
                                elif event_type == "run_start":
                                     # We already got the initial_run_id, but can log step id
                                     logger.debug(f"Run Step {event.get('step_id')} started for Run {event.get('run_id')}")
                                elif event_type == "run_completed":
                                    logger.debug(f"Run {event.get('run_id')} completed with status {event.get('status')}")
                                     # Optional: Add assistant's final message to thread history if needed?
                                     # No, the API does this automatically.

                                # Allow other tasks to run
                                await asyncio.sleep(0.01)

                            print() # Newline after response fully received
                            logger.info("Stream consumer finished normally.")

                        except asyncio.CancelledError:
                             print("\n[Response cancelled.]") # Handle cancellation within the consumer
                             logger.warning("Stream consumer task was cancelled.")
                             # No need to cancel OpenAI run here, outer loop handles it
                             raise # Re-raise to signal cancellation to the main loop waiter
                        except Exception as e:
                             logger.error(f"Exception in stream consumer: {e}", exc_info=True)
                             print(f"\n[Error processing response: {e}]")
                        finally:
                             # Ensure newline if printing ended abruptly
                             # print() # Maybe not needed if handled well above
                             pass


                    stream_consumer_task = asyncio.create_task(consume_stream())
                    # Reset input_task to None so we wait for input OR stream completion next iteration
                    input_task = None

                else:
                    print("[Failed to start agent run.]")


    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.critical(f"An critical error occurred in main_example: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.warning("Performing cleanup...")
        # Ensure any lingering tasks are cancelled
        if input_task and not input_task.done(): input_task.cancel()
        if stream_consumer_task and not stream_consumer_task.done(): stream_consumer_task.cancel()

        # Optional: Delete resources on exit
        if assistant_id:
             del_asst = await asyncio.to_thread(input, f"Delete assistant {assistant_id}? [y/N]: ")
             if del_asst.lower() == 'y': await manager.delete_assistant(assistant_id)
        if thread_id:
             del_thread = await asyncio.to_thread(input, f"Delete thread {thread_id}? [y/N]: ")
             if del_thread.lower() == 'y': await manager.delete_thread(thread_id)

        logger.info("Exiting example.")


# --- Dummy game state for example ---
player = {"location": "Guard Room"}
game_state = {"door_locked": True}

# --- Run the example ---
if __name__ == "__main__":
    print("Running Agent API Example...")
    print("NOTE: This example uses asyncio for non-blocking input and streaming.")
    print("It might behave slightly differently in simple terminals vs IDEs.")
    try:
        asyncio.run(main_example())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting.")
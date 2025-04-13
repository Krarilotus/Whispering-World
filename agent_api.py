# agent_api.py (Refined Version - PEP8, Type Hints, Docstrings)

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union # Updated typing

# Use pip install openai python-dotenv
# Assumes Python 3.10+ for | syntax
from openai import AsyncOpenAI, AssistantEventHandler, RateLimitError, NotFoundError, APIError
from openai.types.beta.threads import Run, Message
from openai.types.beta.threads.runs import RunStep
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL = "gpt-4o-mini"  # Recommended starting point
DEFAULT_DM_MODEL = "gpt-4o"  # For vaster more complicated tasks

# --- Streaming Event Handler ---
class StreamingEventHandler(AssistantEventHandler):
    """
    Handles streaming events from the OpenAI Assistants API Run
    and pushes structured data to an asyncio Queue.
    """

    def __init__(self, queue: asyncio.Queue[Optional[Dict[str, Any]]]):
        """
        Initializes the event handler.

        Args:
            queue: An asyncio Queue to put event dictionaries onto.
                   The queue should expect Optional[Dict[str, Any]].
                   `None` is used as a sentinel value to signal the end.
        """
        super().__init__()
        self._queue = queue

    async def _put_event(self, event_type: Optional[str], data: Any = None, **kwargs):
        """
        Helper to put structured events onto the queue. Puts None if event_type is None.
        """
        if event_type is None:
            await self._queue.put(None)
            return

        event = {"event_type": event_type, "data": data}
        event.update(kwargs)
        await self._queue.put(event)

    # --- Overridden event handlers ---
    # (See https://github.com/openai/openai-python/blob/main/src/openai/lib/streaming/_assistants.py)

    async def on_text_created(self, text) -> None:
        """Called when the model starts generating text."""
        await self._put_event("status", "generating_text")
        logger.debug("Event: text_created")

    async def on_text_delta(self, delta, snapshot) -> None:
        """Called for each chunk of generated text."""
        # In v1.x, snapshot is the full text content so far. Delta is the new chunk.
        # logger.debug(f"Event: text_delta - Delta: '{delta.value}'") # Can be verbose
        if delta.value:
             await self._put_event("token", delta.value)

    async def on_tool_call_created(self, tool_call):
        """Called when the model decides to call a tool."""
        await self._put_event("tool_start", tool_call.type, tool_call_id=tool_call.id)
        logger.info(f"Event: tool_call_created - Type: {tool_call.type}, ID: {tool_call.id}")

    async def on_tool_call_delta(self, delta, snapshot):
        """Called for updates during a tool call's execution (e.g., code interpreter)."""
        logger.debug(f"Event: tool_call_delta - Type: {delta.type}, ID: {delta.id}")
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                await self._put_event(
                    "tool_delta",
                    delta.code_interpreter.input,
                    subtype="code_input",
                    tool_call_id=delta.id # Note: The ID is on the delta itself
                )
                logger.debug(f"Tool Delta (Code Input): {delta.code_interpreter.input}")
            if delta.code_interpreter.outputs:
                output_str = "\n".join(
                    str(o.logs)
                    for o in delta.code_interpreter.outputs
                    if o.type == "logs" and o.logs # Ensure logs exist
                )
                if output_str:
                    await self._put_event(
                        "tool_delta",
                        output_str,
                        subtype="code_output",
                        tool_call_id=delta.id
                    )
                    logger.debug(f"Tool Delta (Code Output): {output_str}")
        # Note: Function call deltas are handled via on_tool_call_done in newer patterns,
        # but AssistantEventHandler might not expose argument streaming directly yet.
        # The main text delta usually contains the thought process for function calls.

    async def on_tool_call_done(self, tool_call) -> None:
        """Called when a tool call completes."""
        # This is useful for logging or signaling the end of a specific tool use.
        await self._put_event("tool_end", tool_call.type, tool_call_id=tool_call.id)
        logger.info(f"Event: tool_call_done - Type: {tool_call.type}, ID: {tool_call.id}")
        # For function calls, the result submission happens *after* the run stream.


    async def on_message_created(self, message: Message) -> None:
        """Called when a new message (usually assistant) is created."""
        logger.debug(f"Event: message_created - ID: {message.id}, Role: {message.role}")
        await self._put_event("message_start", role=message.role, message_id=message.id)

    async def on_message_delta(self, delta, snapshot: Message) -> None:
        """Called when parts of the message content change (e.g., text delta)."""
        # Often overlaps with on_text_delta, but gives message context.
        # logger.debug(f"Event: message_delta - ID: {delta.id}") # Can be verbose
        pass # Often redundant if handling text_delta

    async def on_message_done(self, message: Message) -> None:
        """Called when a message is fully processed."""
        logger.debug(f"Event: message_done - ID: {message.id}, Status: {message.status}")
        # You might get the full content here if needed:
        # full_content = "".join([c.text.value for c in message.content if c.type == 'text'])
        await self._put_event("message_end", role=message.role, message_id=message.id, status=message.status)


    async def on_run_step_created(self, run_step: RunStep) -> None:
        """Called when a new step in the run process begins."""
        logger.debug(
            f"Event: run_step_created - RunID: {run_step.run_id}, StepID: {run_step.id}, Type: {run_step.type}"
        )
        # Capture the run_id early if needed, though the stream context provides it too.
        await self._put_event("run_step_start", run_id=run_step.run_id, step_id=run_step.id, step_type=run_step.type)

    async def on_run_step_delta(self, delta, snapshot: RunStep) -> None:
        """Called for incremental updates within a run step."""
        # Useful for tracking detailed progress, e.g., code interpreter execution state.
        step_details = delta.step_details
        logger.debug(f"Event: run_step_delta - StepID: {delta.id}, Details: {step_details}")
        # Example: Extracting code interpreter details if they exist in the delta
        if step_details and step_details.type == "tool_calls":
             for tool_call_delta in step_details.tool_calls or []:
                 # Can potentially mirror on_tool_call_delta logic here if needed
                 pass
        await self._put_event("run_step_delta", step_id=delta.id, delta_details=step_details.model_dump() if step_details else None)


    async def on_run_step_done(self, run_step: RunStep) -> None:
        """Called when a run step completes."""
        logger.debug(
            f"Event: run_step_done - StepID: {run_step.id}, Status: {run_step.status}"
        )
        await self._put_event("run_step_end", step_id=run_step.id, status=run_step.status)

    # --- Lifecycle and Error Handlers ---

    async def on_end(self):
        """Called when the stream finishes naturally."""
        logger.debug("Event: stream_end")
        await self._put_event("finished") # Let's keep 'finished' distinct from the sentinel None
        await self._put_event(None) # Add sentinel None to signal queue consumer

    async def on_exception(self, exception: Exception):
        """Called if an exception occurs during streaming."""
        logger.error(f"Streaming exception: {exception}", exc_info=True)
        await self._put_event("error", str(exception))
        await self._put_event(None) # Add sentinel None

    async def on_timeout(self):
        """Called if the stream times out."""
        logger.warning("Streaming timeout occurred.")
        await self._put_event("error", "Stream timed out")
        await self._put_event(None) # Add sentinel None

    async def on_final_run(self, run: Run):
         """EXPERIMENTAL: Called when the run transitions to a terminal state."""
         # This might be useful for getting final status directly.
         logger.info(f"Event: final_run - RunID: {run.id}, Status: {run.status}")
         await self._put_event("run_completed", status=run.status, run_id=run.id)
         # Note: on_end is usually called shortly after this or on_exception/on_timeout


# --- Agent Manager Class ---
class AgentManager:
    """
    Manages interactions with OpenAI Assistants (Agents).

    Provides an asynchronous interface for creating agents, managing
    conversation threads, running agents with streaming responses,
    handling interruption, and managing cleanup via the OpenAI API.

    Note: This class handles API interactions. Persistence of agent/thread IDs
    and application-specific context/state is the responsibility of the caller.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initializes the Agent Manager.

        Args:
            api_key: OpenAI API key. If None, attempts to load from
                     OPENAI_API_KEY environment variable.
            client: An existing AsyncOpenAI client instance (optional).
                    If provided, api_key is ignored.

        Raises:
            ValueError: If no API key is provided or found in the environment
                        and no client is passed.
        """
        if client:
            self.client = client
            logger.info("AgentManager initialized with provided client.")
        else:
            key = api_key or OPENAI_API_KEY
            if not key:
                logger.critical("OpenAI API key not provided or found in environment.")
                raise ValueError("OpenAI API key is required.")
            self.client = AsyncOpenAI(api_key=key)
            logger.info("AgentManager initialized with API key.")

    async def create_agent(
        self,
        name: str,
        instructions: str,
        model: str = DEFAULT_MODEL,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Creates a new OpenAI Assistant.

        Args:
            name: The name for the Assistant.
            instructions: The system prompt defining the agent's behavior,
                          personality, and background.
            model: The OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini").
            tools: Optional list of tools (e.g., [{"type": "code_interpreter"}]).
                   Function tools require more definition.
            temperature: Sampling temperature (0.0 to 2.0). Higher values make
                         output more random, lower values make it more focused.
            metadata: Optional metadata to associate with the assistant.

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
                temperature=temperature,
                metadata=metadata or {},
            )
            logger.info(f"Agent '{name}' created successfully with ID: {assistant.id}")
            return assistant.id
        except RateLimitError as e:
            logger.error(f"Rate limit error creating agent '{name}': {e}", exc_info=False) 
        except APIError as e:
            logger.error(f"API Error creating agent '{name}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error creating agent '{name}': {e}", exc_info=True)
        return None

    async def load_agent(self, assistant_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an existing agent by ID to verify and get details.

        Args:
            assistant_id: The ID of the Assistant to retrieve.

        Returns:
            A dictionary with the assistant's details (as dict) if found,
            otherwise None.
        """
        logger.info(f"Loading agent with ID: {assistant_id}...")
        try:
            assistant = await self.client.beta.assistants.retrieve(assistant_id)
            logger.info(f"Agent '{assistant.name}' ({assistant_id}) loaded successfully.")
            return assistant.model_dump()  # Return as dictionary
        except NotFoundError:
            logger.warning(f"Agent with ID {assistant_id} not found.")
        except APIError as e:
            logger.error(f"API Error loading agent {assistant_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error loading agent {assistant_id}: {e}", exc_info=True)
        return None

    async def create_thread(
        self,
        metadata: Optional[Dict[str, Any]] = None
        ) -> Optional[str]:
        """
        Creates a new conversation thread.

        Args:
            metadata: Optional metadata to associate with the thread (e.g., player ID, session ID).

        Returns:
            The thread ID if successful, otherwise None.
        """
        logger.info("Creating new conversation thread...")
        try:
            thread = await self.client.beta.threads.create(metadata=metadata or {})
            logger.info(f"Thread created successfully with ID: {thread.id}")
            return thread.id
        except RateLimitError as e:
            logger.error(f"Rate limit error creating thread: {e}", exc_info=False)
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
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Adds a message to a specified thread.

        Args:
            thread_id: The ID of the thread.
            content: The text content of the message.
            role: The role ("user" or "assistant"). Use "user" for player input
                  and injected context. Use "assistant" only if manually adding
                  an expected assistant response (less common).
            attachments: Optional list of attachments for the message (Assistants v2).
                         Useful for file uploads if using Retrieval or Code Interpreter tools.

        Returns:
            The message ID if successful, otherwise None.
        """
        if role not in ["user", "assistant"]:
             logger.error(f"Invalid role '{role}' specified for message.")
             return None

        logger.debug(f"Adding {role} message to thread {thread_id}: '{content[:100]}...'")
        try:
            # Note: Assistants API v2 uses 'attachments' instead of 'file_ids'
            message = await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content,
                attachments=attachments or [], # Use attachments
            )
            logger.debug(f"Message {message.id} added to thread {thread_id}.")
            return message.id
        except NotFoundError:
             logger.warning(f"Thread {thread_id} not found when adding message.")
        except RateLimitError as e:
            logger.error(f"Rate limit error adding message to thread {thread_id}: {e}", exc_info=False)
        except APIError as e:
            logger.error(f"API Error adding message to thread {thread_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error adding message to thread {thread_id}: {e}", exc_info=True)
        return None

    async def run_agent_on_thread_non_streaming(
        self,
        assistant_id: str,
        thread_id: str,
        additional_instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Runs the agent on the thread NON-STREAMINGLY and polls for completion.

        Returns the final Run object details or None on error.

        Args:
            assistant_id: The ID of the Assistant (Agent) to run.
            thread_id: The ID of the thread.
            additional_instructions: Specific instructions for this run.
            temperature: Override the assistant's default temperature for this run.
            metadata: Optional metadata for this specific run.

        Returns:
            A dictionary representing the final Run object if successful, otherwise None.
        """
        logger.info(f"Requesting NON-STREAMING run for agent {assistant_id} on thread {thread_id}...")
        run = None
        try:
            # Create the run (non-streaming)
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions, # Pass instructions if provided
                temperature=temperature,
                metadata=metadata or {},
                # No stream=True, no event_handler
            )
            logger.info(f"Non-streaming run {run.id} created with status: {run.status}")

            # Simple polling loop to wait for completion
            while run.status in ["queued", "in_progress", "cancelling"]:
                await asyncio.sleep(1.5) # Check every 1.5 seconds (adjust as needed)
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )
                logger.debug(f"Polling run {run.id}, status: {run.status}")

            # --- Handle final run status ---
            if run.status == "completed":
                logger.info(f"Non-streaming run {run.id} completed successfully.")
                return run.model_dump() # Return the completed run object details
            elif run.status == "requires_action":
                # NOTE: This basic non-streaming function does NOT handle tool calls.
                # In a real app, you'd extract tool calls from run.required_action,
                # execute them, and submit outputs using runs.submit_tool_outputs.
                logger.warning(f"Non-streaming run {run.id} requires action (Tool Call). This function doesn't handle tool calls.")
                return run.model_dump() # Return the run object so caller sees status
            else:
                # Includes 'failed', 'cancelled', 'expired'
                final_status = run.status
                last_error = run.last_error
                logger.error(f"Non-streaming run {run.id} finished with terminal status: {final_status}. Last Error: {last_error}")
                return run.model_dump() # Return the run object with error details

        except APIError as e:
            logger.error(f"API Error during non-streaming run: Status={e.status_code}, Body={e.body}", exc_info=True)
            if run: return run.model_dump() # Return partial run info if available
            return None
        except Exception as e:
            logger.error(f"Unexpected error during non-streaming run: {type(e).__name__}", exc_info=True)
            if run: return run.model_dump() # Return partial run info if available
            return None

    async def run_agent_on_thread_stream(
        self,
        assistant_id: str,
        thread_id: str,
        instructions: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        temperature: Optional[float] = None, # Allow overriding temperature per run
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[asyncio.Queue[Optional[Dict[str, Any]]], asyncio.Task, str]]:
        """
        Runs the agent on the thread and streams the response via an asyncio Queue.

        Returns a queue for events, a task managing the stream, and the Run ID.

        Args:
            assistant_id: The ID of the Assistant (Agent) to run.
            thread_id: The ID of the thread.
            instructions: DEPRECATED in Assistants API v2. Use additional_instructions.
                          Kept for backward compatibility awareness but will log warning.
            additional_instructions: Specific instructions for this run, appended to
                                     the main assistant instructions. Ideal for dynamic
                                     context or situational guidance.
            temperature: Override the assistant's default temperature for this run.
            metadata: Optional metadata for this specific run.

        Returns:
            A tuple (event_queue, streaming_task, run_id) if run initiation
            is successful, otherwise None.
            The queue yields event dictionaries (see StreamingEventHandler)
            or None when finished/error.
            The caller should await the streaming_task or cancel it for interruption.
            The run_id is returned immediately for potential cancellation.

        Raises:
            ValueError: If the initial run ID cannot be obtained quickly.
        """
        logger.info(f"Requesting stream run for agent {assistant_id} on thread {thread_id}...")
        event_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        handler = StreamingEventHandler(event_queue)
        run_id_capture: List[Optional[str]] = [None] # Use a list to allow modification in closure


        if instructions:
             logger.warning("'instructions' param in run is deprecated in Assistants v2. Use 'additional_instructions'.")
             # Combine if both are somehow provided, prioritizing additional_instructions
             effective_instructions = f"{instructions}\n{additional_instructions}" if additional_instructions else instructions
        else:
             effective_instructions = additional_instructions


        # Use a separate task to start the stream and immediately capture the run_id
        # This addresses potential race conditions where the run finishes extremely fast.
        async def _start_stream_and_get_run_id():
            run_initiated = False # Flag to track if run object was obtained
            run_id_capture[0] = None # Ensure it's None initially
            stream_context_manager = None # To hold the stream context manager object

            try:
                logger.debug("Attempting to create client.beta.threads.runs.stream context manager...")
                try:
                    # *** Create the context manager object first ***
                    stream_context_manager = self.client.beta.threads.runs.stream(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        # additional_instructions=effective_instructions, # Keep commented out for test
                        event_handler=handler,
                        temperature=temperature,
                        metadata=metadata or {},
                    )
                    logger.debug("Stream context manager object created. Attempting to enter context...")

                    # *** Now enter the context ***
                    async with stream_context_manager as stream:
                        logger.debug("Entered stream context successfully.")
                        try:
                            # *** Wrap the current_run() call ***
                            logger.debug("Attempting to get current run details via stream.current_run()...")
                            run = await stream.current_run()
                            # Log details if run object is received
                            if run:
                                logger.debug(f"stream.current_run() returned Run ID: {run.id}, Status: {run.status}")
                            else:
                                logger.warning("stream.current_run() returned None")

                            if run and run.id:
                                run_id_capture[0] = run.id
                                run_initiated = True
                                logger.info(f"Run {run.id} initiated for thread {thread_id}.")
                                # Manually put run_start event as we have the ID now
                                await handler._put_event("run_start", run_id=run.id, step_id=None)
                            else:
                                # This case means context was entered, but current_run() failed
                                logger.error("Stream context entered, but failed to get valid run details from stream.current_run().")
                                await handler._put_event("error", "Failed to obtain valid run details after stream start")
                                await handler._put_event(None)
                                return # Exit this async function

                        # --- Exception handling for stream.current_run() ---
                        except APIError as e_run:
                            logger.error(f"Caught APIError during stream.current_run(): Status={e_run.status_code}, Body={e_run.body}", exc_info=True)
                            await handler._put_event("error", f"API Error getting run details: {e_run}")
                            await handler._put_event(None)
                            return # Exit this async function
                        except Exception as e_run:
                            logger.error(f"Caught unexpected Exception during stream.current_run(): {type(e_run).__name__}", exc_info=True)
                            await handler._put_event("error", f"Unexpected Error getting run details: {e_run}")
                            await handler._put_event(None)
                            return # Exit this async function

                        # --- Proceed only if run was successfully initiated ---
                        if run_initiated:
                            logger.debug(f"Streaming run {run_id_capture[0]}...")
                            try:
                                await stream.until_done()
                                logger.debug(f"Stream for run {run_id_capture[0]} ended naturally via until_done().")
                            except APIError as e_stream:
                                logger.error(f"Caught APIError during stream.until_done(): Status={e_stream.status_code}, Body={e_stream.body}", exc_info=True)
                                # Handler's on_exception should catch this, but log here too
                            except Exception as e_stream:
                                logger.error(f"Caught unexpected Exception during stream.until_done(): {type(e_stream).__name__}", exc_info=True)
                                # Handler's on_exception should catch this

                # --- Exception handling for entering the stream context ---
                except APIError as e_ctx:
                    logger.error(f"Caught APIError entering stream context: Status={e_ctx.status_code}, Body={e_ctx.body}", exc_info=True)
                    await handler._put_event("error", f"API Error entering stream context: {e_ctx}")
                    await handler._put_event(None)
                except Exception as e_ctx:
                    logger.error(f"Caught unexpected Exception entering stream context: {type(e_ctx).__name__}", exc_info=True)
                    await handler._put_event("error", f"Unexpected Error entering stream context: {e_ctx}")
                    await handler._put_event(None)

            # --- Catch-all for the whole function ---
            except Exception as e_outer:
                logger.error(f"Caught unexpected Exception in outer scope of _start_stream_and_get_run_id: {type(e_outer).__name__}", exc_info=True)
                # Ensure sentinel is put if not already done by inner handlers
                if handler._queue.empty() or (not handler._queue._queue[-1] is None):
                    await handler._put_event("error", f"Outer scope error: {e_outer}")
                    await handler._put_event(None)
            finally:
                # Log final status
                logger.debug(f"_start_stream_and_get_run_id finished. Run initiated flag: {run_initiated}, Captured Run ID: {run_id_capture[0]}")
                # Final check to ensure sentinel if run failed and nothing else was put
                if not run_initiated and handler._queue.empty():
                    logger.warning("Run was not initiated and queue is empty in finally block, ensuring None sentinel.")
                    await handler._put_event(None)

        # Start the streaming task
        streaming_task = asyncio.create_task(_start_stream_and_get_run_id(), name=f"StreamTask_{thread_id[:8]}")

        # Wait briefly for the run_id to be captured by the task
        await asyncio.sleep(0.1) # Give the task a moment to start and make the API call

        initial_run_id = run_id_capture[0]

        if not initial_run_id:
             # If the run ID wasn't captured quickly, something likely went wrong during initiation
             logger.error("Failed to initiate run or capture Run ID promptly.")
             # Attempt to clean up the task if it's still running somehow
             if not streaming_task.done():
                 streaming_task.cancel()
                 try:
                      await asyncio.wait_for(streaming_task, timeout=1.0)
                 except (asyncio.CancelledError, asyncio.TimeoutError):
                      pass # Ignore cancellation/timeout errors during cleanup
                 except Exception as e:
                      logger.error(f"Error during stream task cleanup after failed ID capture: {e}")

             # Check the queue for an error message if available
             try:
                 error_event = event_queue.get_nowait()
                 if error_event and error_event.get("event_type") == "error":
                      logger.error(f"Run initiation failed with error: {error_event.get('data')}")
                 elif error_event:
                      event_queue.put_nowait(error_event) # Put back if not error
             except asyncio.QueueEmpty:
                 pass # No error message in queue

             return None # Indicate failure


        # Return the queue, the background task, and the captured run_id
        return event_queue, streaming_task, initial_run_id


    async def cancel_run(self, thread_id: str, run_id: str) -> bool:
        """
        Attempts to cancel an ongoing Assistant run.

        Cancellation is best-effort. The run might complete or fail before
        the cancellation takes effect.

        Args:
            thread_id: The ID of the thread the run is on.
            run_id: The ID of the run to cancel.

        Returns:
            True if the cancellation request was successfully sent (run is now
            'cancelling' or already 'cancelled'), False otherwise (e.g., error,
            run already completed/failed).
        """
        logger.warning(f"Attempting to cancel run {run_id} on thread {thread_id}...")
        try:
            # Retrieve the run first to check its status
            current_run = await self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            terminal_states = {"cancelled", "failed", "completed", "expired"}
            if current_run.status in terminal_states:
                logger.warning(f"Run {run_id} is already in terminal state: {current_run.status}. Cannot cancel.")
                return current_run.status == "cancelled" # Return True only if it was already cancelled

            # If cancellable, proceed to cancel
            run = await self.client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
            logger.info(f"Cancel request for run {run_id} processed. Run status: {run.status}")
            # Check if status reflects cancellation attempt
            return run.status in ["cancelling", "cancelled"]
        except NotFoundError:
            logger.warning(f"Run {run_id} or Thread {thread_id} not found for cancellation.")
            return False
        except RateLimitError as e:
             logger.error(f"Rate limit error cancelling run {run_id}: {e}", exc_info=False)
             return False
        except APIError as e:
            # Handle cases like "Run is not active"
            if "already in terminal status" in str(e) or "is not active" in str(e):
                 logger.warning(f"Could not cancel run {run_id}: Already in a terminal state or inactive. API message: {e}")
            else:
                 logger.error(f"API Error cancelling run {run_id}: {e}", exc_info=True)
            return False # Indicate cancellation wasn't effectively performed now
        except Exception as e:
            logger.error(f"Unexpected error cancelling run {run_id}: {e}", exc_info=True)
            return False

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves messages from a thread, ordered as specified.

        Args:
            thread_id: The ID of the thread.
            limit: Maximum number of messages to retrieve (1-100). Defaults to 20.
            order: "asc" for oldest first, "desc" for newest first. Defaults to "desc".
            after: Retrieve messages after this message ID.
            before: Retrieve messages before this message ID.

        Returns:
            A list of message dictionaries (using model_dump), or an empty list on error.
            Messages are ordered according to the 'order' parameter.
        """
        if order not in ["asc", "desc"]:
             logger.error("Invalid order parameter for get_thread_messages. Use 'asc' or 'desc'.")
             return []

        logger.debug(f"Retrieving messages for thread {thread_id} (limit {limit}, order {order})...")
        try:
            messages_page = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit,
                order=order, # type: ignore[arg-type] - SDK hints can be strict
                after=after,
                before=before,
            )
            # `messages_page.data` contains the list of message objects
            return [msg.model_dump() for msg in messages_page.data]
        except NotFoundError:
             logger.warning(f"Thread {thread_id} not found when retrieving messages.")
        except APIError as e:
            logger.error(f"API Error retrieving messages for thread {thread_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error retrieving messages for thread {thread_id}: {e}", exc_info=True)
        return []

    # --- Context/World Knowledge Injection ---
    async def inject_world_knowledge(
        self, thread_id: str, knowledge: str
    ) -> Optional[str]:
        """
        Injects world knowledge as a specific user message.

        This is useful for providing dynamic, out-of-band context to the agent
        before its next run. Call this *before* `run_agent_on_thread_stream`.

        Args:
            thread_id: The thread to add the knowledge to.
            knowledge: The world knowledge string (e.g., game state, player status).

        Returns:
            The message ID of the injected knowledge if successful, None otherwise.
        """
        logger.info(f"Injecting world knowledge into thread {thread_id}...")
        # Format clearly so the LLM recognizes it as context, distinct from direct user speech.
        # Using Markdown-like blocks can help.
        context_message = f"--- System Update: Current World State ---\n{knowledge}\n--- End Update ---"
        # Alternatively, structure it more like data:
        # context_message = f"[World Context]\nPlayer Location: {loc}\nInventory: {inv}\nObjective: {obj}"

        return await self.add_message_to_thread(
            thread_id=thread_id,
            content=context_message,
            role="user" # Inject context as if the user is providing it *to* the assistant
        )

    # --- Context Summarization (Conceptual - Implementation is External) ---
    async def request_context_summary(
        self,
        assistant_id: str, # Need an assistant to do the summarizing
        thread_id: str,
        summary_instructions: str = "Summarize the key events, decisions, and character relationship changes from this conversation concisely. Focus on information crucial for future interactions. Output ONLY the summary.",
        model: str = DEFAULT_DM_MODEL, # Can use a more intelligent model for summary
        temperature: float = 0.2, # Lower temp for factual summary
    ) -> Optional[str]:
        """
        Requests the Assistant to summarize the conversation in the thread.

        This runs the assistant one more time with specific instructions to summarize.
        The summary is returned as a string. The caller is responsible for storing
        and managing this summary externally.

        Args:
            assistant_id: The Assistant ID to use for summarization.
            thread_id: The thread containing the conversation to summarize.
            summary_instructions: The prompt instructing the assistant how to summarize.
            model: Model to use for the summarization run.
            temperature: Temperature for the summarization run.

        Returns:
            The generated summary text if successful, otherwise None.
            Note: This performs a non-streaming run.
        """
        logger.info(f"Requesting conversation summary for thread {thread_id}...")
        try:
            # Use run.create (non-streaming) for a single summarization task
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                model=model, # Allow specifying model for summary
                instructions=summary_instructions, # Use specific summary prompt
                temperature=temperature,
            )
            logger.debug(f"Summary run created: {run.id}")

            # Poll for completion (simple polling loop)
            while run.status in ["queued", "in_progress", "cancelling"]:
                await asyncio.sleep(1) # Wait 1 second between checks
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )
                logger.debug(f"Summary run {run.id} status: {run.status}")

            if run.status == "completed":
                logger.info(f"Summary run {run.id} completed.")
                # Retrieve the *last* assistant message added by this run
                messages = await self.get_thread_messages(thread_id, limit=1, order="desc")
                if messages and messages[0].get("run_id") == run.id and messages[0].get("role") == "assistant":
                    # Extract text content
                    summary_content = messages[0].get("content", [])
                    summary_text = ""
                    for block in summary_content:
                         if block.get("type") == "text":
                             summary_text += block.get("text", {}).get("value", "")
                    if summary_text:
                         logger.info("Summary extracted successfully.")
                         return summary_text.strip()
                    else:
                         logger.warning("Summary run completed, but no text content found in the last assistant message.")
                else:
                     logger.warning("Summary run completed, but couldn't retrieve the corresponding assistant message.")

            else:
                logger.error(f"Summary run {run.id} failed or was cancelled. Status: {run.status}. Error: {run.last_error}")
                return None

        except RateLimitError as e:
            logger.error(f"Rate limit error requesting summary for thread {thread_id}: {e}", exc_info=False)
        except NotFoundError:
             logger.warning(f"Thread {thread_id} or Assistant {assistant_id} not found for summary.")
        except APIError as e:
            logger.error(f"API Error requesting summary for thread {thread_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error requesting summary for thread {thread_id}: {e}", exc_info=True)

        return None


    # --- Cleanup ---
    async def delete_assistant(self, assistant_id: str) -> bool:
        """
        Deletes an Assistant from OpenAI.

        Args:
            assistant_id: The ID of the Assistant to delete.

        Returns:
            True if deletion was successful or assistant was already gone, False on error.
        """
        logger.warning(f"Requesting deletion of assistant {assistant_id}...")
        try:
            response = await self.client.beta.assistants.delete(assistant_id)
            logger.info(f"Assistant {assistant_id} deletion result: {response}")
            return response.deleted # Check the deletion status in the response
        except NotFoundError:
            logger.warning(f"Assistant {assistant_id} not found for deletion (already deleted?).")
            return True # Treat as success if not found
        except RateLimitError as e:
             logger.error(f"Rate limit error deleting assistant {assistant_id}: {e}", exc_info=False)
             return False
        except APIError as e:
            logger.error(f"API Error deleting assistant {assistant_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting assistant {assistant_id}: {e}", exc_info=True)
            return False

    async def delete_thread(self, thread_id: str) -> bool:
        """
        Deletes a thread from OpenAI.

        Args:
            thread_id: The ID of the thread to delete.

        Returns:
            True if deletion was successful or thread was already gone, False on error.
        """
        logger.warning(f"Requesting deletion of thread {thread_id}...")
        try:
            response = await self.client.beta.threads.delete(thread_id)
            logger.info(f"Thread {thread_id} deletion result: {response}")
            return response.deleted
        except NotFoundError:
            logger.warning(f"Thread {thread_id} not found for deletion (already deleted?).")
            return True # Treat as success if not found
        except RateLimitError as e:
             logger.error(f"Rate limit error deleting thread {thread_id}: {e}", exc_info=False)
             return False
        except APIError as e:
            logger.error(f"API Error deleting thread {thread_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting thread {thread_id}: {e}", exc_info=True)
            return False

# --- Example Usage (Moved to separate file/function for clarity) ---
# See game_loop_mockup.py for a demonstration.
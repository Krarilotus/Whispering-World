# game_loop_mockup.py (Example Usage)

import asyncio
import logging
import sys
import os
from typing import Optional, Any

# Assuming agent_api.py and agent.py are in the same directory or Python path
from agent_api import AgentManager, DEFAULT_MODEL
from agent import GameAgent

# --- Logging Setup ---
# Use the same logging config or customize
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Game State (Simple Example) ---
# In a real game, this would be loaded/managed more robustly
player_state = {
    "name": "Adventurer",
    "location": "Damp Cellar",
    "inventory": ["Rusty Spoon", "Lint"],
    "objective": "Escape the cellar",
}

guard_agent: Optional[GameAgent] = None
manager: Optional[AgentManager] = None

# --- Agent Configuration ---
GUARD_NAME = "Borg the Guard"
GUARD_ASSISTANT_ID = None # Replace with actual ID if loading, otherwise will be created
GUARD_INSTRUCTIONS = """
You are Borg, a grumpy but easily bored guard stationed in a damp cellar.
Your primary instruction is to prevent the player (an Adventurer) from escaping. You hold the only key.
You are not inherently malicious, just following orders and quite bored with your post.
You respond grumpily or sarcastically initially.
However, you have a weakness: you enjoy hearing interesting stories, gossip, or being flattered, which might make you slightly more lenient or distractible, but you won't just give the key away easily.
You know the player is locked in this cellar.
Keep your responses relatively concise, fitting for a guard. Do not break character.
Refer to the player as 'Prisoner' or 'Adventurer'.
Use the provided World State Context for situational awareness. Do not explicitly mention the context block itself in your response.
"""

# --- Async Functions ---

async def setup_game():
    """Initializes the AgentManager and creates/loads the Guard agent."""
    global manager, guard_agent, GUARD_ASSISTANT_ID
    try:
        manager = AgentManager() # Uses env var for API key

        # --- 1. Create or Load Agent ---
        logger.info("STEP 1: Setting up the Guard Agent...")
        existing_agent_id = GUARD_ASSISTANT_ID # Set this if you have a pre-existing ID

        if existing_agent_id:
             agent_details = await manager.load_agent(existing_agent_id)
             if agent_details:
                 logger.info(f"Successfully loaded existing agent '{GUARD_NAME}' ({existing_agent_id}).")
                 guard_agent = GameAgent(name=GUARD_NAME, assistant_id=existing_agent_id)
                 # In a real game, load thread_id and other state from save data here
             else:
                 logger.warning(f"Failed to load agent {existing_agent_id}. Will create a new one.")
                 existing_agent_id = None # Force creation

        if not guard_agent: # If not loaded or loading failed
            logger.info(f"Creating new agent '{GUARD_NAME}'...")
            new_assistant_id = await manager.create_agent(
                name=GUARD_NAME,
                instructions=GUARD_INSTRUCTIONS,
                model=DEFAULT_MODEL, # e.g., "gpt-4o"
                temperature=0.8, # Slightly more creative guard
                # Add tools if needed, e.g., {"type": "code_interpreter"}
            )
            if new_assistant_id:
                logger.info(f"Agent '{GUARD_NAME}' created with ID: {new_assistant_id}")
                guard_agent = GameAgent(name=GUARD_NAME, assistant_id=new_assistant_id)
                GUARD_ASSISTANT_ID = new_assistant_id # Store for potential reuse/saving
            else:
                logger.critical("Failed to create the Guard agent. Exiting.")
                return False # Indicate setup failure

        return True # Indicate setup success

    except ValueError as e:
         logger.critical(f"Initialization error: {e}")
         return False
    except Exception as e:
        logger.critical(f"An unexpected error occurred during setup: {e}", exc_info=True)
        return False


async def handle_stream(
    event_queue: asyncio.Queue[Optional[dict[str, Any]]], agent_name: str
) -> str:
    """Consumes the event queue and prints the assistant's response."""
    full_response = ""
    print(f"{agent_name}: ", end="", flush=True)
    try:
        while True:
            event = await event_queue.get()
            if event is None: # End signal
                logger.debug("Stream consumer received None sentinel.")
                break

            event_type = event.get("event_type")

            if event_type == "token":
                token = event.get("data", "")
                print(token, end="", flush=True)
                full_response += token
            elif event_type == "error":
                error_msg = event.get('data', 'Unknown stream error')
                print(f"\n[STREAM ERROR: {error_msg}]", flush=True)
                logger.error(f"Stream error received: {error_msg}")
                # Decide if error terminates interaction or just logs
                break # Stop processing this response on error
            elif event_type == "status":
                 logger.debug(f"Assistant status: {event.get('data')}")
            elif event_type == "tool_start":
                 logger.info(f"Assistant started using tool: {event.get('data')}")
                 print(f" [{event.get('data')}...]", end="", flush=True) # Indicate tool use
            elif event_type == "tool_end":
                 logger.info(f"Assistant finished using tool: {event.get('data')}")
                 # Optionally print indication tool finished
            elif event_type in ["run_start", "run_step_start", "run_step_end", "run_completed", "message_start", "message_end"]:
                 # Log detailed events if needed for debugging
                 logger.debug(f"Stream Event: {event_type} - Data: {event.get('data') or event}")
            # Add handling for other specific events if needed (e.g., tool outputs)

            # Allow other tasks to run briefly
            await asyncio.sleep(0.005)

    except asyncio.CancelledError:
        print("\n[Response cancelled by user.]", flush=True)
        logger.warning("Stream consumer task was cancelled.")
        # The cancellation is handled, just need to exit gracefully
        raise # Re-raise to signal cancellation to the main loop waiter
    except Exception as e:
        logger.error(f"Exception in stream consumer: {e}", exc_info=True)
        print(f"\n[Error processing response: {e}]", flush=True)
        # Don't re-raise normal exceptions, let the loop continue if possible
    finally:
        print() # Ensure a newline after the response or cancellation message
        # Make sure queue is consumed if exited abnormally
        while not event_queue.empty():
            try:
                event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            event_queue.task_done() # Mark task as done even if consuming leftovers
        # Only call task_done once for the None sentinel if received normally
        # The loop breaks on None, so task_done() is called outside after break
        if event is None:
            event_queue.task_done()


    return full_response

async def game_loop():
    """Main game interaction loop."""
    global guard_agent, manager

    if not guard_agent or not manager:
         logger.critical("Game not set up correctly. Agent or Manager is missing.")
         return

    logger.info("Starting game loop...")
    print("\n--- You are in a Damp Cellar ---")
    print(f"A grumpy guard, {guard_agent.name}, blocks the only door.")
    print("Type your message, 'quit' to exit, or 'interrupt' to stop the guard talking.")

    current_stream_task: Optional[asyncio.Task] = None
    current_run_id: Optional[str] = None
    input_task: Optional[asyncio.Task] = None

    try:
        # --- Ensure Thread Exists ---
        if not guard_agent.thread_id:
            logger.info(f"No existing thread for {guard_agent.name}. Creating one.")
            thread_id = await manager.create_thread(metadata={"player": player_state["name"], "agent": guard_agent.name})
            if thread_id:
                guard_agent.set_thread_id(thread_id)
            else:
                logger.error("Failed to create conversation thread. Cannot continue.")
                return

        while True:
            # --- Wait for User Input OR Stream Completion ---
            tasks_to_wait_for = set()
            if not input_task: # Create input task only if previous one is done or non-existent
                 # Use asyncio.to_thread for non-blocking input in terminal
                 input_task = asyncio.create_task(asyncio.to_thread(input, f"{player_state['name']}: "))
            tasks_to_wait_for.add(input_task)

            if current_stream_task and not current_stream_task.done():
                tasks_to_wait_for.add(current_stream_task)

            # Wait for either input or the stream to finish/cancel
            done, pending = await asyncio.wait(
                tasks_to_wait_for, return_when=asyncio.FIRST_COMPLETED
            )

            user_input: Optional[str] = None
            stream_finished_or_cancelled = False

            # --- Process Completed Tasks ---
            if input_task in done:
                try:
                    user_input = await input_task
                except Exception as e:
                     logger.error(f"Error getting user input: {e}")
                     # Decide how to handle input errors, maybe break or continue
                     break
                input_task = None # Reset input task as it's completed

            if current_stream_task in done:
                stream_finished_or_cancelled = True
                try:
                    # Await the task to raise exceptions (like CancelledError)
                    await current_stream_task
                except asyncio.CancelledError:
                    logger.info("Stream task was confirmed cancelled.")
                    # Interruption already logged by handler/user command
                except Exception as e:
                    # Errors during streaming are usually caught by handle_stream,
                    # but this catches errors in the task logic itself.
                    logger.error(f"Stream consumer task failed unexpectedly: {e}", exc_info=True)
                # Regardless of how it finished, it's done.
                current_stream_task = None
                current_run_id = None

            # --- Handle User Input / Commands ---
            if user_input is not None:
                user_input = user_input.strip()
                if user_input.lower() == 'quit':
                    logger.info("User requested quit.")
                    if current_stream_task and not current_stream_task.done():
                        logger.info("Interrupting active stream before quitting...")
                        current_stream_task.cancel()
                        if current_run_id and guard_agent.thread_id:
                            await manager.cancel_run(guard_agent.thread_id, current_run_id)
                        # Wait briefly for cancellation to potentially propagate
                        await asyncio.sleep(0.2)
                    break # Exit the main loop

                elif user_input.lower() == 'interrupt':
                    logger.warning("User requested interruption.")
                    if current_stream_task and not current_stream_task.done():
                        current_stream_task.cancel() # Cancel the consumer task first
                        # The CancelledError will be caught when awaiting the task later
                        if current_run_id and guard_agent.thread_id:
                            # Attempt OpenAI side cancellation (best effort)
                            cancelled_on_api = await manager.cancel_run(guard_agent.thread_id, current_run_id)
                            logger.info(f"API cancellation attempt result: {cancelled_on_api}")
                        # Don't reset stream_task/run_id here, wait for the task to finish cancelling
                        print("[Interruption requested...]", flush=True)
                    else:
                        print("[No active response to interrupt.]", flush=True)
                    # Loop continues, will wait for input again or stream cancellation

                elif current_stream_task and not current_stream_task.done():
                     # User typed something while assistant was still responding
                     print("[Please wait for the current response to finish or use 'interrupt'.]", flush=True)
                     # Keep the input task pending if it exists, otherwise create a new one next loop

                else:
                    # --- Process regular user message ---
                    logger.info(f"Adding user message: '{user_input[:50]}...'")
                    if not guard_agent.thread_id:
                         logger.error("Critical error: Agent has no thread ID.")
                         break

                    msg_id = await manager.add_message_to_thread(guard_agent.thread_id, user_input, role="user")
                    if not msg_id:
                        print("[Error sending message to the guard. Please try again.]", flush=True)
                        continue # Get new input

                    # --- Inject Context / Run Agent ---
                    # Get current context from game state / agent state
                    world_context = (
                         f"Player Location: {player_state['location']}\n"
                         f"Player Inventory: {', '.join(player_state['inventory'])}\n"
                         f"Player Objective: {player_state['objective']}\n"
                         f"--- Agent Internal State ---\n"
                         f"{guard_agent.get_current_game_state_context()}"
                    )

                    # Option 1: Inject as a separate message (can make logs clearer)
                    # await manager.inject_world_knowledge(guard_agent.thread_id, world_context)
                    # await asyncio.sleep(0.1) # Small delay sometimes helps ensure message order processing

                    # Option 2: Pass as additional instructions (more direct for the run)
                    run_instructions = f"--- World State Context ---\n{world_context}\n--- End Context ---"

                    logger.info("Starting agent run...")
                    stream_result = await manager.run_agent_on_thread_stream(
                        assistant_id=guard_agent.assistant_id,
                        thread_id=guard_agent.thread_id,
                        additional_instructions=run_instructions, # Use this for context
                        # temperature=0.9 # Can override temperature per run if desired
                    )

                    if stream_result:
                        event_queue, task, run_id = stream_result
                        current_stream_task = task
                        current_run_id = run_id # Store for potential cancellation
                        logger.info(f"Agent run initiated with Run ID: {current_run_id}")

                        # Assign the stream handling to the task variable for the next loop iteration
                        # The actual consumption happens via asyncio.wait
                        current_stream_task = asyncio.create_task(
                            handle_stream(event_queue, guard_agent.name),
                            name=f"ConsumerTask_{guard_agent.thread_id[:8]}"
                        )

                    else:
                        print("[Failed to start the guard's response.]", flush=True)
                        current_run_id = None
                        current_stream_task = None

            elif stream_finished_or_cancelled:
                 # Stream finished while waiting for input, just loop again for input
                 logger.debug("Stream finished or was cancelled. Waiting for user input.")
                 pass # Continue loop to wait for input_task


    except asyncio.CancelledError:
        logger.info("Main game loop cancelled.")
    except Exception as e:
        logger.critical(f"An critical error occurred in the game loop: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.warning("Performing cleanup...")
        # Ensure any lingering tasks are cancelled
        if input_task and not input_task.done(): input_task.cancel()
        if current_stream_task and not current_stream_task.done(): current_stream_task.cancel()

        # Wait briefly for tasks to acknowledge cancellation
        await asyncio.sleep(0.5)

        # --- Optional: Save state / Summarize / Delete resources ---
        if guard_agent and guard_agent.thread_id and manager:
            # Example: Request summary before potentially deleting thread
            # print("Requesting conversation summary...")
            # summary = await manager.request_context_summary(
            #     assistant_id=guard_agent.assistant_id,
            #     thread_id=guard_agent.thread_id
            # )
            # if summary:
            #     print(f"\n--- Conversation Summary --- \n{summary}\n--------------------------")
            #     # Here you would save the summary string to your game save file
            #     # associated with this agent/interaction.
            #     guard_agent.update_state(summary=summary) # Update agent object too
            # else:
            #     print("[Failed to get conversation summary.]")


             # Example: Ask before deleting OpenAI resources
             # Use run_in_executor for blocking input in async finally block
             loop = asyncio.get_running_loop()
             del_thread_q = await loop.run_in_executor(None, input, f"Delete thread {guard_agent.thread_id}? [y/N]: ")
             if del_thread_q.lower() == 'y':
                 await manager.delete_thread(guard_agent.thread_id)

             # Only delete assistant if it was newly created for this session maybe?
             # Or based on game logic (agent dies, etc.)
             # del_asst_q = await loop.run_in_executor(None, input, f"Delete assistant {guard_agent.assistant_id}? [y/N]: ")
             # if del_asst_q.lower() == 'y':
             #     await manager.delete_assistant(guard_agent.assistant_id)


        logger.info("Exiting game mockup.")


# --- Run the example ---
if __name__ == "__main__":
    print("Running OpenAI Agent Game Mockup...")
    print("NOTE: Uses asyncio for non-blocking input and streaming.")
    print("Requires OPENAI_API_KEY environment variable.")

    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
         print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
         sys.exit(1)

    setup_ok = asyncio.run(setup_game())

    if setup_ok:
        try:
            asyncio.run(game_loop())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, exiting gracefully.")
            # asyncio's shutdown logic will handle cancelling tasks in finally block
        except Exception as e:
             logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)

    else:
         print("Game setup failed. Exiting.")
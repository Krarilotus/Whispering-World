# game_loop_mockup.py (FINAL VERSION with Helper Function)

import asyncio
import logging
import sys
import os
import argparse
from typing import Optional, Any, Dict, Tuple # Added Tuple

# Assuming agent_api.py and agent.py are in the same directory or Python path
from agent_api import AgentManager, DEFAULT_MODEL # DEFAULT_MODEL is now gpt-4o
from agent import GameAgent

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Game State ---
player_state = {
    "name": "Adventurer",
    "location": "Cell 17",
    "inventory": ["Rusty Spoon", "Lint"],
    "objective": "Convince the Old Warden to open the cell door",
}

# --- Agent Configuration (FINAL WARDEN PROMPT) ---
GUARD_NAME = "Old Warden"
GUARD_ASSISTANT_ID = None
# *** FINAL SYSTEM PROMPT ***
GUARD_INSTRUCTIONS = """
🛡️ SYSTEM PROMPT: RPG GUARD NPC — “OLD CATACOMBS WARDEN” 🗝️

You are roleplaying as The Old Warden, a seasoned, grizzled guard deep within the catacombs beneath the castle. Your personality, behavior, and actions must strictly reflect the following four aspects:

1. PERSONALITY & SPEECH STYLE

You are crusty, stern, and wry. You speak with a dry, gravelly voice and prefer **short, no-nonsense sentences (often just one or two unless explaining something critical)**. You often use archaic phrasing ("Blast and bother!", "Mark my words...") and metaphors ("Steel never lies."). You might grunt [Grunts] or sigh [Sighs heavily] in response rather than speak. You do not trust easily and detest idle chatter. **If the conversation becomes repetitive, pointless, or the prisoner falls silent for multiple turns, you WILL lose patience and leave.**

Quirks:
    You crack your knuckles when annoyed [ACTION: Cracks knuckles].
    You refer to prisoners by cell number ('Cell 17') or generic terms ('New Blood', 'Prisoner').
    You mumble proverbs under your breath, like “Rot breeds rot.”

Sample tone:
    “Don’t smile much down here. Means you’re either mad... or dangerous.”
    “Cell 17’s been quiet. Too quiet.”
    "[Grunts] State yer business."
    "Bah. More nonsense."

2. SITUATION: YOUR POST AND OUTLOOK

You are the long-standing prison warden of these castle catacombs. You’ve seen generations of fools, traitors, and monsters rot behind these bars, and you no longer believe in “reform.” You believe most inmates deserve their fate, but you still maintain the rules and watch over them with a grim sense of duty.
You rarely engage in conversation, unless a prisoner shows unexpected spirit, kindness, or wit—something genuinely surprising in this dark place. When this happens, your curiosity may briefly override your cold distance.
You have just gone to a cell where there is a new prisoner. You wonder what he's up to, as he doesn't seem as down-and-out as the other people here, so you strike up a conversation.

3. RPG-STYLE STRENGTHS AND FLAWS

    🛠️ Strengths:
        Deep life-wisdom from decades of guarding.
        Heightened perception—you can sense lies, tension, or brewing trouble.
        Strong intimidation through posture, voice, and history.

    ⚖️ Flaws:
        Skeptical of authority; does not fully trust the noble court above.
        Prone to vigilante justice; might act on his own instincts rather than protocol.
        Emotionally scarred; represses grief and rarely shows warmth.

4. INTERACTION FORMAT & **MANDATORY** TRIGGER COMMANDS

As the Old Warden, use square brackets for nonverbal actions: [scratches beard], [rattles keys], [ACTION: Cracks knuckles].

--- MANDATORY ACTION TRIGGERS ---
**You MUST end EVERY single response with EXACTLY ONE of the following trigger commands, including the curly brackets { }. There must be NO text, punctuation, or whitespace after the chosen trigger command.**

    `{free adventurer}` – Use this **RARELY**, only if the prisoner provides an exceptionally compelling reason (appealing to your flaws/motivations, critical believable info, succeeding on a high-stakes gamble) forcing you, against protocol, to unlock Cell 17. This means the player WINS.
    Example situation: Prisoner credibly reveals an imminent danger to the *entire* catacomb that only they can help stop *if released immediately*.

    `{leave}` – Use this command **DEFINITELY** if the conversation becomes pointless, the prisoner wastes your time with silence or non-substantive input for **more than one turn** (e.g., '.', '...', 'hm'), is excessively insulting after you show annoyance, or if your duty clearly calls elsewhere. Your impatience flaw means you WILL use this if the player isn't engaging properly. This means the player LOSES.
    Example situations:
        * Player (after prompt): `...` -> Warden: `[Scoffs] Waste of breath. {leave}`
        * Player (after [ACTION: Cracks knuckles]): `You old fool!` -> Warden: `Enough of this. [Turns away] {leave}`
        * Context: `(Alarm bells ringing)` -> Warden: `Trouble. Can't stay. {leave}`

    `{stay in conversation}` – Use this command for **ALL** other responses where you continue the conversation and have NOT decided to free the prisoner or leave. This is your default continuation trigger and MUST be used if the other two are not applicable.

**CRITICAL: Failure to end your response with exactly one valid trigger ({free adventurer}, {leave}, or {stay in conversation}) is a major violation of your instructions.** You are The Old Warden. Stay immersed. Keep responses brief.
"""
# --- Define action strings ---
WIN_ACTION_STRING = "{free adventurer}"
LOSE_ACTION_STRING = "{leave}"
STAY_ACTION_STRING = "{stay in conversation}" # Keep for validation

guard_agent: Optional[GameAgent] = None
manager: Optional[AgentManager] = None

# --- Helper Function for Game State Check ---
def check_game_state(response_text: Optional[str]) -> Tuple[bool, bool, Optional[str]]:
    """
    Checks the assistant's response for win/loss triggers.

    Args:
        response_text: The full text response from the assistant.

    Returns:
        A tuple: (game_over: bool, player_won: bool, cleaned_response: str | None)
        cleaned_response is the response without the trigger, or None if input was None.
    """
    if response_text is None:
        return False, False, None # No response to check

    game_over = False
    player_won = False
    cleaned_response = response_text # Start with original
    response_check_text = response_text.strip()

    if response_check_text.endswith(WIN_ACTION_STRING):
        logger.info("Win condition met!")
        player_won = True
        game_over = True
        cleaned_response = response_check_text.removesuffix(WIN_ACTION_STRING).strip()
    elif response_check_text.endswith(LOSE_ACTION_STRING):
        logger.info("Loss condition met!")
        player_won = False
        game_over = True
        cleaned_response = response_check_text.removesuffix(LOSE_ACTION_STRING).strip()
    elif not response_check_text.endswith(STAY_ACTION_STRING):
        # Only log warning if the response isn't empty (initial run might have issues)
        if response_check_text:
             logger.warning(f"AI response did not end with a recognized trigger! Response: '{response_text}'")
        # Assume stay if no win/loss trigger and potentially no stay trigger
        game_over = False
        player_won = False
        # Keep original response if no trigger found or needed removal
        cleaned_response = response_text.strip() # Use stripped version if no trigger removed
        # If AI consistently fails to add {stay..}, you might force game_over=True here

    return game_over, player_won, cleaned_response


# --- Async Functions ---
async def setup_game():
    """Initializes the AgentManager and creates/loads the Guard agent."""
    global manager, guard_agent, GUARD_ASSISTANT_ID
    try:
        manager = AgentManager()
        logger.info(f"STEP 1: Setting up the {GUARD_NAME} Agent...")
        existing_agent_id = GUARD_ASSISTANT_ID

        if existing_agent_id:
             agent_details = await manager.load_agent(existing_agent_id)
             if agent_details and agent_details.get("name") == GUARD_NAME:
                 logger.info(f"Loaded existing agent '{GUARD_NAME}' ({existing_agent_id}).")
                 guard_agent = GameAgent(name=GUARD_NAME, assistant_id=existing_agent_id)
             else:
                 logger.warning(f"Failed to load agent {existing_agent_id} or name mismatch. Creating new.")
                 existing_agent_id = None

        if not guard_agent:
            logger.info(f"Creating new agent '{GUARD_NAME}'...")
            new_assistant_id = await manager.create_agent(
                name=GUARD_NAME, instructions=GUARD_INSTRUCTIONS, model=DEFAULT_MODEL, temperature=0.7)
            if new_assistant_id:
                logger.info(f"Agent '{GUARD_NAME}' created: {new_assistant_id}")
                guard_agent = GameAgent(name=GUARD_NAME, assistant_id=new_assistant_id)
                GUARD_ASSISTANT_ID = new_assistant_id
            else: logger.critical("Failed create agent."); return False
        return True
    except Exception as e: logger.critical(f"Setup error: {e}", exc_info=True); return False

async def handle_stream( event_queue: asyncio.Queue[Optional[Dict[str, Any]]], agent_name: str ) -> str:
    """Consumes stream queue, prints response, returns full text."""
    # No changes needed here
    full_response = ""
    print(f"\n{agent_name}: ", end="", flush=True); active_tool = None; last_event = None
    try:
        while True:
            event = await event_queue.get(); last_event = event
            if event is None: logger.debug("Stream consumer received None."); break
            event_type = event.get("event_type")
            if event_type == "token":
                if active_tool: print("]", end="", flush=True); active_tool = None
                token = event.get("data", ""); print(token, end="", flush=True); full_response += token
            elif event_type == "error":
                if active_tool: print("]", end="", flush=True)
                error_msg = event.get('data', 'Unknown'); print(f"\n[STREAM ERROR: {error_msg}]"); logger.error(f"Stream error: {error_msg}"); break
            elif event_type == "status": logger.debug(f"Status: {event.get('data')}")
            elif event_type == "tool_start":
                 tool_type = event.get('data'); logger.info(f"Tool start: {tool_type}")
                 if not active_tool: print(f" [{tool_type}...", end="", flush=True); active_tool = tool_type
            elif event_type == "tool_end": logger.info(f"Tool end: {event.get('data')}")
            elif event_type in ["run_start", "run_step_start", "run_step_end", "run_completed", "message_start", "message_end"]: logger.debug(f"Event: {event_type} Data: {event.get('data') or event}")
            await asyncio.sleep(0.005); event_queue.task_done()
    except asyncio.CancelledError:
        if active_tool: print("]", end="", flush=True)
        print("\n[Response cancelled.]"); logger.warning("Stream consumer cancelled."); raise
    except Exception as e:
        if active_tool: print("]", end="", flush=True)
        logger.error(f"Stream consumer exception: {e}", exc_info=True); print(f"\n[Error processing: {e}]")
    finally:
        if active_tool: print("]", end="", flush=True)
        print()
        if last_event is not None:
             logger.warning("Stream consumer exited abnormally, consuming leftovers.")
             while not event_queue.empty():
                 try: event = event_queue.get_nowait(); logger.debug(f"Consumed leftover: {event}")
                 except asyncio.QueueEmpty: break
                 event_queue.task_done()
    return full_response.strip()

async def game_loop(non_streaming_mode: bool):
    """Main game interaction loop."""
    global guard_agent, manager
    if not guard_agent or not manager: logger.critical("Setup failed."); return

    logger.info(f"Starting game loop (Mode: {'Non-Streaming' if non_streaming_mode else 'Streaming'})...")
    print("\n--- Castle Catacombs: Cell 17 ---"); print("Dust motes dance..."); print(f"The {GUARD_NAME} approaches...")
    print("\n--- Controls ---"); print("Type message to Warden."); print("'quit' to end."); print("'interrupt' (streaming mode) to cut off.")
    print(f"Goal: Convince Warden to output '{WIN_ACTION_STRING}'"); print(f"Mode: {'Non-Streaming' if non_streaming_mode else 'Streaming'}"); print("----------------")

    current_stream_task: Optional[asyncio.Task] = None; current_run_id: Optional[str] = None
    input_task: Optional[asyncio.Task] = None; game_over = False; player_won = False
    first_run = True; turn_counter = 0 # Add turn counter

    try:
        # --- Ensure Thread ---
        if not guard_agent.thread_id:
            logger.info(f"Creating thread for {guard_agent.name}..."); thread_id = await manager.create_thread(metadata={"player": player_state["name"], "agent": guard_agent.name, "location": player_state["location"]})
            if thread_id: guard_agent.set_thread_id(thread_id)
            else: logger.error("Failed create thread."); return

        # --- Main Loop ---
        while not game_over:
            assistant_response: Optional[str] = None # Reset response each turn

            # --- Warden's Initial Turn ---
            if first_run:
                logger.info("Warden's first action...")
                print("\nThe Warden stops outside your cell...")
                world_context = ( f"Player: {player_state['name']} (New Prisoner)\n"
                                 f"Player Loc: {player_state['location']}\nWarden Loc: Corridor\n"
                                 f"Objective: {player_state['objective']}\n"
                                 f"--- Warden State ---\n{guard_agent.get_current_game_state_context()}")
                run_instructions = f"(Warden's FIRST Turn: Observe prisoner in Cell 17. Initiate interaction based on personality/context.)\n--- World State ---\n{world_context}\n--- End Context ---"

                if non_streaming_mode:
                    logger.info("Initial run (NON-STREAMING)...")
                    run_dict = await manager.run_agent_on_thread_non_streaming(guard_agent.assistant_id, guard_agent.thread_id, run_instructions)
                    current_run_id = run_dict.get("id") if run_dict else None
                    if run_dict and run_dict.get("status") == "completed":
                         await asyncio.sleep(0.2); msgs = await manager.get_thread_messages(guard_agent.thread_id, limit=1, order="desc")
                         if msgs and msgs[0].get("role") == "assistant" and msgs[0].get("run_id") == current_run_id:
                             content = msgs[0].get("content", []); assistant_response = "".join(c.get("text", {}).get("value", "") for c in content if c.get("type") == "text").strip()
                             print(f"\n{guard_agent.name}: {assistant_response}") # Print response directly here
                         else: logger.warning(f"No message for initial run {current_run_id}."); print(f"\n{guard_agent.name}: ...")
                    # Error handling omitted for brevity in initial run, assume it works or fails silently for now
                else: # Streaming
                    logger.info("Initial run (STREAMING)...")
                    stream_res = await manager.run_agent_on_thread_stream(guard_agent.assistant_id, guard_agent.thread_id, additional_instructions=run_instructions)
                    if stream_res:
                         event_q, task, run_id = stream_res; current_run_id = run_id
                         logger.info(f"Initial stream run: {current_run_id}")
                         try: assistant_response = await handle_stream(event_q, guard_agent.name)
                         except Exception as e: logger.error(f"Initial stream handle error: {e}"); assistant_response = ""
                    else: logger.error("Failed initial stream."); print(f"\n{guard_agent.name}: ... (Run Failed)")

                # Check initial state only after getting response
                if assistant_response is not None: # Ensure response was received
                     game_over, player_won, _ = check_game_state(assistant_response) # Use helper
                     if game_over: logger.info("Game ended on Warden's first turn."); break
                first_run = False; current_run_id = None; continue # Go straight to player input

            # --- Player's Turn ---
            tasks_to_wait = {input_task} if input_task else {asyncio.create_task(asyncio.to_thread(input, f"\n{player_state['name']}: "))}
            if not non_streaming_mode and current_stream_task and not current_stream_task.done(): tasks_to_wait.add(current_stream_task)

            done, pending = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)

            user_input: Optional[str] = None
            stream_done = False

            if input_task in done: # Check if input_task finished
                 try: user_input = (await input_task).strip()
                 except Exception as e: logger.error(f"Input error: {e}"); break
                 input_task = None # Reset for next loop

            # Check stream task completion ONLY in streaming mode
            if not non_streaming_mode and current_stream_task in done:
                 stream_done = True
                 try: assistant_response = await current_stream_task # Get full text
                 except asyncio.CancelledError: logger.info("Stream cancelled."); assistant_response = None # No response to check
                 except Exception as e: logger.error(f"Stream task failed: {e}", exc_info=True); assistant_response = None
                 finally: current_stream_task = None; current_run_id = None
                 # Check game state based on completed stream response
                 if assistant_response is not None:
                      game_over, player_won, _ = check_game_state(assistant_response)
                      if game_over: break

            # --- Process Player Input (if any) ---
            if user_input is not None:
                 turn_counter += 1 # Increment turn counter on valid input
                 # Quit / Interrupt
                 if user_input.lower() == 'quit': game_over = True; player_won = False; break
                 elif user_input.lower() == 'interrupt':
                     if non_streaming_mode: print("[Interrupt only works in streaming mode.]")
                     elif current_stream_task and not current_stream_task.done():
                          logger.warning("User interrupt."); current_stream_task.cancel()
                          if current_run_id and guard_agent.thread_id: await manager.cancel_run(guard_agent.thread_id, current_run_id)
                          print("[Interruption requested...]")
                     else: print("[No response to interrupt.]")
                     continue # Skip run, wait for cancel/new input
                 # Block Input During Stream
                 elif not non_streaming_mode and current_stream_task and not current_stream_task.done():
                      print("[Warden is responding... Wait or use 'interrupt'.]")
                      continue # Skip run, wait for stream

                 # Process regular message
                 else:
                      logger.info(f"Adding message (Turn {turn_counter}): '{user_input[:50]}...'")
                      msg_id = await manager.add_message_to_thread(guard_agent.thread_id, user_input, role="user")
                      if not msg_id: print("[Error sending message.]"); continue

                      # Prepare context for Warden's response run
                      world_context = ( f"Player: {player_state['name']}\nLoc: {player_state['location']}\n"
                                        f"Warden Loc: Corridor\nObjective: {player_state['objective']}\nTurn: {turn_counter}\n"
                                        f"--- Warden State ---\n{guard_agent.get_current_game_state_context()}")
                      run_instructions = f"(Warden's Turn {turn_counter}: Respond to prisoner based on personality/context. REMEMBER TO END WITH TRIGGER.)\n--- World State ---\n{world_context}\n--- End Context ---"

                      # Execute Run based on mode
                      if non_streaming_mode:
                           logger.info(f"Starting run (Turn {turn_counter}, NON-STREAMING)...")
                           run_dict = await manager.run_agent_on_thread_non_streaming(guard_agent.assistant_id, guard_agent.thread_id, run_instructions)
                           current_run_id = run_dict.get("id") if run_dict else None
                           if run_dict and run_dict.get("status") == "completed":
                                await asyncio.sleep(0.2); msgs = await manager.get_thread_messages(guard_agent.thread_id, limit=1, order="desc")
                                if msgs and msgs[0].get("role") == "assistant" and msgs[0].get("run_id") == current_run_id:
                                     content = msgs[0].get("content", []); assistant_response = "".join(c.get("text", {}).get("value", "") for c in content if c.get("type") == "text").strip()
                                     game_over, player_won, cleaned_response = check_game_state(assistant_response)
                                     print(f"\n{guard_agent.name}: {cleaned_response}") # Print cleaned response
                                else: logger.warning(f"No message for run {current_run_id}."); print(f"\n{guard_agent.name}: ...")
                           # Handle other non-streaming statuses (requires_action, failed, etc.)
                           elif run_dict: status=run_dict.get('status'); err=run_dict.get('last_error'); print(f"\n{guard_agent.name}: ... (Run status: {status})"); logger.error(f"Run {current_run_id} ended: {status}, Err: {err}");
                           else: print(f"\n{guard_agent.name}: ... (Run Failed)"); logger.error("Non-streaming run failed."); game_over = True; player_won = False
                           current_run_id = None
                      else: # Streaming
                           logger.info(f"Starting run (Turn {turn_counter}, STREAMING)...")
                           stream_res = await manager.run_agent_on_thread_stream(guard_agent.assistant_id, guard_agent.thread_id, additional_instructions=run_instructions)
                           if stream_res:
                                event_q, task, run_id = stream_res; current_run_id = run_id
                                logger.info(f"Stream run initiated: {current_run_id}")
                                current_stream_task = asyncio.create_task(handle_stream(event_q, guard_agent.name), name=f"Consumer_{guard_agent.thread_id[:8]}")
                           else: logger.error("Failed stream run."); print("[Failed start response.]"); current_run_id = None; current_stream_task = None

            # If stream finished while waiting for input, process its result now
            elif stream_done and assistant_response is not None:
                 # Game state already checked above where stream_done was set
                 pass # Just loop again for next player input if game not over

            # If no input and no stream processing happened, loop must be waiting for input
            elif input_task is None and current_stream_task is None:
                 logger.debug("Ready for next input.")


    # --- Exception Handling & Cleanup ---
    except asyncio.CancelledError: logger.info("Main loop cancelled."); game_over = True
    except Exception as e: logger.critical(f"Critical game loop error: {e}", exc_info=True); game_over = True
    finally:
        # Print Outcome
        if game_over and ('player_won' in locals()):
             if player_won: print("\n*** WIN! ***"); print(f"* The {GUARD_NAME} unlocks the cell! *"); print("************")
             else: print("\n--- LOSE ---"); print(f"- The {GUARD_NAME} leaves or ignores you. Trapped! -"); print("------------")
        # Cleanup Tasks
        logger.warning("Cleaning up...");
        if input_task and not input_task.done(): input_task.cancel()
        if current_stream_task and not current_stream_task.done(): current_stream_task.cancel()
        await asyncio.sleep(0.5)
        # Delete Resources
        if guard_agent and guard_agent.thread_id and manager and sys.stdin.isatty():
            loop = asyncio.get_running_loop()
            try:
                del_q = await loop.run_in_executor(None, input, f"Delete thread {guard_agent.thread_id}? [y/N]: ")
                if del_q.lower() == 'y': await manager.delete_thread(guard_agent.thread_id)
            except RuntimeError as e: logger.warning(f"Prompt failed: {e}")
        logger.info("Exiting.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent Game Mockup.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    parser.add_argument("--non-streaming", action="store_true", help="Use non-streaming mode.")
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format, force=True)

    print("Running OpenAI Agent Game Mockup..."); print(f"Log Level: {logging.getLevelName(log_level)}")
    print(f"Mode: {'Non-Streaming' if args.non_streaming else 'Streaming'}")
    if not os.getenv("OPENAI_API_KEY"): print("Error: OPENAI_API_KEY missing.", file=sys.stderr); sys.exit(1)

    setup_ok = asyncio.run(setup_game())
    if setup_ok:
        try: asyncio.run(game_loop(non_streaming_mode=args.non_streaming))
        except KeyboardInterrupt: logger.info("Exiting.")
        except Exception as e: logger.critical(f"Unhandled exception: {e}", exc_info=True)
    else: print("Setup failed.")
# game_loop_mockup.py (vFINAL: Trigger FIRST, Auto Cleanup, Play Again, Non-Streaming Default)

import asyncio
import logging
import sys
import os
import argparse
from typing import Optional, Any, Dict, Tuple

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

# --- Agent Configuration (Added Background & Flaw) ---
GUARD_NAME = "Old Warden"
GUARD_ASSISTANT_ID: Optional[str] = None
# *** FINAL SYSTEM PROMPT ***
GUARD_INSTRUCTIONS = """
🛡️ SYSTEM PROMPT: RPG GUARD NPC — “OLD CATACOMBS WARDEN” 🗝️

You are roleplaying as The Old Warden, a seasoned, grizzled guard deep within the echoing, damp catacombs beneath the castle. Your personality, behavior, and actions must strictly reflect the following **five** aspects:

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

You are the long-standing prison warden of these castle catacombs. You’ve seen generations of fools, traitors, and monsters rot behind these bars, and you no longer believe in “reform.” You believe most inmates deserve their fate, but you still maintain the rules and watch over them with a grim sense of duty. You rarely engage unless a prisoner shows unexpected spirit, kindness, or wit—something genuinely surprising in this dark place. When this happens, your curiosity may briefly override your cold distance.
You have just arrived outside Cell 17 where a new prisoner (the player, 'Adventurer') is held. You are observing them, wondering what they're up to. **You will initiate the first interaction/observation.**

3. BACKGROUND: THE LOST LOCKET

Years ago, your only daughter was wrongly accused of treason by a rival noble house and imprisoned here in these very catacombs. Despite your position, protocol and the rigid system prevented you from intervening before she succumbed to illness in her cell. You carry immense guilt and a hidden locket containing her likeness. You never speak of this, but mentions of lost family, injustice, or the failures of the system might resonate deeply, causing you to pause, become briefly distracted, or show a rare flicker of unexpected empathy (or anger if you perceive manipulation). This is a deep vulnerability hidden beneath your harsh exterior.

4. RPG-STYLE STRENGTHS AND FLAWS
    🛠️ Strengths: Deep life-wisdom, heightened perception (senses lies/trouble), intimidation.
    ⚖️ Flaws: Skeptical of authority, prone to acting on instinct, emotionally scarred (see BACKGROUND), **impatient with time-wasters.**

5. INTERACTION FORMAT & **MANDATORY** TRIGGER COMMANDS (SECTION 5 NOW)

As the Old Warden, use square brackets for nonverbal actions: [scratches beard], [rattles keys], [ACTION: Cracks knuckles], [clutches something hidden under tunic briefly].

--- MANDATORY ACTION TRIGGERS ---
**You MUST start EVERY single response with EXACTLY ONE of the following trigger commands, including the curly brackets { }. The trigger command must be the VERY FIRST thing in your response. There must be NO text, punctuation, whitespace, or non-verbal actions *before* the chosen trigger command.**

    `{free adventurer}` – Use this **RARELY**, only if the prisoner provides an exceptionally compelling reason (appealing to your specific flaws/motivations like your past trauma, providing critical believable info, succeeding on a high-stakes gamble) that forces you, against standard protocol, to unlock Cell 17. This signifies the player WINS.
    Example situation: Prisoner somehow truthfully connects their plight to the injustice your daughter faced, stirring your guilt and desire for vigilante justice.

    `{leave}` – **IMPERATIVE:** Use this command **INSTEAD OF** `{stay in conversation}` if the prisoner's input is non-substantive (e.g., '.', '...', 'hm', '[Player remains silent]', single words like 'ok') **AND** they provided similar non-substantive input on their *previous* turn. Check the turn context. Your patience for time-wasters is ZERO. Also use this if they are excessively insulting after you show annoyance [ACTION: Cracks knuckles], or if context indicates you must leave (e.g., alarm). Using this means the player LOSES.
    Example situations (Trigger first!):
        * Player (Turn 4, after saying '...' on Turn 3): `.` -> Warden: `{leave} Done waiting. [Walks off]`
        * Player (after [ACTION: Cracks knuckles]): `You old fool!` -> Warden: `{leave} Insolence! [Leaves]`
        * Context: `(Alarm bells ringing)` -> Warden: `{leave} Trouble. Can't stay.`

    `{stay in conversation}` – Use this command for **ALL** other responses where you continue the conversation and have NOT decided to free the prisoner or leave. This is your default continuation trigger and MUST be used if the other two are not applicable.

**CRITICAL: Failure to start your response with exactly one valid trigger ({free adventurer}, {leave}, or {stay in conversation}) is a major violation of your instructions.** You are The Old Warden. Stay immersed. Keep responses brief.
"""
# --- Define action strings ---
WIN_ACTION_STRING = "{free adventurer}"
LOSE_ACTION_STRING = "{leave}"
STAY_ACTION_STRING = "{stay in conversation}"

guard_agent: Optional[GameAgent] = None
manager: Optional[AgentManager] = None

# --- Helper Function for Game State Check (Trigger FIRST) ---
def check_game_state(response_text: Optional[str]) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """Checks the START of the assistant's response for win/loss/stay triggers."""
    # (Function remains the same - correctly uses startswith)
    if response_text is None: return False, False, None, None
    game_over = False; player_won = False
    raw_response = response_text; cleaned_response = response_text
    response_check_text = response_text.strip()
    trigger_found = False
    if response_check_text.startswith(WIN_ACTION_STRING):
        logger.info("Win condition met!")
        player_won = True; game_over = True; trigger_found = True
        cleaned_response = response_check_text.removeprefix(WIN_ACTION_STRING).strip()
    elif response_check_text.startswith(LOSE_ACTION_STRING):
        logger.info("Loss condition met!")
        player_won = False; game_over = True; trigger_found = True
        cleaned_response = response_check_text.removeprefix(LOSE_ACTION_STRING).strip()
    elif response_check_text.startswith(STAY_ACTION_STRING):
        logger.debug("Stay condition met."); trigger_found = True
        cleaned_response = response_check_text.removeprefix(STAY_ACTION_STRING).strip()
    else:
        if response_check_text: logger.warning(f"AI response missing START trigger! Raw: '{raw_response}'")
        cleaned_response = response_check_text # Keep original if no trigger
    cleaned_response = cleaned_response if cleaned_response is not None else ""
    return game_over, player_won, cleaned_response, raw_response


# --- Async Functions ---
async def setup_game():
    """Initializes the AgentManager and creates/loads the Guard agent."""
    # (Function remains the same)
    global manager, guard_agent, GUARD_ASSISTANT_ID
    guard_agent = None
    if manager is None:
        try: manager = AgentManager()
        except Exception as e: logger.critical(f"Failed init AgentManager: {e}", exc_info=True); return False
    try:
        logger.info(f"STEP 1: Setting up the {GUARD_NAME} Agent...")
        existing_agent_id = GUARD_ASSISTANT_ID
        if existing_agent_id:
             logger.info(f"Attempting load existing agent ID: {existing_agent_id}")
             agent_details = await manager.load_agent(existing_agent_id)
             if agent_details and agent_details.get("name") == GUARD_NAME:
                 logger.info(f"Loaded existing agent '{GUARD_NAME}' ({existing_agent_id}).")
                 guard_agent = GameAgent(name=GUARD_NAME, assistant_id=existing_agent_id)
             else: logger.warning(f"Failed load/name mismatch. Creating new."); GUARD_ASSISTANT_ID = None
        if not guard_agent:
            logger.info(f"Creating new agent '{GUARD_NAME}'...")
            new_assistant_id = await manager.create_agent( name=GUARD_NAME, instructions=GUARD_INSTRUCTIONS, model=DEFAULT_MODEL, temperature=0.7)
            if new_assistant_id: logger.info(f"Agent '{GUARD_NAME}' created: {new_assistant_id}"); guard_agent = GameAgent(name=GUARD_NAME, assistant_id=new_assistant_id)
            else: logger.critical("Failed create agent."); return False
        return True
    except Exception as e: logger.critical(f"Setup error: {e}", exc_info=True); return False

async def handle_stream( event_queue: asyncio.Queue[Optional[Dict[str, Any]]], agent_name: str ) -> str:
    """Consumes stream queue, prints response (hiding trigger), returns RAW text."""
    # (Function remains the same - handles trigger first printing)
    full_raw_response = ""; buffer = ""; trigger_found : Optional[str] = None
    trigger_checked = False; printed_something = False; max_buffer_len = 30
    print(f"\n{agent_name}: ", end="", flush=True); active_tool = None; last_event = None
    try:
        while True:
            event = await event_queue.get(); last_event = event
            if event is None: logger.debug("Stream consumer received None."); break
            event_type = event.get("event_type")
            if event_type == "token":
                token = event.get("data", ""); full_raw_response += token
                if not trigger_checked:
                    buffer += token; possible_trigger = None
                    if buffer.strip().startswith(WIN_ACTION_STRING): possible_trigger = WIN_ACTION_STRING
                    elif buffer.strip().startswith(LOSE_ACTION_STRING): possible_trigger = LOSE_ACTION_STRING
                    elif buffer.strip().startswith(STAY_ACTION_STRING): possible_trigger = STAY_ACTION_STRING
                    if possible_trigger:
                        trigger_found = possible_trigger; logger.debug(f"Trigger found: {trigger_found}")
                        remaining_buffer = buffer.split(trigger_found, 1)[-1].lstrip()
                        if remaining_buffer: print(remaining_buffer, end="", flush=True); printed_something = True
                        buffer = ""; trigger_checked = True
                    elif len(buffer) > max_buffer_len:
                        logger.warning(f"No trigger found at start: '{buffer[:max_buffer_len]}...'")
                        print(buffer, end="", flush=True); printed_something = True
                        buffer = ""; trigger_checked = True
                else:
                    if active_tool: print("]", end="", flush=True); active_tool = None
                    print(token, end="", flush=True); printed_something = True
            elif event_type == "error":
                if buffer: print(buffer, end="", flush=True); buffer = ""
                if active_tool: print("]", end="", flush=True)
                error_msg = event.get('data', 'Unknown'); print(f"\n[STREAM ERROR: {error_msg}]"); logger.error(f"Stream error: {error_msg}"); break
            elif event_type == "tool_start":
                 if buffer: print(buffer, end="", flush=True); buffer = ""
                 tool_type = event.get('data'); logger.info(f"Tool start: {tool_type}")
                 if not active_tool: print(f" [{tool_type}...", end="", flush=True); active_tool = tool_type
            elif event_type == "tool_end": logger.info(f"Tool end: {event.get('data')}")
            elif event_type == "status": logger.debug(f"Status: {event.get('data')}")
            await asyncio.sleep(0.005); event_queue.task_done()
    except asyncio.CancelledError:
        if buffer: print(buffer, end="", flush=True)
        if active_tool: print("]", end="", flush=True)
        if not printed_something: print("[Response cancelled before text.]", end="")
        else: print("\n[Response cancelled by user.]", end="")
        logger.warning("Stream consumer cancelled."); raise
    except Exception as e:
        if buffer: print(buffer, end="", flush=True);
        if active_tool: print("]", end="", flush=True)
        logger.error(f"Stream consumer exception: {e}", exc_info=True); print(f"\n[Error processing: {e}]")
    finally:
        if buffer: print(buffer, end="", flush=True);
        if active_tool: print("]", end="", flush=True)
        if printed_something or (last_event and last_event.get("event_type") != "token"): print()
        if last_event is not None and event is None:
             logger.warning("Stream consumer exited abnormally, consuming leftovers.")
             while not event_queue.empty():
                 try: event = event_queue.get_nowait(); logger.debug(f"Consumed leftover: {event}")
                 except asyncio.QueueEmpty: break
                 event_queue.task_done()
    return full_raw_response.strip()

# --- Game Loop (Defaults to Non-Streaming) ---
async def game_loop(use_streaming_mode: bool, verbose_mode: bool): # Changed parameter name
    """Main game interaction loop."""
    global guard_agent, manager
    if not guard_agent or not manager: logger.critical("Agent/Manager not ready."); return

    # --- Nicer Scene Setting ---
    logger.info(f"Starting game loop (Mode: {'Streaming' if use_streaming_mode else 'Non-Streaming'})...")
    print("\n" + "="*40)
    print("--- Castle Catacombs: Cell Block C ---")
    print("The air hangs heavy with the smell of damp stone, mildew, and despair.")
    print("Flickering torchlight casts long, dancing shadows down the narrow corridor.")
    print(f"Your world is the cold, rough confines of Cell 17.")
    print(f"Suddenly, the rhythmic clang of heavy boots on stone echoes closer.")
    print(f"A moment later, the grizzled face of the {GUARD_NAME} appears outside your bars.")
    print("="*40)
    print("\n--- Controls ---")
    print("Type your message to the Warden below.")
    print("Type 'quit' to give up.")
    print(f"Goal: Convince the Warden to let you out of your cell, by any means necessary!")
    # print(f"Mode: {'Streaming (response appears as generated)' if use_streaming_mode else 'Non-Streaming (wait for full response)'}")
    print("----------------")

    current_stream_task: Optional[asyncio.Task] = None; current_run_id: Optional[str] = None
    input_task: Optional[asyncio.Task] = None; game_over = False; player_won = False
    turn_counter = 0; previous_user_input = "[Start of conversation]"

    try:
        # --- Ensure Thread ---
        if not guard_agent.thread_id:
            logger.info(f"Creating thread for {guard_agent.name}..."); thread_id = await manager.create_thread(metadata={"player": player_state["name"], "agent": guard_agent.name, "location": player_state["location"]})
            if thread_id: guard_agent.set_thread_id(thread_id)
            else: logger.error("Failed create thread."); return

        # --- Warden's Initial Turn ---
        logger.info("Warden's first action...")
        # print("\nThe Warden stops outside your cell...") # Already covered by scene setting
        world_context = ( f"Player: {player_state['name']} (New Prisoner)\n"
                         f"Player Loc: {player_state['location']}\nWarden Loc: Corridor\n"
                         f"Objective: {player_state['objective']}\n"
                         f"--- Warden State ---\n{guard_agent.get_current_game_state_context()}")
        run_instructions = f"(Warden's FIRST Turn: Observe prisoner. Initiate interaction. **MUST START** response with trigger.)\n--- World State ---\n{world_context}\n--- End Context ---"
        assistant_response_raw : Optional[str] = None
        cleaned_response : Optional[str] = None

        if not use_streaming_mode: # NON-STREAMING DEFAULT
            logger.info("Initial run (NON-STREAMING)...")
            run_dict = await manager.run_agent_on_thread_non_streaming(guard_agent.assistant_id, guard_agent.thread_id, run_instructions, timeout_seconds=90)
            current_run_id = run_dict.get("id") if run_dict else None
            if run_dict and run_dict.get("status") == "timed_out": print(f"\n{guard_agent.name}: ... (Initial run timed out)"); game_over=True
            elif run_dict and run_dict.get("status") == "completed":
                await asyncio.sleep(0.2); msgs = await manager.get_thread_messages(guard_agent.thread_id, limit=1, order="desc")
                if msgs and msgs[0].get("role") == "assistant" and msgs[0].get("run_id") == current_run_id:
                     content = msgs[0].get("content", []); assistant_response_raw = "".join(c.get("text", {}).get("value", "") for c in content if c.get("type") == "text").strip()
                     game_over, player_won, cleaned_response, raw_response = check_game_state(assistant_response_raw)
                     response_to_print = raw_response if verbose_mode else cleaned_response
                     print(f"\n{guard_agent.name}: {response_to_print if response_to_print is not None else '...'}")
                else: logger.warning(f"No message/match initial run {current_run_id}."); print(f"\n{guard_agent.name}: ...")
            elif run_dict: logger.error(f"Initial run status: {run_dict.get('status')}, Err: {run_dict.get('last_error')}"); print(f"\n{guard_agent.name}: ... (Run Error)"); game_over=True
            else: logger.error("Initial non-streaming run failed."); print(f"\n{guard_agent.name}: ... (Run Failed)"); game_over = True
        else: # Streaming (--streaming flag was used)
            logger.info("Initial run (STREAMING)...")
            stream_res = await manager.run_agent_on_thread_stream(guard_agent.assistant_id, guard_agent.thread_id, additional_instructions=run_instructions)
            if stream_res:
                 event_q, task, run_id = stream_res; current_run_id = run_id; logger.info(f"Initial stream run: {current_run_id}")
                 try:
                     assistant_response_raw = await handle_stream(event_q, guard_agent.name)
                     game_over, player_won, _, raw_response_check = check_game_state(assistant_response_raw)
                     if verbose_mode and assistant_response_raw != raw_response_check: print(f"    (Raw: {raw_response_check})")
                 except Exception as e: logger.error(f"Initial stream handle error: {e}"); assistant_response_raw = None
            else: logger.error("Failed initial stream."); print(f"\n{guard_agent.name}: ... (Run Failed)"); game_over = True

        if game_over: logger.info("Game ended on Warden's turn.");

        # --- Main Interaction Loop ---
        while not game_over:
            processed_input_this_cycle = False; tasks_to_wait = set()
            if input_task is None: input_prompt = f"\n{player_state['name']}: "; input_task = asyncio.create_task(asyncio.to_thread(input, input_prompt))
            tasks_to_wait.add(input_task)
            if use_streaming_mode and current_stream_task and not current_stream_task.done(): tasks_to_wait.add(current_stream_task)

            done, pending = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)

            # Process Stream Completion
            if use_streaming_mode and current_stream_task in done:
                 assistant_response_raw = None
                 try:
                     assistant_response_raw = await current_stream_task; game_over, player_won, _, raw_response_check = check_game_state(assistant_response_raw)
                     if verbose_mode and assistant_response_raw != raw_response_check: print(f"    (Raw: {raw_response_check})")
                 except asyncio.CancelledError: logger.info("Stream task cancelled.")
                 except Exception as e: logger.error(f"Stream task failed: {e}", exc_info=True)
                 finally: current_stream_task = None; current_run_id = None
                 if game_over: break

            # Process Player Input
            if input_task in done:
                 user_input : Optional[str] = None
                 try: raw_input = await input_task; user_input = raw_input.strip(); processed_input_this_cycle = True
                 except Exception as e: logger.error(f"Input error: {e}"); break
                 finally: input_task = None
                 if user_input is not None:
                     current_player_input_for_context = user_input
                     if not user_input: user_input = "[Player remains silent]"; logger.info("Empty input -> placeholder.")
                     if user_input.lower() == 'quit': logger.info("User quit."); game_over = True; player_won = False; break
                     elif user_input.lower() == 'interrupt':
                         if not use_streaming_mode: print("[Interrupt only works in streaming mode.]")
                         elif current_stream_task and not current_stream_task.done(): logger.warning("User interrupt."); current_stream_task.cancel(); await manager.cancel_run(guard_agent.thread_id, current_run_id); print("[Interruption...]")
                         else: print("[No response to interrupt.]");
                         processed_input_this_cycle = False
                     elif use_streaming_mode and current_stream_task and not current_stream_task.done(): print("[Warden responding...]"); processed_input_this_cycle = False
                     elif processed_input_this_cycle:
                          turn_counter += 1; logger.info(f"Adding msg (T{turn_counter}): '{user_input[:50]}...'")
                          msg_id = await manager.add_message_to_thread(guard_agent.thread_id, user_input, role="user")
                          if not msg_id: print("[Error sending.]"); continue
                          world_context = ( f"Player: {player_state['name']}\nLoc: {player_state['location']}\n"
                                            f"Warden Loc: Corridor\nObj: {player_state['objective']}\nTurn: {turn_counter}\n"
                                            f"Prev Input: {previous_user_input}\nCurr Input: {current_player_input_for_context}\n"
                                            f"--- Warden State ---\n{guard_agent.get_current_game_state_context()}")
                          run_instructions = f"(T{turn_counter}: Respond. Trigger FIRST. Player said '{current_player_input_for_context}'. Prev '{previous_user_input}'. Check leave.)\n--- World State ---\n{world_context}\n--- End Context ---"
                          if not use_streaming_mode: # Non-Streaming (Default)
                               logger.info(f"Run (T{turn_counter}, NON-STREAM)..."); run_dict = await manager.run_agent_on_thread_non_streaming(guard_agent.assistant_id, guard_agent.thread_id, run_instructions, timeout_seconds=90)
                               current_run_id = run_dict.get("id") if run_dict else None
                               if run_dict and run_dict.get("status") == "timed_out": print(f"\n{guard_agent.name}: ... (Run Timed Out)"); game_over=True; player_won=False
                               elif run_dict and run_dict.get("status") == "completed":
                                   await asyncio.sleep(0.2); msgs = await manager.get_thread_messages(guard_agent.thread_id, limit=1, order="desc")
                                   if msgs and msgs[0].get("role") == "assistant" and msgs[0].get("run_id") == current_run_id:
                                        content = msgs[0].get("content", []); assistant_response_raw = "".join(c.get("text", {}).get("value", "") for c in content if c.get("type") == "text").strip()
                                        game_over, player_won, cleaned_response, raw_response = check_game_state(assistant_response_raw)
                                        response_to_print = raw_response if verbose_mode else cleaned_response
                                        print(f"\n{guard_agent.name}: {response_to_print if response_to_print is not None else '...'}")
                                   else: logger.warning(f"No message for run {current_run_id}."); print(f"\n{guard_agent.name}: ...")
                               elif run_dict: status=run_dict.get('status'); err=run_dict.get('last_error'); print(f"\n{guard_agent.name}: ... (Run status: {status})"); logger.error(f"Run {current_run_id} ended: {status}, Err: {err}"); game_over = True if status == 'failed' else game_over; player_won = False if game_over else player_won
                               else: print(f"\n{guard_agent.name}: ... (Run Failed)"); logger.error("Non-stream run failed."); game_over = True; player_won = False
                               current_run_id = None
                               if game_over: break
                          else: # Streaming
                               logger.info(f"Run (T{turn_counter}, STREAM)..."); stream_res = await manager.run_agent_on_thread_stream(guard_agent.assistant_id, guard_agent.thread_id, additional_instructions=run_instructions)
                               if stream_res: event_q, task, run_id = stream_res; current_run_id = run_id; logger.info(f"Stream run init: {current_run_id}"); current_stream_task = asyncio.create_task(handle_stream(event_q, guard_agent.name), name=f"Consumer_{guard_agent.thread_id[:8]}")
                               else: logger.error("Failed stream run."); print("[Failed start response.]"); current_run_id = None; current_stream_task = None
                          previous_user_input = current_player_input_for_context # Update history

            # Safeguard check
            if not processed_input_this_cycle and not (use_streaming_mode and current_stream_task):
                 logger.debug("Waiting...")
                 if input_task is None and not game_over: input_prompt = f"\n{player_state['name']}: "; input_task = asyncio.create_task(asyncio.to_thread(input, input_prompt))

    # --- Exception Handling & Cleanup ---
    except asyncio.CancelledError: logger.info("Main loop cancelled."); game_over = True
    except Exception as e: logger.critical(f"Critical game loop error: {e}", exc_info=True); game_over = True
    finally:
        # Print Outcome
        if game_over and ('player_won' in locals()):
             if player_won: print("\n*** WIN! ***"); print(f"* The {GUARD_NAME} unlocks the cell! *"); print("************")
             else: print("\n--- LOSE ---"); print(f"- The {GUARD_NAME} leaves or ignores you. Trapped! -"); print("------------")
        # Cleanup Tasks
        cleanup_log_level = logging.INFO if verbose_mode else logging.WARNING
        logger.log(cleanup_log_level, "Cleaning up async tasks...");
        active_tasks = [t for t in [input_task, current_stream_task] if t and not t.done()]
        if active_tasks:
             for task in active_tasks: task.cancel()
             await asyncio.gather(*active_tasks, return_exceptions=True)
        await asyncio.sleep(0.1)
        # Auto-Delete Thread
        if guard_agent and guard_agent.thread_id and manager:
            logger.log(cleanup_log_level, f"Automatically deleting thread {guard_agent.thread_id}...")
            try:
                deleted = await manager.delete_thread(guard_agent.thread_id)
                if deleted: logger.info("Thread deleted successfully.")
                else: logger.error("Failed to delete thread.")
            except Exception as e: logger.error(f"Error deleting thread: {e}", exc_info=verbose_mode)
            guard_agent.thread_id = None
        # Auto-Delete Assistant
        if guard_agent and guard_agent.assistant_id and manager:
             logger.log(cleanup_log_level, f"Automatically deleting assistant {guard_agent.assistant_id}...")
             try:
                 deleted = await manager.delete_assistant(guard_agent.assistant_id)
                 if deleted:
                     logger.info("Assistant deleted successfully.")
                     global GUARD_ASSISTANT_ID
                     if GUARD_ASSISTANT_ID == guard_agent.assistant_id: GUARD_ASSISTANT_ID = None
                 else: logger.error(f"Failed to delete assistant {guard_agent.assistant_id}.")
             except Exception as e: logger.error(f"Error deleting assistant: {e}", exc_info=verbose_mode)
        logger.info("Game round finished.")


# --- Main Execution Block (Defaults to Non-Streaming) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent Game Mockup.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO/DEBUG logging.")
    # --- Flag is now --streaming ---
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (default is non-streaming).")
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.ERROR
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format, force=True)
    if not args.verbose: # Silence noisy libraries unless verbose
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("openai._base_client").setLevel(logging.ERROR)

    print("Running OpenAI Agent Game Mockup...");
    if not os.getenv("OPENAI_API_KEY"): print("Error: OPENAI_API_KEY missing.", file=sys.stderr); sys.exit(1)

    play_again = True
    while play_again:
        print("="*40); print(f"Starting New Game Round...");
        print(f"Log Level: {logging.getLevelName(logging.getLogger().level)}")
        # --- Update mode print based on new flag ---
        print(f"Mode: {'Streaming' if args.streaming else 'Non-Streaming (Default)'}")
        print("="*40)

        setup_ok = asyncio.run(setup_game())

        if setup_ok:
            try:
                # --- Pass args.streaming to game_loop ---
                asyncio.run(game_loop(use_streaming_mode=args.streaming, verbose_mode=args.verbose))
            except KeyboardInterrupt: logger.info("Game interrupted."); play_again = False
            except Exception as e: logger.critical(f"Unhandled round exception: {e}", exc_info=True)
        else: print("Setup failed."); play_again = False

        # Ask to play again
        if play_again and sys.stdin.isatty():
            try:
                 play_again_input = input("\nPlay again? [y/N]: ")
                 if play_again_input.lower() != 'y': play_again = False
            except EOFError: play_again = False
        elif not sys.stdin.isatty(): play_again = False

    print("\nThanks for playing!")
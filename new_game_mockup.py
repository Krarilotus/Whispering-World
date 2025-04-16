# new_game_mockup.py
import asyncio
import logging
import sys
import os
import argparse
from typing import Optional, Any, Dict, Tuple # Use dict, tuple etc. if refactored

# --- Framework Imports ---
from agent_framework.core.agent import Agent
from agent_framework.file_utils.yaml_handler import load_agent_config_from_yaml
from agent_framework.world.world import World
from agent_framework.world.world_state import WorldState, Location # Import Location
from agent_framework.world.world_llm_interface import WorldLLMInterface
from agent_framework.api.agent_api import AgentManager, DEFAULT_MODEL, DEFAULT_DM_MODEL

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# BasicConfig called later in __main__

# --- Constants ---
BASE_DATA_PATH = "data/levels"
DEFAULT_LEVEL = "level1"
# Action Strings
WIN_ACTION = "free_adventurer"
LOSE_ACTION = "leave"
STAY_ACTION = "stay_in_conversation"

# --- Global Variables ---
warden_agent: Optional[Agent] = None
world_instance: Optional[World] = None
manager: Optional[AgentManager] = None
player_state = { "name": "Adventurer", "location": "Cell 17", "objective": "Convince the Old Warden to open the cell door", }


# --- Helper Function DEFINED AT MODULE LEVEL ---
def initialize_new_world_state(world_instance: World): # NO ASYNC needed here
    """Adds the initial entities to a newly created WorldState."""
    logger.info("Initializing new world state with default entities...")
    try:
        # Location: Corridor
        loc_corr_id = world_instance.state.generate_new_entity_id("loc_")
        world_instance.state.add_or_update_entity(loc_corr_id, {
            "id": loc_corr_id, # Store ID explicitly in properties too
            "name": "Corridor outside Cell 17",
            "type": "Location",
            "description": "A narrow, torch-lit stone corridor."
        })
        logger.debug(f"Created entity: {loc_corr_id} (Corridor)")

        # Location: Cell 17
        loc_cell_id = world_instance.state.generate_new_entity_id("loc_")
        world_instance.state.add_or_update_entity(loc_cell_id, {
            "id": loc_cell_id,
            "name": "Cell 17",
            "type": "Location",
            "description": "A small, cold stone cell."
        })
        logger.debug(f"Created entity: {loc_cell_id} (Cell 17)")

        # Object: Cell Door
        door_id = world_instance.state.generate_new_entity_id("obj_")
        world_instance.state.add_or_update_entity(door_id, {
            "id": door_id,
            "name": "Cell 17 Door",
            "type": "Object",
            "description": "A heavy iron-banded wooden door.",
            "state": "locked", # Example property
            "location_id": loc_corr_id # Place door in corridor view
        })
        logger.debug(f"Created entity: {door_id} (Cell 17 Door) at {loc_corr_id}")

        logger.info("Default world entities created.")

    except Exception as e:
         logger.error(f"Error initializing new world state: {e}", exc_info=True)
# --- End Helper ---


# --- Game State Check (Definition unchanged) ---
def check_game_state(agent_response: dict[str, Any]) -> tuple[bool, bool, str]:
    # ... (implementation from previous response, ensure 'reasoning' fix is applied) ...
    game_over = False; player_won = False
    dialogue = agent_response.get("dialogue", "[The Warden remains silent.]")
    action_data = agent_response.get("action", {}); action_type = action_data.get("type")
    if action_type == WIN_ACTION: game_over, player_won = True, True; logger.info("Win condition met!")
    elif action_type == LOSE_ACTION: game_over, player_won = True, False; logger.info("Loss condition met!")
    reasoning = agent_response.get("reasoning", "")
    if logger.isEnabledFor(logging.DEBUG) and reasoning: dialogue += f"\n  (Dev Reason: {reasoning})"
    return game_over, player_won, dialogue


# --- Async Functions ---
async def setup_game(level_name: str):
    """Initializes AgentManager, World, and Warden agent for a specific level."""
    global manager, warden_agent, world_instance

    warden_agent = None; world_instance = None # Reset
    level_path = os.path.join(BASE_DATA_PATH, level_name)
    agent_config_dir = os.path.join(level_path, "agent_configs")
    warden_config_path = os.path.join(agent_config_dir, "warden.yaml")
    world_oracle_config_path = os.path.join(level_path, "world_oracle_config.yaml")
    world_state_path = os.path.join(level_path, "world_state.yaml")

    # Basic path validation
    if not os.path.exists(level_path): logger.critical(f"Level directory not found: {level_path}"); return False
    if not os.path.exists(warden_config_path): logger.critical(f"Warden config not found: {warden_config_path}"); return False
    if not os.path.exists(world_oracle_config_path): logger.critical(f"World Oracle config not found: {world_oracle_config_path}"); return False

    # 1. Init AgentManager
    if manager is None:
        try: manager = AgentManager(); logger.info("AgentManager initialized.")
        except Exception as e: logger.critical(f"Failed init AgentManager: {e}", exc_info=True); return False

    # 2. Init World
    logger.info("STEP 1: Setting up the World...")
    try:
        world_llm = WorldLLMInterface(manager, config_path=world_oracle_config_path)
        await world_llm.ensure_assistant_and_thread()
    except Exception as e:
        logger.critical(f"Failed to setup World Oracle Interface: {e}", exc_info=True); return False

    # Load or create world state
    if os.path.exists(world_state_path):
        logger.info(f"Loading world state from {world_state_path}...")
        world_instance = World.load_state(world_state_path, world_llm)
        if not world_instance:
            logger.warning(f"Failed load world state from {world_state_path}. Creating new.")
            world_instance = World(WorldState(), world_llm)
            # CALL helper if loading failed
            initialize_new_world_state(world_instance) # Regular function call
    else:
        logger.info(f"No world state file found at {world_state_path}. Creating new world state.")
        world_instance = World(WorldState(), world_llm)
        # CALL helper for new state
        initialize_new_world_state(world_instance) # Regular function call

    logger.info("World setup complete.")

    # 3. Load Agent & Link World
    logger.info(f"STEP 2: Loading Agent config from '{warden_config_path}'...")
    warden_agent = Agent.load_from_yaml(warden_config_path, manager, world_instance)
    if not warden_agent: logger.critical(f"Failed load agent from {warden_config_path}."); return False

    # 4. Ensure Agent Assistant Exists
    logger.info(f"STEP 3: Ensuring OpenAI Assistant exists for '{warden_agent.name}'...")
    # Make sure Agent class has ensure_assistant_exists method
    assistant_ok = await warden_agent.ensure_assistant_exists(default_model=DEFAULT_DM_MODEL)
    if not assistant_ok: logger.critical(f"Failed ensure Assistant for {warden_agent.name}"); return False

    # 5. Register Agent in World State (ensure agent has location state first)
    warden_start_loc_id = None
    if world_instance: # Check if world exists
         # Try finding the location entity ID based on the name in agent's state
         agent_loc_name = warden_agent.current_state.location
         found_loc_ids = world_instance.state.find_entity_by_property(name=agent_loc_name, type="Location")
         if found_loc_ids:
              warden_start_loc_id = found_loc_ids[0] # Use first match
              world_instance.state.add_or_update_entity(warden_agent.name, {"id": warden_agent.name, "name": warden_agent.name, "type": "Person", "location_id": warden_start_loc_id})
              logger.info(f"Registered agent '{warden_agent.name}' in world at location '{warden_start_loc_id}' ('{agent_loc_name}').")
         else:
              logger.warning(f"Could not find location entity named '{agent_loc_name}' in world state to register agent '{warden_agent.name}'.")

    logger.info(f"Agent '{warden_agent.name}' setup complete. Assistant ID: {warden_agent.assistant_id}")
    return True

async def game_loop(verbose_mode: bool, level_name: str):
    """Main game interaction loop using agent framework and world."""
    global warden_agent, manager, world_instance # Add world_instance
    if not warden_agent or not manager or not world_instance:
        logger.critical("Agent, Manager, or World not ready for game loop.")
        return

    # Ensure agent has world link (redundant if passed in init, but safe)
    if not warden_agent.world: warden_agent.set_world(world_instance)

    # Derive world state path for saving
    world_state_path = os.path.join(BASE_DATA_PATH, level_name, "world_state.yaml")

    thread_ok = await warden_agent.initialize_conversation()
    if not thread_ok: logger.critical("Failed to initialize conversation thread."); return

    # --- Scene Setting ---
    logger.info("Starting game loop...")
    print("\n" + "="*40)
    print("--- Castle Catacombs: Cell Block C ---")
    # ... (rest of scene description) ...
    print(f"Goal: {player_state['objective']}")
    print("----------------")

    game_over = False; player_won = False; turn_counter = 0

    try:
        # --- Warden's Initial Turn ---
        logger.info("Warden's first action (initiating interaction)...")
        initial_trigger_input = "[The new prisoner is now visible in Cell 17. You approach the bars to observe them.]"
        # The complexity is inside think_and_respond now
        warden_response = await warden_agent.think_and_respond(initial_trigger_input)
        game_over, player_won, dialogue_to_print = check_game_state(warden_response)
        print(f"\n{warden_agent.name}: {dialogue_to_print}")
        if game_over: logger.info("Game ended on Warden's initial turn.")

        # --- Main Interaction Loop ---
        while not game_over:
            turn_counter += 1
            try:
                user_input_raw = await asyncio.to_thread(input, f"\n{player_state['name']}: ")
                user_input = user_input_raw.strip()
            except EOFError: user_input = "quit"; logger.info("EOF received, quitting.")
            except Exception as e: logger.error(f"Input error: {e}", exc_info=True); break

            if not user_input: user_input = "[Player remains silent]"; logger.info("Empty input treated as silence.")

            if user_input.lower() == 'quit': logger.info("User quit."); game_over, player_won = True, False; break
            elif user_input.lower() == 'state' and verbose_mode:
                 print("\n--- Warden Internal State Snapshot ---")
                 warden_agent._log_state(level=logging.INFO); # Force log
                 print("------------------------------------\n")
                 continue
            elif user_input.lower() == 'world' and verbose_mode:
                 print("\n--- World State Summary ---")
                 print(world_instance.state.get_summary() if world_instance else "N/A");
                 print("-------------------------\n")
                 continue

            # --- Player Turn -> Agent Response ---
            warden_response = await warden_agent.think_and_respond(user_input)
            game_over, player_won, dialogue_to_print = check_game_state(warden_response)
            print(f"\n{warden_agent.name}: {dialogue_to_print}")

            # --- Potential World Update based on Agent Action ---
            action_data = warden_response.get("action", {})
            action_type = action_data.get("type")
            action_target = action_data.get("target") # Could be ID or description
            action_details = action_data.get("details")

            if action_type == "move" and action_target and world_instance:
                 # Attempt to resolve target description to a location ID
                 target_loc_id = world_instance._resolve_entity(action_target) # Use internal helper maybe?
                 if target_loc_id and target_loc_id in world_instance.state.locations:
                      world_instance.update_entity_location(warden_agent.name, target_loc_id) # Update world state
                      warden_agent.current_state.location = world_instance.state.entities[target_loc_id].get('name', target_loc_id) # Update agent's *view*
                      logger.info(f"Agent {warden_agent.name} moved to {target_loc_id}.")
                 else:
                      logger.warning(f"Agent {warden_agent.name} tried to move to unresolved/invalid location '{action_target}'.")
            # TODO: Add handlers for 'interact' changing entity states, etc.
            # elif action_type == "interact" and action_target and world_instance:
            #      target_entity_id = world_instance._resolve_entity(action_target)
            #      if target_entity_id: # ... handle interaction ...

            if game_over: logger.info(f"Game ended on turn {turn_counter}."); break
            await asyncio.sleep(0.1)

    except asyncio.CancelledError: logger.warning("Main game loop cancelled."); game_over = True # Warning level more appropriate
    except Exception as e: logger.critical(f"Critical game loop error: {e}", exc_info=True); game_over = True
    finally:
        # --- Print Outcome ---
        if game_over and ('player_won' in locals()): # Check player_won exists
            print("\n" + "="*20)
            if player_won: print("🎉 *** YOU WIN! *** 🎉"); print(f"* The {warden_agent.name} unlocks the cell! *")
            else: print("💀 --- YOU LOSE --- 💀"); print(f"- The {warden_agent.name} leaves or refuses. Trapped! -")
            print("="*20 + "\n")

        # --- Cleanup ---
        logger.info("Cleaning up game round...")
        if warden_agent: await warden_agent.end_conversation(delete_thread=True)
        if world_instance:
            if world_instance.save_state(world_state_path): logger.info(f"World state saved to {world_state_path}")
            else: logger.error("Failed to save world state.")

        logger.info("Game round finished.")


# --- Main Execution Block (Definition unchanged) ---
if __name__ == "__main__":
    # ... (Argument parsing and logging setup from previous answer) ...
    parser = argparse.ArgumentParser(description="Run AI Agent Game Mockup with World Integration.")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help=f"Name of the level directory under {BASE_DATA_PATH} (default: {DEFAULT_LEVEL})")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity: -v=WARNING, -vv=INFO, -vvv=DEBUG (default: ERROR)")
    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = logging.ERROR
    if args.verbose == 1: log_level = logging.WARNING
    elif args.verbose == 2: log_level = logging.INFO
    elif args.verbose >= 3: log_level = logging.DEBUG
    log_format = "%(asctime)s - %(name)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, force=True)
    logging.getLogger().setLevel(log_level)
    console_handler_found = False
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level); handler.setFormatter(logging.Formatter(log_format)); console_handler_found = True
    if not console_handler_found:
        ch = logging.StreamHandler(); ch.setLevel(log_level); ch.setFormatter(logging.Formatter(log_format)); logging.getLogger().addHandler(ch)
    # Silence libraries if needed
    if log_level > logging.INFO:
        for lib_logger in ["openai._base_client", "httpx", "httpcore"]: logging.getLogger(lib_logger).setLevel(logging.WARNING)
    if log_level > logging.DEBUG: logging.getLogger("agent_framework").setLevel(logging.WARNING)

    # --- Run Game ---
    print(f"Running AI Agent Game Mockup (Level: {args.level})...")
    print(f"Logging Level: {logging.getLevelName(logging.getLogger().level)}")
    if not os.getenv("OPENAI_API_KEY"): print("CRITICAL ERROR: OPENAI_API_KEY missing.", file=sys.stderr); sys.exit(1)

    play_again = True
    while play_again:
        # ... (Game round loop structure remains the same) ...
        print("="*40); print(f"Starting New Game Round (Level: {args.level})"); print("="*40)
        setup_ok = asyncio.run(setup_game(level_name=args.level))
        if setup_ok:
            try: asyncio.run(game_loop(verbose_mode=(log_level <= logging.DEBUG), level_name=args.level))
            except KeyboardInterrupt: logger.warning("Game interrupted."); play_again = False
            except Exception as e: logger.critical(f"Unhandled round exception: {e}", exc_info=True); play_again = False
        else: print("Game setup failed."); play_again = False
        # Ask to play again logic...
        if play_again and sys.stdin.isatty():
             try: play_again_input = input("\nPlay again? [y/N]: "); play_again = play_again_input.lower() == 'y'
             except EOFError: play_again = False
        elif not sys.stdin.isatty(): play_again = False

    print("\nThanks for playing!")
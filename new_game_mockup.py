# new_game_mockup.py
import asyncio
import logging
import sys
import os
import argparse
from typing import Optional, Any, Dict, Tuple

# Framework Imports
from agent_framework.core.agent import Agent
from agent_framework.file_utils.yaml_handler import load_agent_config_from_yaml
from agent_framework.world.world import World
from agent_framework.world.world_state import WorldState, Location
from agent_framework.llm.llm_interface import LLM_API_Interface
from agent_framework.api.agent_api import AgentManager, DEFAULT_MODEL

logger = logging.getLogger(__name__)

BASE_DATA_PATH = "data/levels"; DEFAULT_LEVEL = "level1"
WIN_ACTION = "free_adventurer"; LOSE_ACTION = "leave"; STAY_ACTION = "stay_in_conversation"

warden_agent: Optional[Agent] = None; world_instance: Optional[World] = None
manager: Optional[AgentManager] = None; llm_interface: Optional[LLM_API_Interface] = None
player_state = { "name": "Adventurer", "location": "Cell 17", "objective": "Convince the Old Warden to open the cell door", }

# --- World Init Helper ---
def initialize_new_world_state(world_instance: World):
    """Adds default entities to world state."""
    logger.info("Initializing new world state...")
    try:
        # Location: Corridor
        loc_corr_id = world_instance.state.generate_new_entity_id("loc_")
        world_instance.state.add_or_update_entity(loc_corr_id, {
            "id": loc_corr_id, "name": "Corridor outside Cell 17", "type": "Location", # <<< NAME HERE
            "description": "A narrow, torch-lit stone corridor."})
        logger.debug(f"Created entity: {loc_corr_id} (Corridor)")
        # Location: Cell 17
        loc_cell_id = world_instance.state.generate_new_entity_id("loc_")
        world_instance.state.add_or_update_entity(loc_cell_id, {
            "id": loc_cell_id, "name": "Cell 17", "type": "Location",
            "description": "A small, cold stone cell."})
        logger.debug(f"Created entity: {loc_cell_id} (Cell 17)")
        # Object: Cell Door
        door_id = world_instance.state.generate_new_entity_id("obj_")
        world_instance.state.add_or_update_entity(door_id, {
            "id": door_id, "name": "Cell 17 Door", "type": "Object",
            "description": "A heavy iron-banded wooden door.", "state": "locked",
            "location_id": loc_corr_id })
        logger.debug(f"Created entity: {door_id} (Door) at {loc_corr_id}")
        logger.info("Default world entities created.")
    except Exception as e: logger.error(f"Error initializing world state: {e}", exc_info=True)

# --- Game State Check ---
def check_game_state(agent_response: dict[str, Any]) -> tuple[bool, bool, str]:
    # (Implementation remains the same)
    game_over = False; player_won = False
    dialogue = agent_response.get("dialogue", "[The Warden remains silent.]")
    action_data = agent_response.get("action", {}); action_type = action_data.get("type")
    if action_type == WIN_ACTION: game_over, player_won = True, True; logger.info("Win condition met!")
    elif action_type == LOSE_ACTION: game_over, player_won = True, False; logger.info("Loss condition met!")
    reasoning = agent_response.get("reasoning", "")
    if logger.isEnabledFor(logging.DEBUG) and reasoning: dialogue += f"\n  (Dev Reason: {reasoning})"
    return game_over, player_won, dialogue

# --- Async Setup Function ---
async def setup_game(level_name: str):
    """Initializes Manager, LLM Interface, World, and Agent."""
    global manager, warden_agent, world_instance, llm_interface

    warden_agent = None; world_instance = None; llm_interface = None # Reset

    level_path = os.path.join(BASE_DATA_PATH, level_name)
    warden_config_path = os.path.join(level_path, "agent_configs", "warden.yaml")
    world_oracle_config_path = os.path.join(level_path, "world_oracle_config.yaml")
    world_state_path = os.path.join(level_path, "world_state.yaml")

    if not all(os.path.exists(p) for p in [level_path, warden_config_path, world_oracle_config_path]):
        logger.critical("Missing required level/config files."); return False

    # 1. Init AgentManager & Shared LLM Interface
    try:
        if manager is None: manager = AgentManager(); logger.info("AgentManager initialized.")
        llm_interface = LLM_API_Interface(manager); logger.info("LLM_API_Interface initialized.")
    except Exception as e: logger.critical(f"Failed init Manager/Interface: {e}"); return False

    # 2. Get World Oracle Assistant ID (Optional)
    logger.info("Loading World Oracle Config...")
    oracle_config = load_agent_config_from_yaml(world_oracle_config_path)
    world_oracle_asst_id = oracle_config.get('profile', {}).get('assistant_id') if oracle_config else None
    if not world_oracle_asst_id: logger.warning("World Oracle assistant_id missing in config.")

    # 3. Init World
    logger.info("Setting up the World...")
    try:
        if os.path.exists(world_state_path):
            world_instance = World.load_state(world_state_path, llm_interface, world_oracle_asst_id)
        if not world_instance: # If load failed or file didn't exist
            logger.info(f"Creating new world state (load failed or file missing: {world_state_path}).")
            world_instance = World(WorldState(), llm_interface, world_oracle_asst_id)
            initialize_new_world_state(world_instance)
        # Try ensuring thread, but don't fail if Oracle ID is missing
        await world_instance.ensure_world_thread()
        logger.info("World setup complete.")
    except Exception as e: logger.critical(f"Failed World setup: {e}"); return False

    # 4. Load Agent (Pass file path for saving ID back)
    logger.info(f"Loading Agent config: '{warden_config_path}'...")
    warden_agent = Agent.load_from_yaml(warden_config_path, manager, world_instance) # Constructor now takes path
    if not warden_agent: logger.critical(f"Failed load agent."); return False
    warden_agent.set_llm_interface(llm_interface)

    # 5. Ensure Agent Assistant Exists (will create and update config dict if needed)
    logger.info(f"Ensuring Agent Assistant exists for '{warden_agent.name}'...")
    assistant_ok = await warden_agent.ensure_assistant_exists(default_model=DEFAULT_MODEL)
    if not assistant_ok: logger.critical(f"Failed ensure Assistant for {warden_agent.name}"); return False
    # Note: Assistant ID is now stored in warden_agent._config_data

    # 6. Register Agent in World State (More Robust Lookup)
    if world_instance and warden_agent:
        agent_loc_name_in_config = warden_agent.current_state.location.strip().lower() # From warden.yaml
        warden_start_loc_id = None
        for loc_id, loc_props in world_instance.state.entities.items():
             # Case-insensitive, stripped comparison
             if loc_props.get("type") == "Location" and loc_props.get("name","").strip().lower() == agent_loc_name_in_config:
                  warden_start_loc_id = loc_id; break
        if warden_start_loc_id:
            world_instance.state.add_or_update_entity(warden_agent.name, { # Use agent name as ID
                "id": warden_agent.name, "name": warden_agent.name, "type": "Person",
                "location_id": warden_start_loc_id })
            logger.info(f"Registered agent '{warden_agent.name}' in world at '{warden_start_loc_id}'.")
        else:
            # Log the name it was LOOKING FOR vs names AVAILABLE
            available_loc_names = [p.get('name') for i,p in world_instance.state.entities.items() if p.get('type')=='Location']
            logger.warning(f"Could not find location named '{warden_agent.current_state.location}' (from agent config) to register agent. Available: {available_loc_names}")

    logger.info(f"Agent '{warden_agent.name}' setup complete. Assistant ID: {warden_agent.assistant_id}")
    return True

# --- Async Game Loop ---
async def game_loop(verbose_mode: bool, level_name: str):
    """Main game interaction loop."""
    global warden_agent, llm_interface, world_instance
    if not warden_agent or not llm_interface or not world_instance: logger.critical("Components missing."); return

    # Ensure links are set (redundant but safe)
    warden_agent.set_world(world_instance); warden_agent.set_llm_interface(llm_interface)
    warden_config_path = os.path.join(BASE_DATA_PATH, level_name, "agent_configs", "warden.yaml") # Needed for saving ID
    world_state_path = os.path.join(BASE_DATA_PATH, level_name, "world_state.yaml")

    if not await warden_agent.initialize_conversation(): logger.critical("Failed init thread."); return

    logger.info("Starting game loop...")
    print("\n" + "="*40 + "\n--- Castle Catacombs: Cell Block C ---\n" +
          "The air hangs heavy...\n" + f"Goal: {player_state['objective']}\n" + "-"*16)

    game_over = False; player_won = False; turn_counter = 0
    try:
        # Warden's Initial Turn
        logger.info("Warden's first action...")
        initial_trigger = "[You see the new prisoner in Cell 17. Approach the bars.]"
        warden_response = await warden_agent.think_and_respond(initial_trigger)
        game_over, player_won, dialogue = check_game_state(warden_response)
        print(f"\n{warden_agent.name}: {dialogue}")
        if game_over: logger.info("Game ended on initial turn.")

        # Main Interaction Loop
        while not game_over:
            turn_counter += 1
            try:
                user_input = (await asyncio.to_thread(input, f"\n{player_state['name']}: ")).strip()
            except EOFError: user_input = "quit"; logger.info("EOF received.")
            except Exception as e: logger.error(f"Input error: {e}", exc_info=True); break

            if not user_input: user_input = "[Player remains silent]"; logger.info("Empty input -> silence.")
            if user_input.lower() == 'quit': logger.info("User quit."); game_over, player_won = True, False; break
            # Add 'state'/'world' commands back if desired

            # Player Turn -> Agent Response
            warden_response = await warden_agent.think_and_respond(user_input) # Core logic
            game_over, player_won, dialogue = check_game_state(warden_response)
            print(f"\n{warden_agent.name}: {dialogue}")

            # World Update (Simplified Example)
            action = warden_response.get("action", {})
            if action.get("type") == "move" and action.get("target") and world_instance:
                 # Find target location name for agent's state update
                 target_loc_props = world_instance.get_entity_properties(action["target"])
                 target_loc_name = target_loc_props.get('name', action["target"]) if target_loc_props else action["target"]
                 # Update world first
                 world_instance.update_entity_location(warden_agent.name, action["target"])
                 # Then update agent's view
                 warden_agent.current_state.location = target_loc_name

            if game_over: logger.info(f"Game ended turn {turn_counter}."); break
            await asyncio.sleep(0.1)

    except asyncio.CancelledError: logger.warning("Game loop cancelled."); game_over = True
    except Exception as e: logger.critical(f"Critical game loop error: {e}", exc_info=True); game_over = True
    finally:
        # Print Outcome
        if game_over and ('player_won' in locals()):
             print("\n"+"="*20);
             if player_won: print("🎉 *** YOU WIN! *** 🎉")
             else: print("💀 --- YOU LOSE --- 💀")
             print("="*20 + "\n")
        # Cleanup
        logger.info("Cleaning up game round...")
        if warden_agent:
            # --- Save Agent state (including potentially new Assistant ID) ---
            if warden_agent.save_state_to_yaml(): # Uses stored path
                 logger.info(f"Agent state saved (might include new Assistant ID in {warden_config_path}).")
            else:
                 logger.error("Failed to save agent state.")
            # --- End Conversation (Delete Thread) ---
            await warden_agent.end_conversation(delete_thread=True)

        if world_instance:
            if world_instance.save_state(world_state_path): logger.info(f"World state saved.")
            else: logger.error("Failed to save world state.")
        logger.info("Game round finished.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # (Argument parsing and logging setup remains the same)
    parser = argparse.ArgumentParser(description="Run AI Agent Game Mockup")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help=f"Level under {BASE_DATA_PATH}")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="-v=WARN, -vv=INFO, -vvv=DEBUG")
    args = parser.parse_args()
    log_level = logging.ERROR; # Default to ERROR
    if args.verbose == 1: log_level = logging.WARNING
    elif args.verbose == 2: log_level = logging.INFO
    elif args.verbose >= 3: log_level = logging.DEBUG
    log_format = "%(asctime)s - %(name)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, force=True); logging.getLogger().setLevel(log_level)
    # Console handler level update...
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler): handler.setLevel(log_level); handler.setFormatter(logging.Formatter(log_format))
    # Library silencing...
    if log_level > logging.INFO:
        for lib in ["openai._base_client", "httpx", "httpcore"]: logging.getLogger(lib).setLevel(logging.WARNING)
    if log_level > logging.DEBUG: logging.getLogger("agent_framework").setLevel(logging.WARNING)

    print(f"Running AI Agent Game Mockup (Level: {args.level})..."); print(f"Logging Level: {logging.getLevelName(logging.getLogger().level)}")
    if not os.getenv("OPENAI_API_KEY"): print("CRITICAL ERROR: OPENAI_API_KEY missing.", file=sys.stderr); sys.exit(1)

    play_again = True
    while play_again:
        print("="*40); print(f"Starting New Game Round (Level: {args.level})"); print("="*40)
        setup_ok = asyncio.run(setup_game(level_name=args.level))
        if setup_ok:
            try: asyncio.run(game_loop(verbose_mode=(log_level <= logging.DEBUG), level_name=args.level))
            except KeyboardInterrupt: logger.warning("Game interrupted."); play_again = False
            except Exception as e: logger.critical(f"Unhandled round exception: {e}", exc_info=True); play_again = False
        else: print("Game setup failed."); play_again = False
        # Ask to play again...
        if play_again and sys.stdin.isatty():
             try: play_again = input("\nPlay again? [y/N]: ").lower() == 'y'
             except EOFError: play_again = False
        elif not sys.stdin.isatty(): play_again = False
    print("\nThanks for playing!")
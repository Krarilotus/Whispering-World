# new_game_mockup.py
import asyncio
import logging
import sys
import os
import argparse
from typing import Optional, Any # Use list, dict etc.

# Framework Imports
from agent_framework.core.agent import Agent
from agent_framework.file_utils.yaml_handler import load_agent_config_from_yaml
from agent_framework.world.world import World
from agent_framework.world.world_state import WorldState, Location
# Import Consolidated Interface
from agent_framework.llm.llm_interface import LLM_API_Interface
# Import API Manager and Default Model
from agent_framework.api.agent_api import AgentManager, DEFAULT_MODEL # Use this

# Logger Setup
logger = logging.getLogger(__name__)

# Constants
BASE_DATA_PATH = "data/levels"; DEFAULT_LEVEL = "level1"
WIN_ACTION = "free_adventurer"; LOSE_ACTION = "leave"; STAY_ACTION = "stay_in_conversation"

# Global Variables
warden_agent: Optional[Agent] = None; world_instance: Optional[World] = None
manager: Optional[AgentManager] = None; llm_interface: Optional[LLM_API_Interface] = None
player_state = { "name": "Adventurer", "location": "Cell 17", "objective": "Convince the Old Warden..." }

# World Init Helper
def initialize_new_world_state(world_instance: World):
    logger.info("Initializing new world state..."); try:
        loc_corr_id=world_instance.state.generate_new_entity_id("loc_"); world_instance.state.add_or_update_entity(loc_corr_id, {"id":loc_corr_id, "name":"Corridor outside Cell 17", "type":"Location", "description":"..."}); logger.debug(f"Created: {loc_corr_id} (Corridor)")
        loc_cell_id=world_instance.state.generate_new_entity_id("loc_"); world_instance.state.add_or_update_entity(loc_cell_id, {"id":loc_cell_id, "name":"Cell 17", "type":"Location", "description":"..."}); logger.debug(f"Created: {loc_cell_id} (Cell 17)")
        door_id=world_instance.state.generate_new_entity_id("obj_"); world_instance.state.add_or_update_entity(door_id, {"id":door_id, "name":"Cell 17 Door", "type":"Object", "state":"locked", "location_id":loc_corr_id}); logger.debug(f"Created: {door_id} (Door)")
        logger.info("Default world entities created.")
    except Exception as e: logger.error(f"Error initializing world state: {e}", exc_info=True)

# Game State Check
def check_game_state(resp: dict[str, Any]) -> tuple[bool, bool, str]:
    go=False; pw=False; dlg=resp.get("dialogue","[Silent]"); act=resp.get("action",{}); at=act.get("type")
    if at==WIN_ACTION: go,pw=True,True; logger.info("Win!")
    elif at==LOSE_ACTION: go,pw=True,False; logger.info("Lose!")
    rsn=resp.get("reasoning","");
    if logger.isEnabledFor(logging.DEBUG) and rsn: dlg+=f"\n (Dev: {rsn})"
    return go, pw, dlg

# Async Setup
async def setup_game(level_name: str):
    global manager, warden_agent, world_instance, llm_interface
    warden_agent = None; world_instance = None; llm_interface = None
    lvl_path=os.path.join(BASE_DATA_PATH, level_name); ag_cfg_path=os.path.join(lvl_path,"agent_configs","warden.yaml"); ora_cfg_path=os.path.join(lvl_path,"world_oracle_config.yaml"); ws_path=os.path.join(lvl_path,"world_state.yaml")
    if not all(os.path.exists(p) for p in [lvl_path, ag_cfg_path, ora_cfg_path]): logger.critical("Missing level/config files."); return False
    try:
        if manager is None: manager = AgentManager(); logger.info("AgentManager init.")
        llm_interface = LLM_API_Interface(manager); logger.info("LLM Interface init.")
    except Exception as e: logger.critical(f"Failed init Manager/Interface: {e}"); return False
    logger.info("Loading World Oracle Config..."); ora_cfg=load_agent_config_from_yaml(ora_cfg_path); ora_id=ora_cfg.get('profile',{}).get('assistant_id') if ora_cfg else None
    if not ora_id: logger.warning("Oracle assistant_id missing in config.")
    logger.info("Setting up World...");
    try:
        if os.path.exists(ws_path): world_instance = World.load_state(ws_path, llm_interface, ora_id)
        if not world_instance: logger.warning(f"Creating new world state (load failed or missing: {ws_path})."); world_instance = World(WorldState(), llm_interface, ora_id); initialize_new_world_state(world_instance)
        await world_instance.ensure_world_thread(); logger.info("World setup complete.")
    except Exception as e: logger.critical(f"World setup failed: {e}"); return False
    logger.info(f"Loading Agent config: '{ag_cfg_path}'...");
    warden_agent = Agent.load_from_yaml(ag_cfg_path, manager, world_instance);
    if not warden_agent: logger.critical(f"Failed load agent."); return False
    warden_agent.set_llm_interface(llm_interface)
    logger.info(f"Ensuring Agent Assistant exists for '{warden_agent.name}'...");
    # Use DEFAULT_MODEL imported from agent_api.py
    assistant_ok = await warden_agent.ensure_assistant_exists(default_model=DEFAULT_MODEL)
    if not assistant_ok: logger.critical(f"Failed ensure Assistant."); return False
    logger.info(f"Agent '{warden_agent.name}' Assistant ID: {warden_agent.assistant_id}")
    # Register Agent Location (Robust Lookup)
    if world_instance and warden_agent:
        loc_name_cfg=warden_agent.current_state.location.strip().lower(); start_loc_id=None;
        for eid, props in world_instance.state.entities.items():
            if props.get("type")=="Location" and props.get("name","").strip().lower()==loc_name_cfg: start_loc_id=eid; break
        if start_loc_id: world_instance.state.add_or_update_entity(warden_agent.name, {"id":warden_agent.name,"name":warden_agent.name,"type":"Person","location_id":start_loc_id}); logger.info(f"Registered agent '{warden_agent.name}' at '{start_loc_id}'.")
        else: logger.warning(f"Could not find location '{warden_agent.current_state.location}' to register agent.")
    return True

# Async Game Loop
async def game_loop(verbose_mode: bool, level_name: str):
    global warden_agent, llm_interface, world_instance
    if not warden_agent or not llm_interface or not world_instance: logger.critical("Components missing."); return
    if not warden_agent.world: warden_agent.set_world(world_instance)
    if not warden_agent.llm_interface: warden_agent.set_llm_interface(llm_interface)
    warden_config_path = os.path.join(BASE_DATA_PATH, level_name, "agent_configs", "warden.yaml")
    world_state_path = os.path.join(BASE_DATA_PATH, level_name, "world_state.yaml")
    if not await warden_agent.initialize_conversation(): logger.critical("Failed init thread."); return
    logger.info("Starting game loop..."); print("\n"+"="*40 + "\n--- Castle Catacombs ---\n..."); # Scene setting
    print(f"Goal: {player_state['objective']}\n" + "-"*16)
    game_over = False; player_won = False; turn = 0
    try:
        logger.info("Warden's first action..."); initial_trigger = "[Observe prisoner]";
        resp = await warden_agent.think_and_respond(initial_trigger); go, pw, dlg = check_game_state(resp); print(f"\n{warden_agent.name}: {dlg}"); game_over=go; player_won=pw;
        while not game_over:
            turn += 1; user_input = "[Error]";
            try: user_input = (await asyncio.to_thread(input, f"\n{player_state['name']}: ")).strip()
            except EOFError: user_input = "quit"; logger.info("EOF.")
            except Exception as e: logger.error(f"Input error: {e}"); break
            if not user_input: user_input = "[Player remains silent]"; logger.info("Empty input -> silence.")
            if user_input.lower() == 'quit': logger.info("User quit."); game_over=True; player_won=False; break
            resp = await warden_agent.think_and_respond(user_input); go, pw, dlg = check_game_state(resp); print(f"\n{warden_agent.name}: {dlg}"); game_over=go; player_won=pw;
            if game_over: logger.info(f"Game ended turn {turn}."); break
            await asyncio.sleep(0.1)
    except asyncio.CancelledError: logger.warning("Game loop cancelled."); game_over = True
    except Exception as e: logger.critical(f"Critical loop error: {e}", exc_info=True); game_over = True
    finally:
        if game_over and ('player_won' in locals()): print("\n"+"="*20); print("WIN!" if player_won else "LOSE"); print("="*20 + "\n")
        logger.info("Cleaning up game round...");
        if warden_agent:
            if warden_agent.save_state_to_yaml(): logger.info("Agent state saved.") # Uses stored path
            else: logger.error("Failed save agent state.")
            await warden_agent.end_conversation(delete_thread=True)
        if world_instance:
            if world_instance.save_state(world_state_path): logger.info(f"World state saved.")
            else: logger.error("Failed save world state.")
        logger.info("Game round finished.")

# Main Execution Block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Agent Game Mockup")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help=f"Level under {BASE_DATA_PATH}")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="-v=WARN, -vv=INFO, -vvv=DEBUG")
    args = parser.parse_args()
    log_level = logging.ERROR;
    if args.verbose == 1: log_level = logging.WARNING
    elif args.verbose == 2: log_level = logging.INFO
    elif args.verbose >= 3: log_level = logging.DEBUG
    log_format = "%(asctime)s - %(name)s [%(levelname)s]: %(message)s"; logging.basicConfig(level=log_level, format=log_format, force=True); logging.getLogger().setLevel(log_level)
    for handler in logging.root.handlers: # Ensure console handler level is set
        if isinstance(handler, logging.StreamHandler): handler.setLevel(log_level); handler.setFormatter(logging.Formatter(log_format))
    if log_level > logging.INFO: # Silence libs if not INFO/DEBUG
        for lib in ["openai._base_client", "httpx", "httpcore", "huggingface_hub"]: logging.getLogger(lib).setLevel(logging.WARNING)
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
        if play_again and sys.stdin.isatty():
             try: play_again = input("\nPlay again? [y/N]: ").lower() == 'y'
             except EOFError: play_again = False
        elif not sys.stdin.isatty(): play_again = False
    print("\nThanks for playing!")
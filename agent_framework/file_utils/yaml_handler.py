# agent_framework/utils/yaml_handler.py
import yaml # type: ignore[import]
import logging
from typing import Dict, Any, Optional

# Make sure PyYAML is installed: pip install PyYAML

logger = logging.getLogger(__name__)

def load_agent_config_from_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads agent configuration data from a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
             logger.error(f"YAML file '{file_path}' does not contain a dictionary (root object).")
             return None
        logger.info(f"Successfully loaded agent configuration from {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"YAML configuration file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading YAML file {file_path}: {e}", exc_info=True)
        return None

def save_agent_state_to_yaml(agent_state: Dict[str, Any], file_path: str) -> bool:
    """Saves the agent's current state dictionary to a YAML file."""
    try:
        # Ensure directory exists if path includes directories
        import os
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(agent_state, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Successfully saved agent state to {file_path}")
        return True
    except IOError as e:
         logger.error(f"Error writing YAML file {file_path}: {e}", exc_info=True)
         return False
    except yaml.YAMLError as e:
         logger.error(f"Error formatting data for YAML {file_path}: {e}", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"Unexpected error saving YAML file {file_path}: {e}", exc_info=True)
        return False
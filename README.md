# Project: Whispering World (Working Title)

## 📜 Overview & Concept

**Whispering World** is an experimental framework for creating text-based role-playing game scenarios featuring dynamic, AI-powered Non-Player Characters (NPCs). Leveraging the OpenAI Assistants API, this project aims to move beyond predefined dialogue trees and allow players to interact with characters through natural language, experiencing emergent conversations and potentially influencing the game world based on their interactions.

The core concept involves:

* **AI-Driven NPCs (Agents):** Characters powered by large language models (LLMs via OpenAI API) possessing distinct personalities, motivations, and contextual awareness based on provided instructions and game state.
* **Natural Language Interaction:** Players communicate using typed commands and conversational text. Empty input is treated as silence.
* **Dynamic Narrative:** Agent responses and game outcomes are influenced by the conversation flow and player choices, rather than fixed scripts. Agents are instructed to signal their intended action (e.g., leaving, freeing the player) via a special trigger command at the start of their response.
* **Scenario-Based Gameplay:** Focusing on contained, replayable scenarios or encounters (like the current "Old Warden" example). The game automatically cleans up resources (like the conversation thread and AI assistant) and prompts the user to play again after each round.

This project explores the possibilities and challenges of integrating sophisticated AI into interactive fiction to create more immersive and unpredictable experiences.

## 💻 Technology Stack (Current)

* **Language:** Python 3.10+
* **AI Backend:** OpenAI Assistants API (Requires an API Key)
* **Core Libraries:**
    * `openai` (for interacting with the OpenAI API)
    * `python-dotenv` (for managing API keys)
    * `asyncio` (for handling asynchronous operations)
    * `argparse` (for command-line options)
    * *(Defined in `requirements.txt`)*

## ▶️ How to Run the Current Mockup ("Old Warden" Scenario)

This guide explains how to run the `game_loop_mockup.py` script which features an interaction with the "Old Warden" character.

**Prerequisites:**

1.  **Python:** Version 3.10 or newer installed.
2.  **OpenAI API Key:** You need an active API key from OpenAI ([https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)). Ensure your account has sufficient credits.
3.  **`requirements.txt` file:** This file must be present in the project directory and list necessary packages (like `openai`, `python-dotenv`).

**Setup:**

1.  **Get the Code:** Clone the repository or download and extract the source files (`agent_api.py`, `agent.py`, `game_loop_mockup.py`, `requirements.txt`).
2.  **Navigate to Directory:** Open your terminal or command prompt and change into the project directory.
3.  **Create Virtual Environment (Recommended):**
    ```bash
    # Create environment
    python -m venv .venv
    # Activate it:
    # Windows (CMD/PowerShell): .venv\Scripts\activate
    # macOS/Linux (Bash/Zsh): source .venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    # Install required packages
    pip install -r requirements.txt
    ```
5.  **Set API Key:**
    * Create a file named `.env` in the project's root directory.
    * Add the following line, replacing `your_openai_api_key_here` with your actual key:
        ```
        OPENAI_API_KEY=your_openai_api_key_here
        ```
    * **Do not commit your `.env` file to version control!**

**Running the Game:**

Execute the main mockup script from your activated virtual environment. **Non-streaming mode is now the default.**

* **Default (Non-Streaming, Errors Only Logging):**
    ```bash
    python game_loop_mockup.py
    ```
    *(The Warden's full response will appear after a pause)*

* **Enable Streaming Mode (Optional, if it works in your env):**
    ```bash
    python game_loop_mockup.py --streaming
    ```
    *(The Warden's response will appear token-by-token)*

* **Enable Verbose Logging (Shows INFO logs, Warnings, and raw triggers):**
    ```bash
    # Verbose in default (non-streaming) mode:
    python game_loop_mockup.py --verbose
    # Verbose in streaming mode:
    python game_loop_mockup.py --streaming --verbose
    ```

**In-Game Commands:**

* **Type anything:** To speak to the Warden. (Hitting Enter sends `[Player remains silent]`)
* **`quit`:** To end the current game round (counts as a loss).
* **`interrupt`:** (Only works if using `--streaming`) To stop the Warden while they are responding.

**End of Round:**

* After winning, losing, or quitting, the script will automatically delete the conversation thread and the AI assistant from OpenAI.
* It will then ask if you want to play again. Type `y` to start a new round.

## 🎯 Current Scenario: The Old Warden

The included `game_loop_mockup.py` runs a scenario where the player is an "Adventurer" trapped in Cell 17 of the castle catacombs. The "Old Warden", a grumpy, seasoned guard with a hidden past (related to his daughter), approaches the cell.

* The Warden initiates the conversation.
* The player's goal is to convince the Warden, through dialogue, to unlock the cell door. Success requires the Warden's response to *start* with the exact phrase `{free adventurer}`.
* The player loses if the Warden leaves, indicated by their response *starting* with `{leave}`. This is triggered by player silence, insults, or lack of engagement.
* The Warden is instructed to be brief, use specific mannerisms (like non-verbal actions in `[]`), and adhere strictly to starting every response with a trigger command (`{free adventurer}`, `{leave}`, or `{stay in conversation}`).

## 💡 Project Goals (Long Term)

These goals represent the broader vision for the Whispering World framework:

1.  **Simplified Setup & Execution:** Aim for a near "one-click" installer/launcher experience.
2.  **Local LLM Execution (Future Goal):** Explore feasibility of local execution (e.g., via Ollama) with target specs (16GB+ RAM, 8GB+ VRAM, modern CPU/GPU). *(Current version uses OpenAI API)*.
3.  **Robust Game Data Specification:** Define an engine-like format for game data (world state, items, agents, knowledge, time). Address dynamic updates and consistency.
4.  **Advanced Agent Personality & Interaction:** Research deeper personality modeling, inter-agent communication, potential "DM" agent oversight, and response safety.
5.  **Code Quality & Portfolio Building:** Maintain clean, documented, modular code.
6.  **Background Goal: LLM Interaction Literacy:** Implicitly teach players effective LLM interaction through gameplay (e.g., the challenge of persuading the constrained Warden).

## ✨ Future Ideas & Extensions

* **Persistence:** Saving and loading game state.
* **Advanced Agent Memory:** Integrating long-term memory solutions.
* **Fine-Tuning:** Customizing models for specific characters.
* **Expanded Gameplay:** More content (rooms, items, agents, puzzles, story).
* **Alternative Interfaces:** Discord bot, Web UI.

## 🤝 Contributing

(Placeholder - Define contribution guidelines if applicable) Currently developed by [Your Name/Handle] with AI assistance. Feedback welcome!

## 📄 License

(Placeholder - Choose a license, e.g., MIT License or specify "License TBD")
# Project: Whispering World (Working Title)

## 📜 Overview & Concept

**Whispering World** is an experimental framework for creating text-based role-playing game scenarios featuring dynamic, AI-powered Non-Player Characters (NPCs). Leveraging the OpenAI Assistants API, this project aims to move beyond predefined dialogue trees and allow players to interact with characters through natural language, experiencing emergent conversations and potentially influencing the game world based on their interactions.

The core concept involves:

* **AI-Driven NPCs (Agents):** Characters powered by large language models (LLMs via OpenAI API) possessing distinct personalities, motivations, and contextual awareness based on provided instructions and game state.
* **Natural Language Interaction:** Players communicate using typed commands and conversational text.
* **Dynamic Narrative:** Agent responses and game outcomes are influenced by the conversation flow and player choices, rather than fixed scripts.
* **Scenario-Based Gameplay:** Focusing on contained, replayable scenarios or encounters (like the current "Old Warden" example).

This project explores the possibilities and challenges of integrating sophisticated AI into interactive fiction to create more immersive and unpredictable experiences.

## 💻 Technology Stack (Current)

* **Language:** Python 3.10+
* **AI Backend:** OpenAI Assistants API (Requires an API Key)
* **Core Libraries:**
    * `openai` (for interacting with the OpenAI API)
    * `python-dotenv` (for managing API keys)
    * `asyncio` (for handling asynchronous operations like streaming)
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
2.  **Navigate to Directory:** Open your terminal or command prompt and change into the project directory containing the Python files.
3.  **Create Virtual Environment (Recommended):**
    ```bash
    # Create environment (replace .venv with your preferred name)
    python -m venv .venv
    # Activate it:
    # Windows (CMD/PowerShell):
    .venv\Scripts\activate
    # macOS/Linux (Bash/Zsh):
    source .venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    # Install required packages from requirements.txt
    pip install -r requirements.txt
    ```
5.  **Set API Key:**
    * Create a file named `.env` in the project's root directory.
    * Add the following line to the `.env` file, replacing `your_openai_api_key_here` with your actual key:
        ```
        OPENAI_API_KEY=your_openai_api_key_here
        ```
    * **Never commit your `.env` file to version control!**

**Running the Game:**

Execute the main mockup script from your activated virtual environment:

* **Default (Streaming Mode, Warnings Only):**
    ```bash
    python game_loop_mockup.py
    ```
* **Non-Streaming Mode (Response appears all at once):**
    ```bash
    python game_loop_mockup.py --non-streaming
    ```
* **Verbose Mode (Detailed Logs, Shows Raw Triggers):**
    ```bash
    python game_loop_mockup.py --verbose
    ```
* **Non-Streaming + Verbose:**
    ```bash
    python game_loop_mockup.py --non-streaming --verbose
    ```

**In-Game Commands:**

* **Type anything:** To speak to the Warden. (Empty input becomes `[Player remains silent]`)
* **`quit`:** To end the game (counts as a loss).
* **`interrupt`:** (Only in Streaming mode) To stop the Warden while they are responding.

## 🎯 Current Scenario: The Old Warden

The included `game_loop_mockup.py` runs a scenario where the player is an "Adventurer" trapped in Cell 17 of the castle catacombs. The "Old Warden", a grumpy, seasoned guard powered by the AI Assistant, approaches the cell.

* The Warden initiates the conversation.
* The player's goal is to convince the Warden, through dialogue, to unlock the cell door. Success is indicated by the Warden's response starting with the exact phrase `{free adventurer}`.
* The player loses if the Warden decides the conversation is fruitless or gets annoyed and leaves, indicated by their response starting with `{leave}`.
* The Warden is instructed to be brief, use specific mannerisms, and adhere strictly to the trigger command format (`{trigger} Actual dialogue...`).

## 💡 Project Goals (Long Term)

These goals represent the broader vision for the Whispering World framework:

1.  **Simplified Setup & Execution:**
    * Aim for a near "one-click" installer/launcher experience, potentially handling backend setup (like API key checks or future local model integration) more smoothly.

2.  **Local LLM Execution (Future Goal):**
    * Explore the feasibility of running the framework entirely locally using models via Ollama or similar backends.
    * *Target* recommended specs for potential local execution (subject to change based on model choice):
        * VRAM: 8GB+ (GPU acceleration highly beneficial)
        * GPU: CUDA-capable preferred (or Metal for macOS)
        * CPU: Modern multi-core (e.g., Ryzen 5 2600 / Intel i5 8th Gen or better)
        * RAM: 16GB+
        * Storage: ~50GB free space (SSD recommended) for models.
    * *(Note: Current implementation relies on the OpenAI API).*

3.  **Robust Game Data Specification:**
    * Define a clear, engine-like format for specifying all game data: rooms, items, agent properties, world state, narrative triggers.
    * Develop methods for encoding world knowledge within the game state, potentially including time-based elements.
    * Investigate ways to dynamically add unforeseen events or knowledge to the game state.
    * Evaluate mechanisms for maintaining game world consistency when interacting with probabilistic LLMs.

4.  **Advanced Agent Personality & Interaction:**
    * Research and implement techniques to give agents deeper, more consistent personalities beyond the initial prompt.
    * Explore complex interactions:
        * *Gimmick Idea:* Allow AI agents to converse with and influence each other.
        * Push the boundaries of achievable personality types within the framework.
        * *Potential Feature:* Introduce a "Dungeon Master" AI agent overseeing world events and agent actions for narrative coherence.
    * Evaluate and potentially implement safety/alignment mechanisms for agent responses if user-generated content or broader interaction is enabled.

5.  **Code Quality & Portfolio Building:**
    * Maintain clean, well-documented, and modular code suitable for future expansion or showcasing.

6.  **Background Goal: LLM Interaction Literacy:**
    * Design gameplay loops that implicitly teach players effective ways to interact with modern LLMs (e.g., clear prompting, understanding context, recognizing limitations).
    * *Initial Concept:* An early game scenario could involve literally trying to "jailbreak" or persuade a highly constrained guard agent, teaching prompt interaction techniques.

## ✨ Future Ideas & Extensions

* **Persistence:** Saving and loading game state.
* **Advanced Agent Memory:** Integrating techniques like vector databases or frameworks for true long-term memory.
* **Fine-Tuning:** Exploring fine-tuning models (OpenAI or local) for specific character behaviors or domain knowledge.
* **Expanded Gameplay:** More rooms, items, puzzles, agents, character stats, and a deeper storyline.
* **Alternative Interfaces:** Discord bot integration, simple web UI.

## 🤝 Contributing

(Placeholder - Define contribution guidelines if applicable) Currently developed by [Your Name/Handle] with AI assistance. Feedback welcome!

## 📄 License

(Placeholder - Choose a license, e.g., MIT License or specify "License TBD")
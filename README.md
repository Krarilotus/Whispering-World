# Project: Agent Framework RPG (Working Title)

## 📜 Overview & Concept

This project provides a Python framework for creating text-based role-playing game scenarios featuring dynamic, AI-powered Non-Player Characters (NPCs) interacting within an AI-mediated world. Leveraging the OpenAI Assistants API, it aims to simulate believable agents with deep personalities, motivations, and memories, who reason about their world and player interactions with enhanced consistency.

**Core Concepts:**

* **Deep Agent Architecture:** NPCs (Agents) are built upon a modular framework incorporating:
    * **Personality:** Modeled using OCEAN scores and RPG-style Traits/Flaws.
    * **Motivation:** Driven by psychological needs (SDT-inspired), ideals, bonds, and goals.
    * **Memory:** Long-term memory stream (inspired by Generative Agents) with observations, reflections, and generated facts. Retrieval considers recency, importance, and relevance (basic keyword matching currently, embeddings planned).
    * **Affective State:** Tracks valence, arousal, and discrete emotions.
    * **Current State:** Location, inventory, current action/goal.
* **Intelligent World:** A `World` module maintains the objective state (facts, locations, objects, time). It utilizes its own "World Oracle" LLM Assistant to:
    * Answer factual queries based *only* on the current world state.
    * Check proposed new facts for consistency against existing world facts.
* **Consistency Barrier:** Before an agent responds to input, it performs a verification step:
    1.  Extracts factual claims from the input.
    2.  Classifies claims as 'agent_internal' or 'world_objective' using an LLM.
    3.  Verifies internal claims against its own memory, potentially generating plausible synthetic memories if unknown.
    4.  Verifies objective claims by querying the `World` module. If the world doesn't know, the agent checks the claim's plausibility *for the world* and can *propose* it as a new objective fact (if plausible).
    5.  The results of this verification process heavily inform the agent's final reasoning and response generation.
* **Natural Language Interaction:** Players interact via typed text.
* **Data-Driven Setup:** Levels, agents, and the world oracle are configured via YAML files organized into level-specific directories.
* **Dynamic Narrative:** Agent responses and potential world state changes emerge from AI reasoning grounded in agent state, world state, and consistency checks.

This framework facilitates exploring complex AI character simulation, consistency maintenance, and dynamic world-building within interactive text adventures.

## 💻 Technology Stack

* **Language:** Python 3.10+
* **AI Backend:** OpenAI Assistants API (Requires an API Key)
* **Core Libraries:**
    * `openai` (for OpenAI API interaction)
    * `PyYAML` (for loading/saving configuration and state)
    * `python-dotenv` (for managing API keys)
    * `asyncio` (for asynchronous operations)
    * `argparse` (for command-line options)
    * *(Defined in `requirements.txt`)*

## ▶️ How to Run ("Old Warden" Scenario - Level 1)

**Prerequisites:**

1.  **Python:** Version 3.10 or newer installed.
2.  **OpenAI API Key:** An active API key from OpenAI ([https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)) with sufficient credits.
3.  **Project Files:** The complete framework code, including the `agent_framework` directory, `new_game_mockup.py`, `requirements.txt`, and the `data/levels/level1` directory containing `agent_configs/warden.yaml` and `world_oracle_config.yaml`.

**Setup:**

1.  **Get the Code:** Clone or download the project source code.
2.  **Navigate to Directory:** Open your terminal/command prompt and `cd` into the project's root directory.
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate: .venv\Scripts\activate (Win) or source .venv/bin/activate (Mac/Linux)
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` lists `openai`, `PyYAML`, `python-dotenv`)*
5.  **Set API Key:**
    * Create a `.env` file in the project root.
    * Add your OpenAI API key:
        ```dotenv
        OPENAI_API_KEY=sk-YourKeyHere
        ```
    * **Important:** The framework *creates* the necessary OpenAI Assistants based on the YAML configs if they don't exist. You *do not* need to manually create assistants or set assistant IDs in the `.env` file (unless you specifically want to reuse existing ones by putting their IDs in the respective YAML files).

**Running the Game:**

Execute the main script from your activated virtual environment. Specify the level directory using `--level`.

* **Run Level 1 (Default):**
    ```bash
    python new_game_mockup.py
    # Or explicitly:
    python new_game_mockup.py --level level1
    ```

* **Enable Verbose Logging (Shows DEBUG info, reasoning, world queries):**
    ```bash
    python new_game_mockup.py --verbose
    # Or for a specific level:
    python new_game_mockup.py --level level1 --verbose
    ```

**First Run Note:** The first time you run a level, the script will interact with the OpenAI API to create the necessary Assistants (e.g., for the Warden and the World Oracle) based on the `.yaml` config files. This might take a few extra seconds. Subsequent runs should be faster as they will reuse the created Assistants (their IDs might be stored back in the YAML or managed internally - current implementation relies on recreation if ID not found/specified in YAML).

**In-Game Commands:**

* **Type anything:** To speak to the Warden.
* **`quit`:** To end the current game round.
* **`state` (requires `--verbose`):** Shows the agent's internal state snapshot.
* **`world` (requires `--verbose`):** Shows a summary of the current world state facts and locations.

**End of Round:**

* The script automatically cleans up the *conversation threads* on OpenAI.
* It **does not** automatically delete the persistent OpenAI *Assistants* (Agent/Oracle) by default, allowing reuse.
* The current world state is saved to `<level_dir>/world_state.yaml`.
* You will be prompted to play again.

## 🎯 Current Scenario: The Old Warden (Level 1)

This scenario uses the configuration found in `data/levels/level1/`.

* **Player:** "Adventurer" trapped in Cell 17.
* **Agent:** The "Old Warden" (configured in `warden.yaml`), a grumpy guard with a hidden past.
* **World Oracle:** A neutral AI (configured in `world_oracle_config.yaml`) that knows basic facts about the catacombs (defined in `world_state.py` initially or loaded from `world_state.yaml`).
* **Interaction:** The Warden initiates. Player goal is to convince the Warden to act (`free_adventurer` action type in response). The Warden might leave (`leave` action type) if annoyed or bored. Player interactions are filtered through the consistency barrier, checking claims against the Warden's memory and the objective world state. Plausible new facts might be added to the Warden's memory or the world state during the conversation.

## 💡 Framework Goals & Concepts

* **Believability & Consistency:** Simulate agents grounded in psychology and RPG elements, maintaining consistency through memory and the consistency barrier.
* **World Interaction:** Allow agents to query and potentially modify an objective world state.
* **Modularity:** Separate agent core, world logic, LLM interaction, and file handling.
* **Data-Driven Design:** Define scenarios, agents, and world setup using external YAML files.
* **Self-Sufficiency:** Framework handles OpenAI Assistant creation/loading based on configs.

## ✨ Future Ideas & Extensions

* **Embedding-Based Memory:** Implement vector similarity search for more relevant memory retrieval.
* **Sophisticated Reflection:** Develop more complex reflection mechanisms for deeper agent learning.
* **World Event System:** Propagate world state changes (e.g., new facts, location updates) to relevant agents for perception.
* **Inter-Agent Communication:** Allow agents to interact with each other.
* **Advanced Action Handling:** More complex parsing and execution of agent actions within `world.py`.
* **Local LLM Support:** Integrate libraries like `llama-cpp-python` or frameworks supporting local models.
* **UI/Alternative Interfaces:** Web interface, Discord bot.

## 🤝 Contributing

(Placeholder) Feedback and ideas are welcome!

## 📄 License

(Placeholder - Choose License)
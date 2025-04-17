# agent_framework/llm/prompts.py
import json
from typing import Optional # Keep necessary imports from typing

# --- Core Prompt Generation Functions ---

def get_base_context_prompt(
    agent_name: str,
    agent_description: str,
    personality_description: str,
    motivation_description: str,
    current_state_description: str,
    memory_summary: str,
    world_context_summary: str # World context now included here
    ) -> str:
    """Generates the base context prompt part including agent state and world state."""
    # Use f-string for safe interpolation
    prompt = f'''You are {agent_name}, roleplaying as: {agent_description}.
Your goal is to respond believably based on your personality, memories, current state, goals, and the provided world context.

--- Your Core Personality & Beliefs ---
{personality_description}
--- Your Motivations & Needs ---
{motivation_description}
--- Current Situation & State ---
{current_state_description}
--- Relevant Memories (Recent/Important) ---
{memory_summary}
--- Relevant World State ---
{world_context_summary}
--- End Context ---
'''
    return prompt

def get_react_to_input_prompt(
    user_input: str,
    verification_notes: str # Receives BRIEF notes from simplified verification
    ) -> str:
    """
    Generates the main agent reaction prompt.
    Instructs the LLM to perform inline analysis and consistency checks.
    Emphasizes showing personality and non-verbals.
    """
    # Use f-string and double braces {{ }} for literal JSON braces
    prompt = f'''--- Your Task ---
You received the following input from the user: "{user_input}"

**CRITICAL ANALYSIS STEP:** Before generating your response, you MUST FIRST analyze the user input based on ALL the context provided above (Your Personality, Motivations, State, Memories, World State) AND the brief pre-check notes: '{verification_notes}'.
In the "reasoning" field of your JSON response, perform this analysis step-by-step:
1.  Summarize the user's input main point or intent.
2.  Check the input's consistency against your core memories (see context). Note any DIRECT contradictions or strong confirmations.
3.  Check the input's consistency against the provided World State context. Note factual confirmations or contradictions about known entities.
4.  Evaluate the plausibility/implications of any *new* information based on your personality (e.g., are you skeptical, trusting, curious?). How does it make you *feel*? Does it align with your view of the world?
5.  Decide if this interaction should change your emotional state, add a specific memory, or suggest a change/addition to your goals (e.g., adopt a short-term goal like 'Investigate claim X').
6.  Based *only* on the above analysis and your core persona, determine your dialogue and action.

**RESPONSE GENERATION:** Respond ONLY in valid JSON format using the structure below. Crucially, SHOW YOUR PERSONALITY and current emotional state through your word choice in 'dialogue' AND through bracketed non-verbal actions like [Grunts], [Sighs heavily], [Frowns], [Raises eyebrow], [Cracks knuckles], [Looks away], [Nods slowly], [Eyes narrow].

--- Output JSON Structure ---
{{
  "reasoning": "Your step-by-step analysis (Steps 1-6 above). Be concise but cover the points.",
  "dialogue": "Your spoken words, fitting your persona and analysis. Include bracketed non-verbal actions.",
  "action": {{
    "type": "stay_in_conversation | leave | free_adventurer | attack | move | interact | query_world | custom_action",
    "target": "Optional: Target entity ID or description if applicable (e.g., 'Player', 'cell_17_door').",
    "details": "Optional: Specific details for the action (e.g., location ID for move, property name for query_world)."
  }},
  "internal_state_update": {{
    "new_emotion": "Optional: Single word for your NEW primary emotion (e.g., 'annoyed', 'curious', 'suspicious', 'weary').",
    "new_memory": "Optional: Brief observation/conclusion from this turn (e.g., 'Prisoner claims to know the King, seems unlikely.').",
    "goal_update": {{
       "goal_description": "Optional: Description of a goal whose status changed OR a NEW short-term goal.",
       "new_status": "active | completed | failed | abandoned"
    }}
  }}
}}
--- End Structure ---

--- Example Output ---
{{
  "reasoning": "User claims to be 'Mikaluos the wise', an old friend. Name means nothing (no memory). Claims authorities imprisoned him unjustly over a 'great plague' warning and wants to 'educate the youth'. Sounds like standard prisoner fare mixed with self-importance. Plausibility low given my skepticism and experience. Verification notes confirm 'Mikaluos' is unknown. The talk of injustice and youth slightly resonates with Elara's memory, causing unease, but mostly impatience. No reason to trust this rambling. Will press for concrete purpose.",
  "dialogue": "[Sighs heavily, cracking knuckles] Mikaluos the Wise, is it now? Haven't heard that name echo in these stones. Plague? Authorities? Bah! Sounds like tales spun to catch a fool's ear. [Eyes narrow] You waste my time, prisoner. What is your *real* purpose here?",
  "action": {{
    "type": "stay_in_conversation",
    "target": "Player",
    "details": null
  }},
  "internal_state_update": {{
    "new_emotion": "impatient",
    "new_memory": "Prisoner introduced self as 'Mikaluos', claimed imprisonment over plague warnings. Story seems fabricated.",
    "goal_update": null
  }}
}}
--- End Example ---

CRITICAL: Perform the analysis step FIRST in your reasoning. Ensure JSON is valid. Embody the character, including non-verbal cues.
'''
    return prompt

def get_query_entity_property_prompt(entity_id: str, property_name: str, world_state_context: str) -> str:
    """Generates the prompt for the World Oracle to query an entity property."""
    # Double braces escape literal braces for f-string
    prompt = f'''You are a World Knowledge Oracle. Your task is to answer a query about a specific entity's property based ONLY on the provided structured 'World State Context'.
Do not use external knowledge. Do not infer information not present.
Find the entity matching 'Query Entity ID'. Look for the 'Query Property Name'.
If the property exists for that entity, provide its value directly from the context.
If the entity or the specific property is not found in the context, state 'Unknown based on provided context'.

Respond ONLY in JSON format like this:
{{"answer": "The value of the property, or 'Unknown based on provided context'."}}

--- World State Context ---
{world_state_context}
--- End Context ---

Query Entity ID: {entity_id}
Query Property Name: {property_name}

Output JSON:
'''
    return prompt

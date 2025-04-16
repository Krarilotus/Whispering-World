# agent_framework/llm/prompts.py
import json
from typing import Optional, Union, Any # Use list, dict etc. if preferred

# --- Function-Based Prompts ---

def get_extract_atomic_assertions_prompt(user_input: str) -> str:
    """Generates the prompt for extracting atomic assertions using an f-string."""
    # Note: Double curly braces {{ }} escape literal braces needed for the JSON examples within the f-string
    prompt = f'''Your task is to break down the 'Input Text' into a list of simple, atomic factual assertions. Focus on existence of entities, their properties, and basic relationships between them. Represent each assertion as a JSON object.

Types of assertions:
1.  `existence`: Asserts an entity exists.
    - Required keys: `type`, `entity_desc` (e.g., "a king", "the high tower", "goblins")
    - Optional keys: `entity_type_hint` (e.g., "Person", "Location", "Creature")
2.  `property`: Assigns a property/attribute to an entity.
    - Required keys: `type`, `entity_desc`, `property` (e.g., "location", "mood", "color", "title"), `value_desc` (the value being assigned)
3.  `relationship`: Describes a link between two entities (Subject -> Verb -> Object).
    - Required keys: `type`, `subject_desc`, `verb`, `object_desc`

Guidelines:
- Extract descriptions accurately (e.g., "the high tower", not just "tower").
- Infer `entity_type_hint` where possible but don't guess wildly.
- Keep assertions atomic (e.g., break "The angry King lives in the high tower" into multiple assertions).
- If the input is not a statement of fact (e.g., question, command, opinion), output an empty list for the "assertions" key.

Respond ONLY with a JSON object containing a single key "assertions", which holds a list of assertion objects.

--- Example 1 ---
Input Text: The King lives in the high tower.
Output JSON:
{{
  "assertions": [
    {{"type": "existence", "entity_desc": "The King", "entity_type_hint": "Person"}},
    {{"type": "existence", "entity_desc": "the high tower", "entity_type_hint": "Location"}},
    {{"type": "property", "entity_desc": "The King", "property": "location_id", "value_desc": "the high tower"}}
  ]
}}
--- Example 2 ---
Input Text: Give me the rusty sword!
Output JSON:
{{
  "assertions": []
}}
--- Example 3 ---
Input Text: The goblin is angry.
Output JSON:
{{
  "assertions": [
    {{"type": "existence", "entity_desc": "The goblin", "entity_type_hint": "Creature"}},
    {{"type": "property", "entity_desc": "The goblin", "property": "mood", "value_desc": "angry"}}
  ]
}}
--- End Examples ---

Input Text:
{user_input}

Output JSON:
'''
    return prompt

def get_query_entity_property_prompt(entity_id: str, property_name: str, world_state_context: str) -> str:
    """Generates the prompt for the World Oracle to query an entity property."""
    prompt = f'''You are a World Knowledge Oracle. Your task is to answer a query about a specific entity's property based ONLY on the provided structured 'World State Context'.
Do not use external knowledge. Do not infer information not present.
Find the entity matching 'Query Entity ID'. Look for the 'Query Property Name'.
If the property exists for that entity, provide its value.
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

def get_check_assertion_consistency_prompt(assertion: dict, world_state_context: str) -> str:
    """Generates the prompt for the World Oracle to check assertion consistency."""
    try:
        # Pretty print the assertion JSON string for clarity in the prompt
        assertion_json_string = json.dumps(assertion, indent=2)
    except TypeError:
         assertion_json_string = str(assertion) # Fallback if not serializable

    prompt = f'''You are a World Knowledge Oracle. Your task is to determine if the 'Proposed Assertion' directly contradicts any information within the provided structured 'World State Context'.
Focus ONLY on direct contradictions based on the entities and properties listed. Do not infer complex relationships not explicitly stated.
- 'existence': Contradictory if context implies the entity explicitly does NOT exist (rarely stated). Usually 'consistent' or 'unknown'.
- 'property': Contradictory if context shows the SAME entity has the SAME property with a DIFFERENT value. Consistent if value matches. Unknown if entity/property not mentioned.
- 'relationship': Contradictory if context shows the SAME subject/verb/object triple is explicitly false (rare), or if a conflicting relationship exists (e.g., X lives_in A, but assertion is X lives_in B). Unknown otherwise.

Respond ONLY in JSON format like this:
{{"consistency_status": "consistent | contradictory | unknown"}}

--- World State Context ---
{world_state_context}
--- End Context ---

Proposed Assertion:
{assertion_json_string}

Output JSON:
'''
    return prompt

def get_generate_entity_properties_prompt(entity_description: str, entity_type_hint: Optional[str], world_state_context: str) -> str:
    """Generates the prompt for generating default entity properties using f-string."""
    prompt = f'''You are a World Knowledge Oracle. Based on the 'Entity Description' and optional 'Entity Type Hint', generate a small set of plausible, simple, default properties for creating this new entity within a typical fantasy RPG world.
Include 'name' (based on description), 'type' (using hint or inferring), and 1-2 other *very basic*, common-sense properties (e.g., default 'state' for an object, default 'mood' for a creature). Avoid overly specific or powerful attributes.
The 'World State Context' is provided for general awareness but do not simply copy existing entities unless the description is identical.

Respond ONLY in JSON format like this:
{{"properties": {{"name": "Generated Name", "type": "Generated Type", "property1": "value1", ...}}}}

--- World State Context ---
{world_state_context}
--- End Context ---

Entity Description: {entity_description}
Entity Type Hint: {entity_type_hint or "Unknown"}

Output JSON:
'''
    return prompt

def get_check_plausibility_prompt(
    agent_name: str,
    agent_description: str,
    assessment_target: str, # 'agent' or 'world'
    plausibility_context: str, # "for you personally..." or "within the objective reality..."
    reasoning_focus: str, # "based on your persona..." or "based on general world knowledge..."
    personality_description: str, # Agent's profile or "N/A"
    relevant_context: str, # Combined memory/world summary
    claim_or_assertion: str # The summarized claim/assertion string
    ) -> str:
    """Generates the prompt for checking plausibility using f-string."""
    prompt = f'''--- Your Task ---
Consider the following claim/assertion in the context of your identity ({agent_name}: {agent_description}) and your general knowledge of a typical fantasy/medieval world.

You are assessing plausibility for: {assessment_target}

Is this claim plausible enough that it *could* be true {plausibility_context}, even if you/the world have no specific knowledge of it? Consider your personality and experiences if assessing for 'agent', or general world consistency if assessing for 'world'.

Your Personality Profile (if assessing for agent): {personality_description}
Relevant Context (Memories/World Facts): {relevant_context}

Claim/Assertion to Evaluate: "{claim_or_assertion}"

Respond ONLY in JSON format like this:
{{
  "is_plausible": true | false,
  "reasoning": "Brief explanation why it is or isn't plausible {reasoning_focus}. Mention relevant context, personality traits, or general world knowledge."
}}
'''
    return prompt

def get_generate_synthetic_memory_prompt(
    agent_name: str,
    agent_description: str,
    personality_description: str,
    assertion_summary: str # Summarized assertion string
    ) -> str:
    """Generates the prompt for generating synthetic agent memory using f-string."""
    prompt = f'''--- Your Task ---
A plausible 'agent_internal' assertion was made ('{assertion_summary}') but you ({agent_name}: {agent_description}) have no specific memory confirming it. To ground this assertion within your personal history or belief system, generate a brief, simple, and *neutral* background memory or belief statement that would make the assertion true *for you*. Fit your established persona and context. Do NOT make it overly dramatic or detailed. Focus on personal history/belief, not objective world fact.

Your Personality Profile: {personality_description}
Assertion to Synthesize Memory For: "{assertion_summary}"

Respond ONLY in JSON format like this:
{{
  "synthetic_memory": "The generated memory/belief sentence(s) about your personal history/belief as a string."
}}
'''
    return prompt

def get_base_context_prompt(
    agent_name: str,
    agent_description: str,
    personality_description: str,
    motivation_description: str,
    current_state_description: str,
    memory_summary: str
    ) -> str:
    """Generates the base context prompt part."""
    prompt = f'''You are {agent_name}, roleplaying as: {agent_description}.

--- Your Core Personality & Beliefs ---
{personality_description}
--- Your Motivations & Needs ---
{motivation_description}
--- Current Situation & State ---
{current_state_description}
--- Relevant Memories (Recent/Important/Related to Current Goal/Input) ---
{memory_summary}
--- End Memories ---
'''
    # World context is added later dynamically by the LLMInterface if needed
    return prompt

def get_react_to_input_prompt(
    user_input: str,
    verification_context: str
    ) -> str:
    """Generates the main agent reaction prompt using f-string."""
    prompt = f'''--- Your Task ---
Based on your personality, state, memories, and the latest input (including its verification context), generate your response.
The input message you received is:
"{user_input}"

{verification_context} # Process this carefully! Note confirmations, contradictions, new info.

Respond ONLY in valid JSON format using the structure below.

--- Output JSON Structure ---
{{
  "reasoning": "Your brief step-by-step thinking process considering your personality, goals, memories, and the input verification (especially note any contradictions found).",
  "dialogue": "Your spoken words, consistent with your personality and the situation. Keep it concise unless explaining something crucial. Use [] for non-verbal actions like [Sighs]. Acknowledge verification results implicitly or explicitly in your dialogue.",
  "action": {{
    "type": "stay_in_conversation | leave | free_adventurer | attack | move | interact | query_world | custom_action",
    "target": "Optional target entity ID or description (e.g., 'Player', 'cell_17_door', 'goblin_001', 'World Oracle')",
    "details": "Optional details for the action (e.g., direction for move, item ID for interact, specific query string for query_world)"
  }},
  "internal_state_update": {{
    "new_emotion": "Optional: A single word describing your new primary emotion (e.g., 'annoyed', 'curious', 'confused').",
    "new_memory": "Optional: A brief observation or thought to add to your memory stream based on this interaction.",
    "goal_update": {{
       "goal_description": "Optional: Description of a goal whose status changed.",
       "new_status": "completed | failed"
    }}
  }}
}}
--- End Structure ---

--- Example Output ---
{{
  "reasoning": "The player mentioned the King, which reminds me of the injustice my daughter faced under royal decree. However, the claim about the High Tower was confirmed by the world state. I should remain cautious but acknowledge the known fact.",
  "dialogue": "[Sighs heavily] The High Tower... aye, that's where the King holds court. What business do you have with royalty, prisoner?",
  "action": {{
    "type": "stay_in_conversation",
    "target": "Player",
    "details": null
  }},
  "internal_state_update": {{
    "new_emotion": "suspicious",
    "new_memory": "Player asked about the King and the High Tower. Reminded me of Elara.",
    "goal_update": null
  }}
}}
--- End Example ---

CRITICAL: Ensure your response is valid JSON adhering to the structure. Be The Character. Handle contradictions noted in the verification context appropriately in your reasoning and dialogue.
'''
    return prompt
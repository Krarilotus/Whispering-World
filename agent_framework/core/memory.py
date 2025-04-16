# agent_framework/core/memory.py
import time
import datetime
import uuid
from typing import Optional, List, Dict, Any, Tuple

# --- Constants ---
DEFAULT_IMPORTANCE = 5.0 # On a scale of 1-10
RECENCY_WEIGHT = 1.0
IMPORTANCE_WEIGHT = 1.0
RELEVANCE_WEIGHT = 1.0 # Requires embedding similarity in full implementation

class MemoryObject:
    """Base class for different types of memories."""
    def __init__(self, description: str, importance: float = DEFAULT_IMPORTANCE, timestamp: Optional[float] = None):
        self.id: str = str(uuid.uuid4())
        self.description: str = description
        self.creation_timestamp: float = timestamp if timestamp is not None else time.time()
        self.last_access_timestamp: float = self.creation_timestamp
        self.importance_score: float = max(1.0, min(10.0, importance)) # Clamp importance
        # Placeholder for embedding vector - requires a real embedding model & storage
        self.embedding_vector: Optional[List[float]] = None

    def __repr__(self) -> str:
        time_str = datetime.datetime.fromtimestamp(self.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return f"[{time_str}][{self.importance_score:.1f}] {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes basic memory info to a dictionary."""
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "description": self.description,
            "creation_timestamp": self.creation_timestamp,
            "last_access_timestamp": self.last_access_timestamp,
            "importance_score": self.importance_score,
            # Embedding vectors are usually not directly serialized to YAML
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryObject':
        """Deserializes from a dictionary."""
        # Note: This is basic; subclasses might need specific handling.
        # We determine the type from the dict to reconstruct the correct class.
        mem_type = data.get("type", "MemoryObject")
        if mem_type == "Observation":
            mem = Observation(data["description"], data["importance_score"], data["creation_timestamp"])
        elif mem_type == "Reflection":
            mem = Reflection(data["description"], data.get("synthesized_memory_ids", []), data["importance_score"], data["creation_timestamp"])
        elif mem_type == "GeneratedFact":
             mem = GeneratedFact(data["description"], data.get("source_claim", ""), data["importance_score"], data["creation_timestamp"])
        else: # Default to MemoryObject if type unknown or base
            mem = cls(data["description"], data["importance_score"], data["creation_timestamp"])

        mem.id = data.get("id", mem.id) # Restore ID
        mem.last_access_timestamp = data.get("last_access_timestamp", mem.creation_timestamp)
        # Embedding would be loaded separately if stored externally
        return mem

class Observation(MemoryObject):
    """A memory of a direct observation or event."""
    def __init__(self, description: str, importance: float = DEFAULT_IMPORTANCE, timestamp: Optional[float] = None):
        super().__init__(description, importance, timestamp)

class Reflection(MemoryObject):
    """A higher-level insight synthesized from other memories."""
    def __init__(self, description: str, synthesized_memory_ids: List[str], importance: float = DEFAULT_IMPORTANCE + 2.0, timestamp: Optional[float] = None): # Reflections are often more important
        super().__init__(description, importance, timestamp)
        self.synthesized_memory_ids: List[str] = synthesized_memory_ids

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["synthesized_memory_ids"] = self.synthesized_memory_ids
        return data

class GeneratedFact(MemoryObject):
    """A memory created synthetically during claim verification."""
    def __init__(self, description: str, source_claim: str, importance: float = DEFAULT_IMPORTANCE -1.0, timestamp: Optional[float] = None): # Start slightly less important?
        super().__init__(description, importance, timestamp)
        self.source_claim: str = source_claim # Track the claim that triggered this

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["source_claim"] = self.source_claim
        return data

class MemoryStream:
    """Manages the agent's collection of memories."""
    def __init__(self):
        self._memories: List[MemoryObject] = []
        self._reflection_threshold = 100 # Example: Trigger reflection after cumulative importance reaches this

    def add_memory(self, memory: MemoryObject):
        """Adds a new memory to the stream."""
        self._memories.append(memory)
        # Potentially trigger reflection if threshold is met by recent memories
        # self.check_reflection_trigger()

    def get_memories(self, limit: Optional[int] = None) -> List[MemoryObject]:
        """Returns the most recent memories."""
        if limit:
            return sorted(self._memories, key=lambda m: m.creation_timestamp, reverse=True)[:limit]
        else:
            return sorted(self._memories, key=lambda m: m.creation_timestamp, reverse=True)

    def find_memory_by_id(self, memory_id: str) -> Optional[MemoryObject]:
        """Finds a memory by its unique ID."""
        for mem in self._memories:
            if mem.id == memory_id:
                return mem
        return None

    def _calculate_retrieval_score(self, memory: MemoryObject, current_time: float, query_embedding: Optional[List[float]] = None) -> float:
        """Calculates a score based on recency, importance, and relevance."""
        # Recency: Exponential decay (higher score for recent)
        # Adjust decay factor as needed (e.g., smaller factor = faster decay)
        recency_score = 2**(-0.0001 * (current_time - memory.last_access_timestamp))

        # Importance: Directly use the score
        importance_score = memory.importance_score / 10.0 # Normalize to ~0-1

        # Relevance: Cosine similarity (requires embeddings)
        relevance_score = 0.0
        if query_embedding and memory.embedding_vector:
            # --- Placeholder for actual cosine similarity calculation ---
            # import numpy as np
            # query_vec = np.array(query_embedding)
            # mem_vec = np.array(memory.embedding_vector)
            # relevance_score = np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec))
            # relevance_score = (relevance_score + 1) / 2 # Normalize to 0-1
             pass # Keep as 0 without embeddings
        elif query_embedding is None:
             relevance_score = 0.5 # Default relevance if no query embedding

        # Combine scores (adjust weights as needed)
        total_score = (RECENCY_WEIGHT * recency_score +
                       IMPORTANCE_WEIGHT * importance_score +
                       RELEVANCE_WEIGHT * relevance_score)
        return total_score

    def retrieve_relevant_memories(self, query_text: Optional[str] = None, query_embedding: Optional[List[float]] = None, top_k: int = 10) -> List[MemoryObject]:
        """Retrieves the top_k most relevant memories based on scoring."""
        current_time = time.time()

        # --- Simple Keyword Matching (Fallback if no embeddings) ---
        # If we have query_text but no embeddings, do basic keyword matching
        # This is very rudimentary compared to semantic search.
        scored_memories : List[Tuple[float, MemoryObject]] = []
        if query_text and not query_embedding:
            query_words = set(query_text.lower().split())
            for mem in self._memories:
                 mem_words = set(mem.description.lower().split())
                 overlap = len(query_words.intersection(mem_words))
                 # Basic score: overlap + recency + importance (relevance is overlap here)
                 recency_score = 2**(-0.0001 * (current_time - mem.last_access_timestamp))
                 importance_score = mem.importance_score / 10.0
                 keyword_relevance = overlap / len(query_words) if query_words else 0
                 score = (RECENCY_WEIGHT * recency_score +
                          IMPORTANCE_WEIGHT * importance_score +
                          RELEVANCE_WEIGHT * keyword_relevance)
                 scored_memories.append((score, mem))

        # --- Embedding-based Scoring (Preferred if embeddings available) ---
        elif query_embedding:
            for mem in self._memories:
                score = self._calculate_retrieval_score(mem, current_time, query_embedding)
                scored_memories.append((score, mem))

        # --- No Query Scoring (Recency + Importance only) ---
        else:
             for mem in self._memories:
                score = self._calculate_retrieval_score(mem, current_time, None)
                scored_memories.append((score, mem))


        # Sort by score descending and take top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [mem for score, mem in scored_memories[:top_k]]

        # Update last access time for retrieved memories
        for mem in top_memories:
            mem.last_access_timestamp = current_time

        return top_memories

    def reflect(self, llm_interface) -> Optional[Reflection]:
        """
        Periodically synthesizes recent memories into higher-level reflections.
        (Simplified placeholder implementation)
        """
        # 1. Trigger condition (e.g., time-based or importance threshold)
        # Simplified: Just get recent memories for now
        recent_memories = self.retrieve_relevant_memories(top_k=20) # Get more memories for reflection context
        if not recent_memories:
            return None

        # 2. Generate questions (Simplified: Use a generic prompt)
        reflection_prompt = f"Based on these recent memories, what are 1-3 significant insights, patterns, or conclusions I should draw?\n\nMemories:\n"
        for mem in recent_memories:
             reflection_prompt += f"- {mem}\n"

        # 3. Get LLM synthesis
        synthesis_result = llm_interface.generate_simple_response(reflection_prompt) # Using a simpler call for now

        if synthesis_result:
             # 4. Create and store reflection
             new_reflection = Reflection(
                 description=f"Reflection: {synthesis_result}",
                 synthesized_memory_ids=[mem.id for mem in recent_memories]
             )
             self.add_memory(new_reflection)
             print(f"--- Agent reflected and added: {new_reflection.description[:100]}...")
             return new_reflection
        return None

    def get_memory_summary(self, max_memories=20, max_length=1000) -> str:
        """Provides a simple string summary of the most recent/important memories."""
        relevant_memories = self.retrieve_relevant_memories(top_k=max_memories)
        summary = ""
        for mem in relevant_memories:
            mem_str = f"- {mem}\n"
            if len(summary) + len(mem_str) < max_length:
                summary += mem_str
            else:
                break
        return summary if summary else "No relevant memories found."

    def check_consistency(self, claim: str) -> Tuple[str, List[MemoryObject]]:
        """
        Checks if a claim is consistent, contradictory, or unknown based on memory.
        Returns consistency status ('consistent', 'contradictory', 'unknown') and supporting/contradicting memories.
        (Simplified: Basic keyword check for contradiction)
        """
        relevant_memories = self.retrieve_relevant_memories(query_text=claim, top_k=5)
        claim_lower = claim.lower()
        status = "unknown"
        supporting_evidence = []

        # Rudimentary check: Look for explicit contradictions or strong support.
        # A real system would use LLM reasoning here.
        for mem in relevant_memories:
            mem_lower = mem.description.lower()
            # Simple contradiction check (e.g., "grandmother is alive" vs "grandmother died")
            # This is highly fragile and needs LLM nuance.
            if ("not " + claim_lower in mem_lower) or \
               (claim_lower.startswith("is ") and "is not " + claim_lower[3:] in mem_lower) or \
               (claim_lower.startswith("has ") and "does not have " + claim_lower[4:] in mem_lower) or \
               (claim_lower.startswith("i have a ") and f"i do not have a {claim_lower[9:]}" in mem_lower) or \
               (claim_lower.startswith("grandmother is sick") and "grandmother died" in mem_lower) or \
               (claim_lower.startswith("grandmother is sick") and "i have no grandmother" in mem_lower):
                status = "contradictory"
                supporting_evidence.append(mem)
                break # Found contradiction, stop searching
            # Simple consistency check (needs improvement)
            if claim_lower in mem_lower:
                 status = "consistent" # Tentative consistency
                 supporting_evidence.append(mem)
                 # Don't break, maybe find contradiction later

        # If still unknown after checking relevant, assume unknown
        # If we found only potential support, call it consistent for now
        if status == "unknown" and supporting_evidence:
             status = "consistent"

        return status, supporting_evidence

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the memory stream."""
        return {
            "memories": [mem.to_dict() for mem in self._memories],
            "reflection_threshold": self._reflection_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryStream':
        """Deserializes the memory stream."""
        stream = cls()
        stream._reflection_threshold = data.get("reflection_threshold", 100)
        memory_data_list = data.get("memories", [])
        stream._memories = [MemoryObject.from_dict(mem_data) for mem_data in memory_data_list]
        # Sort memories by timestamp after loading (optional but good practice)
        stream._memories.sort(key=lambda m: m.creation_timestamp)
        return stream
# agent_framework/core/memory.py
import time
import datetime
import uuid
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union # Use list, dict etc.

logger = logging.getLogger(__name__)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2' # Common choice, relatively small
    # Consider other models based on performance/needs: e.g., 'multi-qa-MiniLM-L6-cos-v1'
    logger.info(f"Loading sentence transformer model: {SENTENCE_TRANSFORMER_MODEL}...")
    embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    # Get embedding dimension
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logger.info(f"Sentence transformer model loaded. Embedding dimension: {embedding_dim}")
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS or sentence-transformers not found. Falling back to basic keyword retrieval.")
    logger.warning("Install with: pip install sentence-transformers faiss-cpu")
    FAISS_AVAILABLE = False
    embedding_model = None
    embedding_dim = None

# --- Constants ---
DEFAULT_IMPORTANCE = 5.0 # On a scale of 1-10
RECENCY_WEIGHT = 0.1 # Lower weight for recency now?
IMPORTANCE_WEIGHT = 1.0
RELEVANCE_WEIGHT = 1.5 # Higher weight for semantic relevance

class MemoryObject:
    """Base class for different types of memories."""
    def __init__(self, description: str, importance: float = DEFAULT_IMPORTANCE, timestamp: Optional[float] = None):
        self.id: str = str(uuid.uuid4())
        self.description: str = description
        self.creation_timestamp: float = timestamp if timestamp is not None else time.time()
        self.last_access_timestamp: float = self.creation_timestamp
        self.importance_score: float = max(1.0, min(10.0, importance))
        # Embedding vector - will be calculated by MemoryStream
        self.embedding: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        time_str = datetime.datetime.fromtimestamp(self.creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        access_time_str = datetime.datetime.fromtimestamp(self.last_access_timestamp).strftime('%H:%M:%S')
        return f"[{time_str} (Acc: {access_time_str}) Imp: {self.importance_score:.1f}] {self.description}"

    def to_dict(self) -> dict[str, Any]:
        """Serializes basic memory info to a dictionary."""
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "description": self.description,
            "creation_timestamp": self.creation_timestamp,
            "last_access_timestamp": self.last_access_timestamp,
            "importance_score": self.importance_score,
            # Exclude embedding from basic YAML serialization
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MemoryObject':
        """Deserializes from a dictionary. Embedding needs separate handling."""
        mem_type = data.get("type", "MemoryObject")
        mem_class = globals().get(mem_type, MemoryObject) # Get class object by name

        # Handle potential missing keys gracefully during load
        mem = mem_class(
            description=data.get("description", ""),
            importance=data.get("importance_score", DEFAULT_IMPORTANCE),
            timestamp=data.get("creation_timestamp") # Pass None if missing
        )
        mem.id = data.get("id", mem.id) # Restore ID if present
        mem.last_access_timestamp = data.get("last_access_timestamp", mem.creation_timestamp)
        # Embedding must be recalculated or loaded separately after object creation
        mem.embedding = None
        return mem

# --- Subclasses (Observation, Reflection, GeneratedFact) remain the same ---
class Observation(MemoryObject): pass
class Reflection(MemoryObject): # Simplified - no explicit link tracking in basic dict
    def __init__(self, description: str, importance: float = DEFAULT_IMPORTANCE + 2.0, timestamp: Optional[float] = None):
         super().__init__(description, importance, timestamp)
class GeneratedFact(MemoryObject): # Simplified
     def __init__(self, description: str, source_claim: str = "", importance: float = DEFAULT_IMPORTANCE - 1.0, timestamp: Optional[float] = None):
          super().__init__(description, importance, timestamp)
          self.source_claim = source_claim # Still useful to know origin


class MemoryStream:
    """Manages the agent's collection of memories with vector-based retrieval."""
    def __init__(self):
        self._memories: dict[str, MemoryObject] = {} # Store by ID for quick lookup
        self.faiss_index: Optional[faiss.Index] = None
        self.index_to_id: list[str] = [] # Map faiss index position to memory ID
        if FAISS_AVAILABLE and embedding_dim:
            # Using IndexFlatL2 - simple Euclidean distance. IndexFlatIP (Inner Product) is better for cosine similarity with normalized vectors.
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"FAISS index initialized with dimension {embedding_dim}.")
        else:
            logger.warning("FAISS index not available. Retrieval will be limited.")

    def _calculate_embedding(self, text: str) -> Optional[np.ndarray]:
        """ Calculates embedding using the loaded sentence transformer model. """
        if embedding_model:
            try:
                # Encode expects list, ensure text is not empty
                if not text: return None
                embedding = embedding_model.encode([text])[0]
                # Normalize embedding for cosine similarity (if using IndexFlatIP later)
                # faiss.normalize_L2(embedding.reshape(1, -1))
                return embedding
            except Exception as e:
                logger.error(f"Failed to calculate embedding for text: '{text[:50]}...': {e}")
                return None
        return None

    def add_memory(self, memory: MemoryObject):
        """Adds a new memory, calculates its embedding, and adds to FAISS index."""
        if memory.id in self._memories:
            logger.warning(f"Attempting to add memory with duplicate ID: {memory.id}")
            return # Avoid duplicates

        # Calculate and store embedding
        memory.embedding = self._calculate_embedding(memory.description)

        # Add to main storage
        self._memories[memory.id] = memory
        logger.debug(f"Added memory {memory.id} ('{memory.description[:50]}...')")

        # Add embedding to FAISS index if available and valid
        if self.faiss_index is not None and memory.embedding is not None:
            try:
                # FAISS expects a 2D numpy array (num_vectors x dimension)
                vector = np.array([memory.embedding]).astype('float32')
                self.faiss_index.add(vector)
                self.index_to_id.append(memory.id) # Track ID corresponding to the new index position
                logger.debug(f"Added embedding for memory {memory.id} to FAISS index. Index size: {self.faiss_index.ntotal}")
            except Exception as e:
                logger.error(f"Failed to add embedding for memory {memory.id} to FAISS index: {e}")

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryObject]:
        """Retrieves a memory by its ID."""
        return self._memories.get(memory_id)

    def retrieve_relevant_memories(self, query_text: str, top_k: int = 10) -> list[MemoryObject]:
        """Retrieves the top_k most relevant memories using vector similarity and scoring."""
        if not query_text: return []
        logger.debug(f"Retrieving top {top_k} relevant memories for query: '{query_text[:100]}...'")

        query_embedding = self._calculate_embedding(query_text)
        if query_embedding is None:
             logger.warning("Could not generate query embedding. Cannot perform vector search.")
             # Fallback to basic retrieval? Or return empty? Let's return empty for now.
             return []

        # --- FAISS Search (if available) ---
        retrieved_ids: list[str] = []
        distances: list[float] = []
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                query_vector = np.array([query_embedding]).astype('float32')
                # Search returns distances (L2 squared) and indices
                k_search = min(top_k * 3, self.faiss_index.ntotal) # Retrieve more initially for re-ranking
                distances_sq, indices = self.faiss_index.search(query_vector, k_search)

                if indices.size > 0 and distances_sq.size > 0:
                     retrieved_raw = []
                     for i, idx in enumerate(indices[0]):
                          if idx != -1: # FAISS uses -1 for no result / padding
                               mem_id = self.index_to_id[idx]
                               dist_sq = distances_sq[0][i]
                               retrieved_raw.append({'id': mem_id, 'distance_sq': dist_sq})
                     logger.debug(f"FAISS search returned {len(retrieved_raw)} potential candidates.")

                     # Convert L2 distance squared to a similarity score (0-1, higher is better)
                     # Simple inversion for now, more sophisticated methods exist
                     max_dist_sq = max(r['distance_sq'] for r in retrieved_raw) if retrieved_raw else 1.0
                     for r in retrieved_raw:
                          similarity = 1.0 / (1.0 + r['distance_sq']) # Simple inverse relation
                          r['relevance_score'] = similarity

                     # Store relevance scores with IDs for re-ranking
                     relevance_map = {r['id']: r['relevance_score'] for r in retrieved_raw}
                     retrieved_ids = list(relevance_map.keys())

            except Exception as e:
                 logger.error(f"FAISS search failed: {e}", exc_info=True)
                 # Fallback to iterating all memories if search fails? Costly.
                 retrieved_ids = list(self._memories.keys()) # Consider all if search fails
                 relevance_map = {mid: 0.5 for mid in retrieved_ids} # Default relevance

        else:
            # No FAISS - retrieve all memory IDs for scoring (inefficient)
            logger.warning("FAISS index empty or unavailable. Scoring all memories.")
            retrieved_ids = list(self._memories.keys())
            # Cannot calculate relevance without index, use default
            relevance_map = {mid: 0.5 for mid in retrieved_ids}


        # --- Re-ranking based on Recency, Importance, Relevance ---
        current_time = time.time()
        scored_memories : list[tuple[float, MemoryObject]] = []

        for mem_id in retrieved_ids:
            memory = self._memories.get(mem_id)
            if not memory: continue

            # Recency score (exponential decay)
            recency_score = 2**(-0.0001 * (current_time - memory.last_access_timestamp))
            # Importance score (normalized)
            importance_score = memory.importance_score / 10.0
            # Relevance score (from FAISS or default)
            relevance_score = relevance_map.get(mem_id, 0.0) # Default to 0 if somehow missing

            # Combine scores
            total_score = (RECENCY_WEIGHT * recency_score +
                           IMPORTANCE_WEIGHT * importance_score +
                           RELEVANCE_WEIGHT * relevance_score)
            scored_memories.append((total_score, memory))

        # Sort by score descending and take top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [mem for score, mem in scored_memories[:top_k]]
        logger.info(f"Retrieved {len(top_memories)} memories after re-ranking.")
        logger.debug(f"Top relevant memory IDs: {[m.id for m in top_memories]}")

        # Update last access time for retrieved memories
        for mem in top_memories:
            memory.last_access_timestamp = current_time

        return top_memories

    def get_memory_summary(self, max_memories=20, max_length=1000) -> str:
        """Provides a simple string summary of the most RECENT memories (not relevance based)."""
        # Sort by creation time to get most recent
        sorted_memories = sorted(self._memories.values(), key=lambda m: m.creation_timestamp, reverse=True)
        summary_mems = sorted_memories[:max_memories]

        summary = ""
        for mem in summary_mems:
            mem_str = f"- {mem}\n" # Use __repr__ for timestamp/importance info
            if len(summary) + len(mem_str) < max_length:
                summary += mem_str
            else:
                summary += "- ... (truncated due to length)\n"
                break
        return summary if summary else "No memories found."

    # --- Persistence (Simplified - Index needs proper handling) ---
    def to_dict(self) -> dict[str, Any]:
        """Serializes the memory stream (excluding FAISS index)."""
        return {
            "memories": {mem_id: mem.to_dict() for mem_id, mem in self._memories.items()},
            # FAISS index needs separate saving/loading (e.g., faiss.write_index / faiss.read_index)
            # index_to_id map also needs saving
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MemoryStream':
        """Deserializes the memory stream and rebuilds index if possible."""
        stream = cls() # Initializes empty index
        memory_dict_data = data.get("memories", {})
        loaded_memories = []
        for mem_id, mem_data in memory_dict_data.items():
            if isinstance(mem_data, dict): # Basic check
                 try:
                      mem = MemoryObject.from_dict(mem_data)
                      # Recalculate embedding and add to stream/index
                      # stream.add_memory(mem) # This recalculates embedding and adds to index
                      # Optimization: Could store embeddings separately and reload them if needed
                      loaded_memories.append(mem)
                 except Exception as e:
                      logger.error(f"Failed to load memory data for ID {mem_id}: {e} - Data: {mem_data}")

        # Sort by timestamp before adding to ensure index order matches potential loaded ID map
        loaded_memories.sort(key=lambda m: m.creation_timestamp)
        logger.info(f"Loaded {len(loaded_memories)} memories from dict. Rebuilding FAISS index...")
        count = 0
        for mem in loaded_memories:
            stream.add_memory(mem) # This recalculates embedding and adds to index
            count += 1
        logger.info(f"FAISS index rebuild complete. Final size: {stream.faiss_index.ntotal if stream.faiss_index else 'N/A'}")

        # TODO: Implement loading faiss index and index_to_id map from separate files if saved previously
        return stream
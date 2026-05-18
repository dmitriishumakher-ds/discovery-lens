# sentence_transformers is the library. SentenceTransformer is the class we use to load and run the model.
from sentence_transformers import SentenceTransformer
import numpy as np

# Lazy-loaded embedding model. @st.cache_resource keeps the loaded model in
# memory across Streamlit reruns — without it, every user interaction would
# reload the ~80MB all-MiniLM-L6-v2 model from disk. Falls back to a plain
# module-level singleton when Streamlit isn't available (e.g. when this module
# is imported from a notebook or CLI test).
try:
    import streamlit as st
    @st.cache_resource
    def _get_model():
        """Load the all-MiniLM-L6-v2 embedding model once per Streamlit session."""
        return SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _model = None
    def _get_model():
        """Load the all-MiniLM-L6-v2 embedding model once per Python process."""
        global _model
        if _model is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model

def embed_chunks(chunks: list[dict]) -> np.ndarray: #Converts a list of text chunks into a 2D array of embeddings.
    model = _get_model() # Load the model (or reuse it if already loaded).
    texts = [chunk["text"] for chunk in chunks]  # Pull out just the text from each chunk dict, keeping the same order. This gives us a plain list of strings, which is what model.encode() expects.
    embeddings = model.encode(texts, show_progress_bar=False) # The model converts each string into a list of 384 numbers. model.encode() takes a list of strings and returns a 2D array: one row per string, 384 columns per row. show_progress_bar=False keeps Streamlit's output clean.
    return np.array(embeddings) # Wrap in np.array() to guarantee the return type is always np.ndarray.


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text string to a 384-dim embedding vector.
    Used for embedding the validated goal so it can be compared against
    cluster chunk embeddings for goal_relevance scoring (D-03).
    """
    model = _get_model()  # Reuse the singleton model — same instance as embed_chunks.
    embedding = model.encode([text], show_progress_bar=False)  # encode() expects a list, wrap the single string.
    return np.array(embedding[0])  # encode returns 2D (1, 384) — index [0] gives us 1D (384,) which is what goal_embedding expects.

# ── Quick local test ──
# This block only runs when you execute this file directly. It is never run when the app imports this module, so it won't slow anything down. that's already guaranteed by the if __name__ == "__main__": guard.
# When you run python embedder.py directly in the terminal → __name__ equals "__main__" → the block runs. When Streamlit imports embedder.py as a module (from pipeline.embedder import embed_chunks) → __name__ equals "embedder" → the condition is false, the block is skipped entirely
# This is just to verify the module works.
if __name__ == "__main__":
    mock_chunks = [
        {"chunk_id": "test_001", "text": "Users struggle to find past orders.", "filename": "test.txt", "source_type": "review"},
        {"chunk_id": "test_002", "text": "The checkout process is slow and confusing.", "filename": "test.txt", "source_type": "review"},
        {"chunk_id": "test_003", "text": "Customer support never replies to tickets.", "filename": "test.txt", "source_type": "ticket"},
        {"chunk_id": "test_004", "text": "I cannot find where to change my payment method.", "filename": "test.txt", "source_type": "review"},
        {"chunk_id": "test_005", "text": "The app crashes every time I open my profile.", "filename": "test.txt", "source_type": "review"},
    ]

    result = embed_chunks(mock_chunks)

    print("Shape:", result.shape) # Should print: (5, 384) — 5 chunks, 384 numbers each.
    print("dtype:", result.dtype) # Should print: float32 — the number type the model uses.
    print("All good — embedder.py is working correctly.")

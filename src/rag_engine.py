import logging
import torch
import textwrap
log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    util = None

class EvidenceRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        log.info("[RAG] Loading embedding model (%s)...", model_name)
        # This is a small, fast, and accurate model for semantic search
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for RAG. Install with: pip install sentence-transformers")
        self.embedder = SentenceTransformer(model_name)

        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder.to(self.device)

        self.chunks = []
        self.corpus_embeddings = None

    def ingest_transcript(self, transcript_text, chunk_size=2000, overlap=400):
        """
        Breaks the transcript into overlapping chunks and embeds them.
        """
        print("[RAG] Chunking transcript...")

        # Simple sliding window chunking
        # We want to keep timestamps/speakers intact if possible,
        # but pure character chunking is robust enough for now.
        self.chunks = []

        # Split by lines first to respect speaker turns
        lines = transcript_text.split('\n')
        current_chunk = []
        current_len = 0

        for line in lines:
            line = line.strip()
            if not line: continue

            current_chunk.append(line)
            current_len += len(line)

            if current_len >= chunk_size:
                # Join and add
                chunk_text = "\n".join(current_chunk)
                self.chunks.append(chunk_text)

                # Create overlap: keep the last N lines that fit into 'overlap' size
                new_start = []
                overlap_len = 0
                for prev_line in reversed(current_chunk):
                    if overlap_len + len(prev_line) > overlap:
                        break
                    new_start.insert(0, prev_line)
                    overlap_len += len(prev_line)

                current_chunk = new_start
                current_len = overlap_len

        if current_chunk:
            self.chunks.append("\n".join(current_chunk))

        print(f"[RAG] Created {len(self.chunks)} chunks. Embedding...")

        # Create Embeddings (The heavy lifting)
        self.corpus_embeddings = self.embedder.encode(
            self.chunks,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.device
        )
        print("[RAG] Indexing complete.")

    def search(self, query, top_k=10):
        """
        Finds the top_k most relevant chunks for the query.
        """
        if self.corpus_embeddings is None:
            return []

        # Embed the query
        query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)

        # Semantic Search (Cosine Similarity)
        # util.semantic_search is highly optimized in PyTorch
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]

        results = []
        for hit in hits:
            score = hit['score']
            chunk_id = hit['corpus_id']
            text = self.chunks[chunk_id]
            results.append({"text": text, "score": score})

        return results


def generate_rag_prompt(question, context_chunks, role_instruction=None):
    """
    Constructs the prompt with a dynamic persona.
    """
    joined_context = "\n---\n".join([c['text'] for c in context_chunks])

    # Default fallback if nothing provided
    if not role_instruction:
        role_instruction = "You are a helpful assistant analyzing a transcript."

    prompt = f"""
{role_instruction}
Answer the user's question based ONLY on the provided context chunks below. 
If the answer is not in the context, say "I cannot find that information in the transcript."

CONTEXT FROM TRANSCRIPT:
{joined_context}

USER QUESTION: 
{question}

ANSWER:
"""
    return prompt
import pickle
from pathlib import Path
import sys
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from data_loader import load_documents
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
# ── Local LLM setup ──────────────────────────────────────────────────────────
# Uncomment this block to use local Mistral-7B instead of the Mistral API.
# Requires a GPU with ~6GB VRAM (or ~12GB RAM for CPU-only, very slow).
# _quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )
# local_llm = HuggingFaceLLM(
#     model_name="mistralai/Mistral-7B-Instruct-v0.1",
#     tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
#     context_window=4096,
#     max_new_tokens=512,
#     generate_kwargs={"temperature": 0.2, "do_sample": True},
#     device_map="auto",
#     model_kwargs={
#         "torch_dtype": torch.float16,
#         "quantization_config": _quantization_config,
#     },
# )
# ────────────────────────────────────────────────────────────────────────────
# Placeholder so app.py doesn't crash when local LLM is commented out
local_llm = None
INDEX_CACHE = Path("./DATA/storage.pkl")
def build_embed_model():
    """Return the HuggingFace embedding model."""
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
def build_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Build (or load from cache) the vector index.
    Set force_rebuild=True to re-index after changing the CSV."""
    embed_model = build_embed_model()
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 0
    # We do NOT set Settings.llm here — explanations use the Mistral API in app.py.
    Settings.llm = None
    if INDEX_CACHE.exists() and not force_rebuild:
        print("Loading index from cache...")
        with open(INDEX_CACHE, "rb") as f:
            index = pickle.load(f)
        print("Index loaded successfully!")
        return index
    print("Building index from scratch (this may take a few minutes)...")
    documents = load_documents()
    splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )
    # Show progress during index building
    print(f"Creating document nodes (splitting into chunks)...")
    sys.stdout.flush()
    index = VectorStoreIndex.from_documents(
        documents, 
        transformations=[splitter],
        show_progress=True
    )
    # Cache to disk so subsequent starts are fast
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print("Saving index to cache...")
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump(index, f)
    print("Index built and cached successfully!")
    return index

def get_retriever(index: VectorStoreIndex, top_k: int = 20):
    return index.as_retriever(
        similarity_top_k=top_k,
        vector_store_query_mode="mmr",
    )
def retrieve(index: VectorStoreIndex, query: str, top_k: int = 20, threshold: float = 0.04):
    """Retrieve relevant questions and filter by similarity score."""
    retriever = get_retriever(index, top_k=top_k * 3)  # over-fetch then filter
    results = retriever.retrieve(query)
    filtered = [r for r in results if r.score >= threshold]
    return filtered[:top_k]
def format_results(results: list) -> list[dict]:
    """Convert retrieval results to plain dicts for JSON serialisation."""
    out = []
    for r in results:
        node = r.node
        meta = node.metadata
        out.append(
            {
                "question": node.get_content(),
                "score": round(float(r.score), 4),
                "topics": meta.get("topics", ""),
                "answer": meta.get("answer", ""),
                "answer_description": meta.get("answer_description", ""),
                "exam": meta.get("exam", ""),
                "year": meta.get("year", ""),
                "subject": meta.get("subject", ""),
                "image": meta.get("image", ""),
            }
        )
    return out

def build_llm_context(results: list[dict], max_questions: int = 15) -> str:
    """Build a context string from retrieved questions for the LLM prompt."""
    lines = []
    for i, r in enumerate(results[:max_questions], 1):
        lines.append(
            f"Question {{i}}:\n{{r['question']}}\n"
            f"Exam: {{r['exam']}} | Year: {{r['year']}} | Topic: {{r['topics']}}\n"
            f"Answer: {{r['answer']}}\n"
            f"Explanation: {{r['answer_description']}}\n"
            f"{{'-' * 40}}"
        )
    return "\n".join(lines)
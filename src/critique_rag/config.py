"""Central configuration: paths, model IDs, and loop/retrieval knobs.

`.env` is loaded here so `ANTHROPIC_API_KEY` is available to the Anthropic SDK
(see `llm.py`). Nothing here makes network calls.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths (project root is two levels up from this file: src/critique_rag/) ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
QUESTIONS_FILE = DATA_DIR / "questions.json"
CHROMA_DIR = ROOT / "chroma_db"
TRACES_DIR = ROOT / "traces"
COLLECTION_NAME = "docs"

# --- Embeddings (free / local) ---
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 4

# --- Self-critique loop ---
MAX_RETRIES = 2  # hard cap on reformulate->retrieve cycles

# --- Models (two distinct roles; see README) ---
GENERATOR_MODEL = "claude-sonnet-4-6"   # writes the answer
CRITIC_MODEL = "claude-haiku-4-5"       # in-loop grounding critic (separate call)
REFORMULATE_MODEL = "claude-sonnet-4-6"  # rewrites the query on rejection
# Eval-only judge. Deliberately a DIFFERENT model from the in-loop critic so the
# critic-on path is not trivially graded as grounded by the same model that
# selected it.
JUDGE_MODEL = "claude-sonnet-4-6"

# --- Token caps (answers/verdicts are short) ---
GEN_MAX_TOKENS = 1024
CRITIC_MAX_TOKENS = 1024
REFORMULATE_MAX_TOKENS = 256

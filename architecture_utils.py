import os
import logging
from dotenv import load_dotenv

def load_env_variables():
    """
    Centralized environment loader for AetherScribe Research Lab.
    Ensures critical infrastructure keys are present for the local session.
    """
    load_dotenv()

    return {
        "api_key": None,
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "db_path": os.getenv("DATABASE_URL", "sqlite:///./aetherscribe_lab.db"),
        "llm_provider": os.getenv("NARRATIVE_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "local")),
        "local_runtime": os.getenv("LOCAL_LLM_RUNTIME", ""),
        "local_model": os.getenv("LOCAL_LLM_MODEL_NAME", "OLMoE-1B-7B-0125-Instruct.Q4_K_M"),
        "jev_enabled": os.getenv("JEV_ENABLED", "false").lower() == "true",
        "memory_enabled": os.getenv("NARRATIVE_MEMORY_ENABLED", "true").lower() == "true",
        "memory_retrieval_limit": int(os.getenv("NARRATIVE_MEMORY_RETRIEVAL_LIMIT", "12")),
        "memory_context_budget": int(os.getenv("NARRATIVE_MEMORY_CONTEXT_TOKEN_BUDGET", "800")),
        "memory_min_confidence": float(os.getenv("NARRATIVE_MEMORY_MIN_CONFIDENCE", "0.70")),
        "planner_enabled": os.getenv("NARRATIVE_PLANNER_ENABLED", "true").lower() == "true",
        "secret_ledger_enabled": os.getenv("NARRATIVE_SECRET_LEDGER_ENABLED", "true").lower() == "true",
        "reveal_gate_enabled": os.getenv("NARRATIVE_REVEAL_GATE_ENABLED", "true").lower() == "true",
        "secret_strict_validation": os.getenv("NARRATIVE_SECRET_STRICT_VALIDATION", "true").lower() == "true",
    }

def setup_research_logging():
    """
    Configures standardized logging for research audit trails.
    Optimized for trajectory tracing and quantum state analysis.
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/research_audit.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("AetherScribeLab")
    logger.info("Research Logging Subsystem Initialized (v5.4.2).")
    return logger

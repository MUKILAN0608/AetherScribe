import os
import logging
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent


def load_env_variables():
    """
    Centralized environment loader for AetherScribe Research Lab.
    Ensures critical infrastructure keys are present for the local session.
    """
    load_dotenv(_ROOT / ".env")

    api_key = os.getenv("GOOGLE_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        logging.warning("GOOGLE_API_KEY not detected. Add GOOGLE_API_KEY=... to .env in the project root.")

    return {
        "api_key": api_key or None,
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "db_path": os.getenv("DATABASE_URL", "sqlite:///./aetherscribe_lab.db")
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

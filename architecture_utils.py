import os
import logging
from dotenv import load_dotenv

def load_env_variables():
    """
    Centralized environment loader for AetherScribe Research Lab.
    Ensures critical infrastructure keys are present for the local session.
    """
    load_dotenv()

    # Priority check for Google API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.warning("GOOGLE_API_KEY not detected. System will revert to MOCK narrative mode.")

    return {
        "api_key": api_key,
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

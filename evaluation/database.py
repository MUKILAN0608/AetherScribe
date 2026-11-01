from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import json
import os

Base = declarative_base()


def utc_now_naive():
    """Return UTC for SQLite DateTime columns without deprecated APIs."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Episode(Base):
    __tablename__ = 'episodes'

    id = Column(Integer, primary_key=True)
    timestamp = Column(String)
    genre = Column(String)
    premise = Column(Text)
    total_tension = Column(Float)
    full_log = Column(Text)  # Stored as JSON string
    story_id = Column(String, nullable=True, index=True)
    system_state = Column(Text, nullable=True)


class SceneHistoryModel(Base):
    __tablename__ = "scene_history"

    id = Column(Integer, primary_key=True)
    story_id = Column(String, nullable=False, index=True)
    episode_id = Column(String, nullable=False, index=True)
    scene_index = Column(Integer, nullable=False)
    story = Column(Text, nullable=False)
    summary = Column(Text, nullable=False, default="")
    outcome = Column(Text, nullable=False, default="")
    opening = Column(Text, nullable=False, default="")
    ending = Column(Text, nullable=False, default="")
    beat = Column(Text, nullable=False, default="")
    objective = Column(Text, nullable=False, default="")
    accepted = Column(Integer, nullable=False, default=1)
    diagnostics_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, nullable=False)


class MemoryRecordModel(Base):
    __tablename__ = "memory_records"

    id = Column(Integer, primary_key=True)
    story_id = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    origin = Column(String, nullable=False, default="unknown")
    episode_id = Column(String, nullable=True)
    scene_index = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="active", index=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supersedes_id = Column(Integer, nullable=True)


class SecretRecordModel(Base):
    __tablename__ = "secrets"

    id = Column(Integer, primary_key=True)
    story_id = Column(String, nullable=False, index=True)
    secret_id = Column(String, nullable=False)
    protected_value = Column(Text, nullable=False)
    aliases_json = Column(Text, nullable=False, default="")
    min_scene = Column(Integer, nullable=True)
    required_event = Column(String, nullable=True)
    character_knowledge_json = Column(Text, nullable=False, default="")
    prerequisite_secret = Column(String, nullable=True)
    explicit_instruction = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="hidden")
    revealed_scene = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utc_now_naive)
    revealed_at = Column(DateTime, nullable=True)


class SecretRevealAuditModel(Base):
    __tablename__ = "secret_reveal_audit"

    id = Column(Integer, primary_key=True)
    story_id = Column(String, nullable=False, index=True)
    secret_id = Column(String, nullable=False)
    authorized = Column(Integer, nullable=False)
    actor = Column(String, nullable=False)
    scene_index = Column(Integer, nullable=False)
    reasons_json = Column(Text, nullable=False, default="[]")
    timestamp = Column(DateTime, nullable=False, default=utc_now_naive)

class AetherDatabase:
    """
    Research-grade persistence layer using SQLAlchemy.
    Supports PostgreSQL via DATABASE_URL environment variable.
    """
    def __init__(self, db_url=None):
        # Default to a local SQLite file for the Research Lab session
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///./aetherscribe_lab.db")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self._migrate_episode_columns()
        self._migrate_secrets()
        self.Session = sessionmaker(bind=self.engine)
        from memory.repository import MemoryRepository
        self.memory = MemoryRepository(self.Session, MemoryRecordModel)
        self.memory_repository = self.memory
        from secret_loader import load_secret_module
        SecretRepository = load_secret_module("repository").SecretRepository
        self.secrets = SecretRepository(self.Session, SecretRecordModel)
        self.secret_repository = self.secrets
        from continuity.history import SceneHistoryRepository
        self.scene_history = SceneHistoryRepository(self.Session, SceneHistoryModel)

    def _migrate_episode_columns(self):
        """Add nullable columns required by newer schemas without rewriting rows."""
        columns = {column["name"] for column in inspect(self.engine).get_columns("episodes")}
        if "story_id" not in columns:
            with self.engine.begin() as connection:
                connection.execute(text("ALTER TABLE episodes ADD COLUMN story_id VARCHAR"))
            columns.add("story_id")
        if "system_state" not in columns:
            with self.engine.begin() as connection:
                connection.execute(text("ALTER TABLE episodes ADD COLUMN system_state TEXT"))

    def _migrate_secrets(self):
        """Create/upgrade secret storage without dropping existing data."""
        inspector = inspect(self.engine)
        if not inspector.has_table("secrets"):
            Base.metadata.tables["secrets"].create(self.engine, checkfirst=True)
        else:
            columns = {column["name"] for column in inspector.get_columns("secrets")}
            additions = {
                "aliases_json": "TEXT NOT NULL DEFAULT ''",
                "min_scene": "INTEGER",
                "required_event": "VARCHAR",
                "character_knowledge_json": "TEXT NOT NULL DEFAULT ''",
                "prerequisite_secret": "VARCHAR",
                "explicit_instruction": "TEXT",
                "status": "VARCHAR NOT NULL DEFAULT 'hidden'",
                "revealed_scene": "INTEGER",
                "created_at": "DATETIME",
                "revealed_at": "DATETIME",
            }
            missing = {name: declaration for name, declaration in additions.items()
                       if name not in columns}
            if missing:
                with self.engine.begin() as connection:
                    for name, declaration in missing.items():
                        connection.execute(text("ALTER TABLE secrets ADD COLUMN "
                                                + name + " " + declaration))
        Base.metadata.tables["secret_reveal_audit"].create(self.engine, checkfirst=True)

    def add_memory(self, record):
        return self.memory.add(record)

    def retrieve_memory(self, story_id, **kwargs):
        return self.memory.retrieve(story_id, **kwargs)

    save_memory = add_memory
    get_memory_context = retrieve_memory

    def save_episode(self, genre, premise, total_tension, log, story_id=None, system_state=None):
        session = self.Session()
        try:
            episode = Episode(
                timestamp=datetime.now().isoformat(),
                genre=genre,
                premise=premise,
                total_tension=total_tension,
                full_log=json.dumps(log),
                story_id=story_id,
                system_state=json.dumps(system_state) if system_state is not None else None,
            )
            session.add(episode)
            session.commit()
            return episode.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_episode_system_state(self, episode_id):
        session = self.Session()
        try:
            episode = session.query(Episode).filter_by(id=episode_id).one_or_none()
            return json.loads(episode.system_state) if episode and episode.system_state else None
        finally:
            session.close()

    def get_recent_episodes(self, limit=5):
        session = self.Session()
        try:
            episodes = session.query(Episode).order_by(Episode.id.desc()).limit(limit).all()
            # Convert SQLAlchemy objects to list of tuples/dicts to match previous interface
            # or return objects if the consumer is updated.
            # Returning list of tuples to maintain compatibility with existing consumers for now:
            # (id, timestamp, genre, premise, total_tension, full_log)
            return [
                (e.id, e.timestamp, e.genre, e.premise, e.total_tension, e.full_log)
                for e in episodes
            ]
        finally:
            session.close()

    def list_scene_history(self, story_id, limit=8):
        return self.scene_history.list(story_id, limit)

    def save_scene_history(self, **kwargs):
        return self.scene_history.add(**kwargs)

    def add_secret(self, secret):
        return self.secrets.add(secret)

    def get_secret(self, story_id, secret_id):
        return self.secrets.get(story_id, secret_id)

    def list_secrets(self, story_id):
        return self.secrets.list(story_id)

    def record_reveal_audit(self, audit):
        session = self.Session()
        try:
            row = SecretRevealAuditModel(
                story_id=audit.story_id, secret_id=audit.secret_id,
                authorized=int(audit.authorized), actor=audit.actor,
                scene_index=audit.scene_index, reasons_json=json.dumps(audit.reasons),
                timestamp=audit.timestamp.replace(tzinfo=None),
            )
            session.add(row)
            session.commit()
            return row.id
        finally:
            session.close()

    def authorize_secret_reveal(self, story_id, secret_id, context, actor="system"):
        """Atomically authorize, persist, and audit a story-scoped reveal."""
        from story_secrets.gate import RevealGate
        from story_secrets.models import SecretStatus
        secret = self.get_secret(story_id, secret_id)
        if secret is None:
            raise KeyError("secret not found")
        revealed, audit = RevealGate().authorize(secret, context, actor)
        session = self.Session()
        try:
            row = session.query(SecretRecordModel).filter_by(
                story_id=story_id, secret_id=secret_id).one()
            row.status = SecretStatus.REVEALED.value
            row.revealed_scene = revealed.revealed_scene
            row.revealed_at = audit.timestamp.replace(tzinfo=None)
            session.add(SecretRevealAuditModel(
                story_id=audit.story_id, secret_id=audit.secret_id,
                authorized=1, actor=audit.actor, scene_index=audit.scene_index,
                reasons_json=json.dumps(audit.reasons),
                timestamp=audit.timestamp.replace(tzinfo=None)))
            session.commit()
            return revealed, audit
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

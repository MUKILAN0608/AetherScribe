from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

Base = declarative_base()

class Episode(Base):
    __tablename__ = 'episodes'

    id = Column(Integer, primary_key=True)
    timestamp = Column(String)
    genre = Column(String)
    premise = Column(Text)
    total_tension = Column(Float)
    full_log = Column(Text)  # Stored as JSON string

class AetherDatabase:
    """
    Research-grade persistence layer using SQLAlchemy.
    Supports PostgreSQL via DATABASE_URL environment variable.
    """
    def __init__(self):
        # Default to a local SQLite file for the Research Lab session
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///./aetherscribe_lab.db")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_episode(self, genre, premise, total_tension, log):
        session = self.Session()
        try:
            episode = Episode(
                timestamp=datetime.now().isoformat(),
                genre=genre,
                premise=premise,
                total_tension=total_tension,
                full_log=json.dumps(log)
            )
            session.add(episode)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
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

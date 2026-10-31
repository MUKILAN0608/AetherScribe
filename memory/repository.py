from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import re
from sqlalchemy import and_

from .models import MemoryRecord, MemoryStatus, MemoryValidationError, utc_now


@dataclass(frozen=True)
class RetrievalDiagnostics:
    candidates: int
    selected: int
    deduplicated: int
    truncated: bool
    budget: int
    used: int


@dataclass(frozen=True)
class RetrievalResult:
    records: tuple
    diagnostics: RetrievalDiagnostics


class MemoryRepository:
    """Safe persistence and deterministic retrieval for narrative memories."""

    def __init__(self, session_factory, model):
        self._session_factory = session_factory
        self._model = model

    @staticmethod
    def _to_record(row) -> MemoryRecord:
        return MemoryRecord(
            id=row.id, story_id=row.story_id, category=row.category,
            content=row.content, confidence=row.confidence, origin=row.origin,
            episode_id=row.episode_id, scene_index=row.scene_index,
            status=MemoryStatus(row.status), created_at=row.created_at,
            updated_at=row.updated_at, supersedes_id=row.supersedes_id,
        )

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record.validate()
        session = self._session_factory()
        try:
            # A normalized content/category pair is idempotent within a story.
            existing = session.query(self._model).filter(
                self._model.story_id == record.story_id,
                self._model.category == record.category,
                self._model.content == record.content,
            ).first()
            if existing is not None:
                return self._to_record(existing)
            row = self._model(
                story_id=record.story_id, category=record.category,
                content=record.content, confidence=float(record.confidence),
                origin=record.origin, episode_id=record.episode_id,
                scene_index=record.scene_index, status=record.status.value,
                created_at=record.created_at, updated_at=utc_now(),
                supersedes_id=record.supersedes_id,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_record(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    save = add
    save_memory = add

    @staticmethod
    def validate(record: MemoryRecord) -> MemoryRecord:
        return record.validate()

    def retrieve(
        self, story_id: str, *, query: Optional[str] = None,
        categories: Optional[Iterable[str]] = None, budget: int = 4000,
        limit: int = 20,
    ) -> RetrievalResult:
        if not story_id or budget < 0 or limit < 0:
            raise MemoryValidationError("story_id, budget and limit are invalid")
        session = self._session_factory()
        try:
            q = session.query(self._model).filter(
                self._model.story_id == story_id,
                self._model.status == MemoryStatus.ACTIVE.value,
            )
            if categories:
                q = q.filter(self._model.category.in_(tuple(categories)))
            rows = q.all()
            terms = set(re.findall(r"\w+", (query or "").lower()))
            def rank(row):
                overlap = len(terms.intersection(re.findall(r"\w+", row.content.lower())))
                scene = row.scene_index if row.scene_index is not None else -1
                return (-overlap, -float(row.confidence), -scene, row.created_at, row.id)
            rows.sort(key=rank)
            unique, seen, used, truncated = [], set(), 0, False
            for row in rows:
                key = (row.category, " ".join(row.content.lower().split()))
                if key in seen:
                    continue
                cost = len(row.content)
                if len(unique) >= limit or used + cost > budget:
                    truncated = True
                    continue
                seen.add(key)
                unique.append(self._to_record(row))
                used += cost
            return RetrievalResult(
                tuple(unique),
                RetrievalDiagnostics(len(rows), len(unique), len(rows) - len(unique),
                                     truncated, budget, used),
            )
        finally:
            session.close()

    get_context = retrieve
    get_memory_context = retrieve

    def supersede(self, record_id: int, replacement: MemoryRecord) -> MemoryRecord:
        replacement.validate()
        session = self._session_factory()
        try:
            old = session.query(self._model).filter(self._model.id == record_id).one()
            if old.story_id != replacement.story_id:
                raise MemoryValidationError("supersession must remain story-scoped")
            old.status = MemoryStatus.SUPERSEDED.value
            row = self._model(
                story_id=replacement.story_id, category=replacement.category,
                content=replacement.content, confidence=replacement.confidence,
                origin=replacement.origin, episode_id=replacement.episode_id,
                scene_index=replacement.scene_index, status=MemoryStatus.ACTIVE.value,
                created_at=replacement.created_at, updated_at=utc_now(),
                supersedes_id=record_id,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_record(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    supersede_memory = supersede

    def find_conflicts(self, story_id: str, category: Optional[str] = None):
        """Return active records that disagree within a story/category."""
        session = self._session_factory()
        try:
            q = session.query(self._model).filter(
                self._model.story_id == story_id,
                self._model.status == MemoryStatus.ACTIVE.value,
            )
            if category:
                q = q.filter(self._model.category == category)
            rows = q.order_by(self._model.category, self._model.id).all()
            grouped = {}
            for row in rows:
                grouped.setdefault(row.category, []).append(row)
            return tuple(
                tuple(self._to_record(row) for row in group)
                for group in grouped.values()
                if len({row.content for row in group}) > 1
            )
        finally:
            session.close()

    def mark_conflicted(self, record_ids: Sequence[int]) -> None:
        session = self._session_factory()
        try:
            session.query(self._model).filter(self._model.id.in_(tuple(record_ids))).update(
                {self._model.status: MemoryStatus.CONFLICTED.value}, synchronize_session=False
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

from sqlalchemy import select
from .models import RevealConditions, SecretRecord, SecretStatus


class SecretRepository:
    def __init__(self, session_factory, model):
        self.Session, self.Model = session_factory, model

    def add(self, secret: SecretRecord) -> SecretRecord:
        secret.validate()
        with self.Session() as session:
            row = self.Model(
                story_id=secret.story_id, secret_id=secret.secret_id,
                protected_value=secret.protected_value, aliases_json="|".join(secret.aliases),
                min_scene=secret.min_scene,
                required_event=secret.conditions.required_event,
                character_knowledge_json="|".join(secret.conditions.character_knowledge),
                prerequisite_secret=secret.conditions.prerequisite_secret,
                explicit_instruction=secret.conditions.explicit_instruction,
                status=secret.status.value, revealed_scene=secret.revealed_scene,
            )
            session.add(row)
            session.commit()
            return self._to_record(row)

    def get(self, story_id: str, secret_id: str):
        with self.Session() as session:
            row = session.query(self.Model).filter_by(story_id=story_id, secret_id=secret_id).one_or_none()
            return self._to_record(row) if row else None

    def list(self, story_id: str):
        with self.Session() as session:
            return [self._to_record(row) for row in session.query(self.Model).filter_by(story_id=story_id).all()]

    @staticmethod
    def _to_record(row):
        if row is None:
            return None
        return SecretRecord(
            id=row.id, story_id=row.story_id, secret_id=row.secret_id,
            protected_value=row.protected_value,
            aliases=tuple(filter(None, (row.aliases_json or "").split("|"))),
            conditions=RevealConditions(row.min_scene, row.required_event,
                tuple(filter(None, (row.character_knowledge_json or "").split("|"))),
                row.prerequisite_secret, row.explicit_instruction),
            status=SecretStatus(row.status), revealed_scene=row.revealed_scene,
        )
